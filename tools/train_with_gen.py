from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import os

from models.pipeline_EIGent import InferenceConfig, initialize_pipeline, infer_single_video
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EIGenter_config = None
EIGenter = None
difixer = None
depth_anything = None

import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from tools.eval import do_evaluation, do_fix
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def diffusion_fusion_vis(outputs, image_infos, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Process GT and Rendered Images
    gt_tensor = image_infos["pixels"].detach().cpu().clip(min=0, max=1)  # [H, W, 3]
    render_tensor = outputs["rgb"].detach().cpu().clip(min=0, max=1)      # [H, W, 3]
    gt_image = Image.fromarray((gt_tensor.numpy() * 255).astype(np.uint8))
    render_image = Image.fromarray((render_tensor.numpy() * 255).astype(np.uint8))
    target_size = render_image.size  # Standardize size: (width, height)

    # 2. Process gain_map: Divide into intervals and map to grayscale values (0-255)
    gain_map = outputs["gain_map"].detach().cpu()  # [H, W]
    opacity_filter = outputs["opacity"] < 0.1      # Opacity filter
    gain_map[opacity_filter[:, :, 0].cpu()] = 100.0  # Set filtered regions to a large value (falls into the ≥5.0 interval)

    # Define interval boundaries (as required: <0.01, 0.01~0.1, 0.1~5.0, ≥5.0)
    bounds = [0.1, 0.5, 5.0]
    # Initialize grayscale values (0-255, 4 intervals correspond to 4 uniform grayscale levels)
    gain_gray = torch.zeros_like(gain_map, dtype=torch.uint8)  # Single channel grayscale image (0-255)
    
    # Map intervals to grayscale values (0=black, 255=white, uniformly distributed in between)
    gain_gray[gain_map < bounds[0]] = 0        # <0.01 → Black (0)
    gain_gray[(gain_map >= bounds[0]) & (gain_map < bounds[1])] = 85  # 0.01~0.1 → Dark Gray (85)
    gain_gray[(gain_map >= bounds[1]) & (gain_map < bounds[2])] = 170 # 0.1~5.0 → Medium Gray (170)
    gain_gray[gain_map >= bounds[2]] = 255     # ≥5.0 → White (255)

    # Convert to single channel grayscale image (PIL 'L' mode)
    gain_image = Image.fromarray(gain_gray.numpy(), mode='L')

    # 3. Resize gain image and combine
    gain_image = gain_image.resize(target_size, Image.Resampling.LANCZOS)
    combined_image = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    combined_image.paste(gt_image, (0, 0))
    combined_image.paste(render_image, (target_size[0], 0))
    combined_image.paste(gain_image, (target_size[0] * 2, 0))  # Grayscale image pasted into RGB image will be automatically converted to three-channel grayscale

    # 4. Save combined image
    combined_path = os.path.join(output_dir, f"combined_{step}.png")
    combined_image.save(combined_path)

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./", 
        ["configs", "datasets", "models", "utils", "tools"], 
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

def initialize_models(cfg):
    """Initialize corresponding models based on command line arguments"""
    global EIGenter_config, EIGenter, difixer, depth_anything
    from depth_anything_v2.dpt import DepthAnythingV2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs["vitl"])
    depth_anything.load_state_dict(torch.load(
        f'depth_anything_v2/pretrained/depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    logger.info("Depth model initialized successfully")
    
    # Initialize inpainting model
    if cfg.diffusion.fix_model == "EIGent":
        if cfg.diffusion.use_VDM:
            # Define configuration first, initialize model on demand later
            EIGenter_config = InferenceConfig(
                prompt = "A sequence of street view frames where the camera moves horizontally (left or right or forward) in a static world. All vehicles present in the scene (including cars, trucks, motorcycles, and other motorized/non-motorized vehicles) must remain completely stationary—no positional shifts, motion blurs, or frame-to-frame changes in their posture, orientation, or location. Ensure no ghost objects such as fake obstacles, incorrect shadows, or misplaced pixels. Background elements like buildings, road markings, trees, and stationary vehicles should remain fixed except for natural positional shifts due to camera movement. Output clean, artifact-free frames with realistic and consistent street reconstruction, and maintain visual coherence of all static elements (especially vehicles) across the entire sequence.",
                model_path="ckpt/CogVideoX-5b-I2V",
                inpainting_branch="control_ckpt/waymo_gain_drop_augref_decouple/checkpoint/branch",
                num_inference_steps=50,
                generate_type="i2v_inpainting",
                inpainting_frames=cfg.diffusion.infer_frame_num,
                waymo_dir="data/waymo/processed",
                gain_th=cfg.diffusion.gain_th,
                seed=43,
                replace_gt=False,
                guidance_scale=0.0,
                strength=1.0
            )

            EIGenter = initialize_pipeline(EIGenter_config)
            logger.info("EIGent model initialized successfully")
        from Difix3D_src.model import Difix
        difixer = Difix(
            pretrained_name="fusion_certainty/model_last.pkl",
            pretrained_path="fusion_certainty/model_last.pkl",
            timestep=199,
            mv_unet=True,
            use_gain=True
        )
        difixer.to("cuda")
        logger.info("Difix model initialized successfully")
    elif cfg.diffusion.fix_model == "difix":
        from models.pipeline_difix import DifixPipeline
        os.environ["XFORMERS_DISABLE_CUTLASS"] = "1"
        difixer = DifixPipeline.from_pretrained("difix_ref", trust_remote_code=True)
        difixer.set_progress_bar_config(disable=True)
        difixer.to("cuda")
        logger.info("Difix model initialized successfully")
    else:
        raise ValueError("The diffusion model must be used")

def main(args):    
    cfg = setup(args)
    cfg.data.log_dir = cfg.log_dir
    if 'diffusion' in cfg and cfg.diffusion.diffusion_fusion:
        cfg.data["diffusion"] = cfg.diffusion
        diffusion_fusion = cfg.diffusion.diffusion_fusion
    else:
        diffusion_fusion = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
        diffusion_fusion=diffusion_fusion,
        diffusion_used_times=len(dataset.train_timesteps),
    )
    
    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    # setup optimizer  
    trainer.initialize_optimizer()
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    
    # Initialize diffusion models
    initialize_models(cfg)
    
    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    # do_fix(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    #     difixer=difixer,
    #     depth_anything=depth_anything,
    #     EIGenter=EIGenter,
    #     EIGenter_config=EIGenter_config
    # )
    # raise NotImplementedError

    last_gain_step = 0
    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                )
            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                    }
                )
            if not args.skip_video_output:
                vis_frame_dict = save_videos(
                    render_results,
                    save_pth=os.path.join(
                        cfg.log_dir, "images", f"step_{step}.png"
                    ),  # don't save the video
                    layout=dataset.layout,
                    num_timestamps=1,
                    keys=render_keys,
                    save_seperate_video=cfg.logging.save_seperate_video,
                    num_cams=dataset.pixel_source.num_cams,
                    fps=cfg.render.fps,
                    verbose=False,
                )
                if args.enable_wandb:
                    for k, v in vis_frame_dict.items():
                        wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()
                
        
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad() # zero grad
        
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next_with_diffusion(train_step_camera_downscale, step, cfg.diffusion.fix_model, cfg)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        
        # forward & backward
        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
        )
        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        trainer.backward(loss_dict)
        
        # after training step
        trainer.postprocess_per_train_step(step=step)
        
        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            if (step - last_gain_step) > cfg.diffusion.vis_freq and cfg.diffusion.vis_freq > 0:
                if "gain" in image_infos and image_infos['gain'] is not None:
                        outputs["gain_map"] = image_infos['gain'].clone()
                        diffusion_fusion_vis(outputs, image_infos, os.path.join(cfg.log_dir,"diffusion_fuse"), step)
                        last_gain_step = step

            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        )
        if do_save:  
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )
        
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")
        
        do_fix(
            step=step,
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            render_keys=render_keys,
            args=args,
            difixer=difixer,
            depth_anything=depth_anything,
            EIGenter=EIGenter,
            EIGenter_config=EIGenter_config
        )
    
    logger.info("Training done!")

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    parser.add_argument("--skip_video_output", action="store_true", help="skip video output")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    final_step = main(args)
