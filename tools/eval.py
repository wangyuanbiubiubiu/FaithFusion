from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
from torch import Tensor
import shutil

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views,
    render_train_views_uncertainty,
    render_novel_views_uncertainty
)
from PIL import Image
import numpy as np
import imageio

from models.waymo_dataset_img_pair_parallel import WaymoImgPairDataset
from models.pipeline_EIGent import infer_single_video

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def load_gain_as_tensor(path, gain_th = 1.0):
    if path.lower().endswith((".pt", ".pth")):
        g = torch.load(path, map_location="cpu")
        if not isinstance(g, torch.Tensor):
            g = torch.as_tensor(g)
        g = g.detach().float()
        no_opacity = g > 99.0
        g = torch.clamp(g, max=float(gain_th)) / float(gain_th)
        g[no_opacity] = 1.0
        return g.unsqueeze(0)

def generate_segment_list(start, end, step, need_append_last=True):
    # 生成从start到end(含)的列表，步长为step
    seg_list = list(range(start, end + 1, step))
    
    # 如果提供了last_item且不在列表中，则添加到末尾
    if need_append_last and end not in seg_list:
        seg_list.append(end)
    
    return seg_list

def prepare_EIGent_render_data(trainer, dataset, dist_max=0.0):
    render_data = []
    camera_downscale = 1.0

    step = 40
    seg_list = generate_segment_list(start=0, end=39, step=step, need_append_last=False)
    for seg_begin in seg_list:
        num_x_dist = 49 - step
        mock_dist = dist_max / num_x_dist
        for i in range(51):
            if i < num_x_dist:
                seg = seg_begin
                mock_times = i
            elif i < 49:
                seg = i - num_x_dist + seg_begin
                if seg > 39:
                    seg = 39
                mock_times = num_x_dist
            else:
                seg = seg_begin + step - 1
                if seg > 39:
                    seg = 39
                mock_times = num_x_dist
            image_infos, cam_infos = dataset.get_image_mock(seg, camera_downscale, mock_times, mock_dist)
            render_data.append({
                "cam_infos": cam_infos,
                "image_infos": image_infos,
            })
    return render_data

def prepare_difix_render_data(trainer, dataset, dist_max=0.0):
    render_data = []
    camera_downscale = trainer._get_downscale_factor()
    for i in range(len(dataset.split_indices)):
        seg = i
        mock_times = 1
        mock_dist = dist_max
        image_infos, cam_infos = dataset.get_image_mock(seg, camera_downscale, mock_times, mock_dist)
        image_infos['diffusion_img_idx'] = torch.full(image_infos['img_idx'].shape, 0, dtype=torch.long)
        render_data.append({
            "cam_infos": cam_infos,
            "image_infos": image_infos,
        })
    return render_data

def prepare_test_render_data(trainer, dataset):
    render_data = []
    camera_downscale = trainer._get_downscale_factor()
    for i in range(len(dataset)):
        image_infos, cam_infos = dataset.get_image(i, camera_downscale)
        render_data.append({
            "cam_infos": cam_infos,
            "image_infos": image_infos,
        })
    return render_data

def do_fix(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True,
    difixer=None,
    depth_anything=None,
    EIGenter=None,
    EIGenter_config=None
):
    trainer.set_eval()
    
    next_step = step + 1
    start_step = cfg.diffusion.difix_start_step
    fix_inter = cfg.diffusion.difix_fix_interval
    fixer_type = cfg.diffusion.fix_model
    is_expansion = True
    if next_step >= start_step:
        if (next_step - start_step) % fix_inter == 0:
            difix_epoch_used_time = int(min(int(((next_step - start_step) / fix_inter) + 1), cfg.diffusion.epoch_used_times))
            difix_output_dist = difix_epoch_used_time * cfg.diffusion.dist_max
            if difix_epoch_used_time > 1:
                difix_extra_used_time = int(min(int(((next_step - start_step) / fix_inter)), cfg.diffusion.epoch_used_times))
                extra_dist = difix_extra_used_time * cfg.diffusion.dist_max
                if difix_extra_used_time == cfg.diffusion.epoch_used_times:
                    is_expansion = False
                    if not is_expansion:
                        if (next_step - start_step) % (3 * fix_inter) != 0:
                            return
            else:
                extra_dist = None
        else:
            return
    else:
        return
    
    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        traj_type = "lane_shift_diffuison"
        video_output_dir = f"{cfg.log_dir}/{traj_type}/dist_{difix_epoch_used_time}"
        # Delete this folder first
        backup_dir = f"{video_output_dir}_bk_step_{next_step}"
        if os.path.exists(video_output_dir):
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            shutil.copytree(video_output_dir, backup_dir)
            shutil.rmtree(video_output_dir)
        
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        
        if fixer_type == "difix":
            render_data = prepare_difix_render_data(trainer, dataset.train_image_set, difix_output_dist)
        elif fixer_type == "EIGent":
            render_train_views_uncertainty(trainer=trainer, dataset=dataset.train_image_set, extra_dist=extra_dist)
            render_data = prepare_EIGent_render_data(trainer, dataset.train_image_set, difix_output_dist)
        
        # Render and save video
        save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
        render_novel_views_uncertainty(
            trainer, render_data, save_path,
            fps=render_novel_cfg.get("fps", cfg.render.fps),
            compute_uncertainty=(fixer_type == "EIGent")
        )
        logger.info(f"Saved novel view video for trajectory type: {traj_type} to {save_path}")
        
        trainer.set_eval() #compute uncertainty will set it to train
        if fixer_type == "difix":
            for i in range(len(dataset.train_indices)):
                gt_image_path = os.path.join(video_output_dir, f"gt_{i}.png")
                render_image_path = os.path.join(video_output_dir, f"rgb_{i}.png")
                difix_image_path = os.path.join(video_output_dir, f"difix_{i}.png")
                sky_image_path = os.path.join(video_output_dir, f"skymask_{i}.png")

                gt_image = Image.open(gt_image_path).convert("RGB")
                render_image = Image.open(render_image_path).convert("RGB")
                origin_size = render_image.size
                target_wh = (1024, 576)
                gt_image = gt_image.resize(target_wh, Image.LANCZOS)
                render_image = render_image.resize(target_wh, Image.LANCZOS)
                difix_image = difixer(prompt="remove degradation", image=render_image, ref_image=gt_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
                difix_image = difix_image.resize(origin_size, Image.LANCZOS)
                difix_image.save(difix_image_path)
                
                difix_image_np = np.array(Image.open(difix_image_path))
                depth = depth_anything.infer_image(difix_image_np, 518)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth_sky_mask = (depth == 0)
                imageio.imwrite(sky_image_path, depth_sky_mask.astype(np.uint8) * 255)
        elif fixer_type == "EIGent":
            use_VDM = cfg.diffusion.use_VDM
            if not is_expansion:
                use_VDM = False
           
            print("use_VDM: ", use_VDM)

            for i in range(cfg.diffusion.base_render_per_frame):
                gt_image_path = os.path.join(video_output_dir, f"gt_{i}.png")
                render_image_path = os.path.join(video_output_dir, f"rgb_{i}.png")
                difix_image_path = os.path.join(video_output_dir, f"difix_{i}.png")
                gain_pt_path = os.path.join(video_output_dir, f"gain_{i}.pt")
                gt_image = Image.open(gt_image_path).convert("RGB")
                render_image = Image.open(render_image_path).convert("RGB")
                gain_img = load_gain_as_tensor(gain_pt_path)
                prompt = "remove degradation"
                ref_gain_img = torch.zeros_like(gain_img)
                difix_image = difixer.sample(
                    render_image, 
                    height=576, 
                    width=1024, 
                    ref_image=gt_image,
                    gain=gain_img, 
                    ref_gain=ref_gain_img, 
                    prompt=prompt
                )
                difix_image = difix_image.resize(render_image.size, Image.LANCZOS)
                difix_image.save(difix_image_path)
                if not use_VDM:
                    pifix_image_path = os.path.join(video_output_dir, f"EIGent_{i}.png")
                    difix_image.save(pifix_image_path)

            if use_VDM:
                EIGenter_config.waymo_img_pair_dir = video_output_dir
                EIGenter_config.image_or_video_path = video_output_dir
                valid_scene_list = [
                    {"scene_id": int(dataset.pixel_source.data_path.split("/")[-1]), "pair_name": "cam0_" + str(difix_output_dist), 
                        "start_idx_list": [0], "data_type": "valid", "video_out": False, "gain_th": EIGenter_config.gain_th}
                ]
                
                tmp_dataset = WaymoImgPairDataset(
                    base_dir=EIGenter_config.waymo_dir,
                    img_pair_dir=EIGenter_config.waymo_img_pair_dir,
                    caption_offline_dir=EIGenter_config.caption_offline_dir,
                    num_frames=EIGenter_config.inpainting_frames,
                    max_start_frame=100,
                    point_size=2,
                    output_dir=EIGenter_config.image_or_video_path,
                    cam_id=[0,1,2,3,4],
                    traj_mode="first_ori",
                    scene_list=valid_scene_list,
                    pair_list=valid_scene_list,
                    height=EIGenter_config.height,
                    width=EIGenter_config.width
                )

                res = tmp_dataset[0]
                infer_single_video(
                    pipe=EIGenter,
                    video=res["video"],
                    video_rendering=res["video_rendering"],
                    video_ref=res["video_ref"],
                    output_name=res["path"],
                    raw_gains=res["raw_gain"],
                    config=EIGenter_config,
                )
            pre_frame = 9
            for fake_i in range(len(dataset.train_indices)):
                i = fake_i + pre_frame
                render_image_path = os.path.join(video_output_dir, f"rgb_{i}.png")
                EIGent_image_path = os.path.join(video_output_dir, f"EIGent_{i}.png")
                sky_image_path = os.path.join(video_output_dir, f"skymask_{i}.png")

                render_image = Image.open(render_image_path).convert("RGB")
                EIGent_image =  Image.open(EIGent_image_path).convert("RGB")
                EIGent_image = EIGent_image.resize(render_image.size, Image.LANCZOS)
                EIGent_image.save(EIGent_image_path)
                
                difix_image_np = np.array(EIGent_image)
                depth = depth_anything.infer_image(difix_image_np, 518)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth_sky_mask = (depth == 0)
                imageio.imwrite(sky_image_path, depth_sky_mask.astype(np.uint8) * 255)

# @torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
        
    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            full_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    
    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        render_traj = dataset.get_novel_render_traj(
            traj_types=render_novel_cfg.traj_types,
            target_frames=render_novel_cfg.get("frames", dataset.frame_num),
        )
        video_output_dir = f"{cfg.log_dir}/videos{post_fix}/novel_{step}"
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        render_EIG = render_novel_cfg.get("render_EIG", False)
        if render_EIG:
            #compute total EIG
            if "switch_camera" in render_traj:
                if len(render_traj) != 1 :
                    raise ValueError("switch_camera only support trans cam")
                trainer.initialize_optimizer_uncertainty()
            render_train_views_uncertainty(trainer=trainer, dataset=dataset.train_image_set)
        for traj_type, traj in render_traj.items():
            # Prepare rendering data
            if traj_type == "lane_shift":
                EIGent_output_dist = render_novel_cfg["lane_shift_dist"]
                render_data = prepare_EIGent_render_data(trainer, dataset.train_image_set, EIGent_output_dist)
            elif traj_type == "switch_camera":
                render_data = prepare_test_render_data(trainer, dataset.test_image_set)
            else:
                render_data = dataset.prepare_novel_view_render_data(traj)
            
            # Render and save video
            save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
            if render_EIG:
                save_img_path = os.path.join(video_output_dir, traj_type, f"{traj_type}.mp4")
                if os.path.exists(os.path.join(video_output_dir, traj_type)):
                    shutil.rmtree(os.path.join(video_output_dir, traj_type))
                if not os.path.exists(os.path.join(video_output_dir, traj_type)):
                    os.makedirs(os.path.join(video_output_dir, traj_type))
                render_novel_views_uncertainty(
                    trainer, render_data, save_img_path,
                    fps=render_novel_cfg.get("fps", cfg.render.fps)
                )
            else:
                render_novel_views(
                    trainer, render_data, save_path,
                    fps=render_novel_cfg.get("fps", cfg.render.fps)
                )
            logger.info(f"Saved novel view video for trajectory type: {traj_type} to {save_path}")
            
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'diffusion' in cfg and cfg.diffusion.diffusion_fusion:
        cfg.diffusioncheck_diffusion_depth = False
        cfg.data["diffusion"] = cfg.diffusion
        if "gain_folder" not in cfg.data["diffusion"]:
            cfg.data["diffusion"]["gain_folder"] = os.path.join(os.path.dirname(args.resume_from), "switch_camera", "cam0_to_cam0")
        diffusion_fusion = cfg.diffusion.diffusion_fusion
    else:
        diffusion_fusion = False
            
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
        diffusion_used_times=len(dataset.train_indices) - len(dataset.train_timesteps),
    )
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
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
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval"
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)