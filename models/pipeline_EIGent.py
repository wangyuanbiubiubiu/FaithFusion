from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings("ignore")
from typing import Literal, Optional
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers_dev import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintPipeline,
    CogVideoXI2VDualInpaintAnyLPipeline,
    FluxFillPipeline
)
import cv2
from diffusers_dev.utils import export_to_video, load_image, load_video
from PIL import Image
from io import BytesIO
import base64
from tqdm.auto import tqdm

@dataclass
class InferenceConfig:
    """All parameter configurations for video inference"""
    # Model path related
    model_path: str = "THUDM/CogVideoX-5b"
    inpainting_branch: Optional[str] = None
    
    # Output related
    image_or_video_path: str = ""
    
    # Inference parameters
    prompt: Optional[str] = None
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    num_videos_per_prompt: int = 1
    generate_type: Literal["t2v", "i2v", "v2v", "inpainting"] = "t2v"
    seed: int = 42
    dtype: torch.dtype = torch.bfloat16  # Note: Use torch type directly in dataclass
    height: int = 480
    width: int = 720
    strength: float = 1.0
    
    # Inpainting specific parameters
    inpainting_mask_meta: Optional[str] = None
    inpainting_sample_id: Optional[int] = None
    inpainting_frames: Optional[int] = None
    mask_background: bool = False
    first_frame_gt: bool = False
    overlap_frames: int = 0
    prev_clip_weight: float = 0.0
    replace_gt: bool = False
    mask_add: bool = True
    
    # Other configurations
    img_inpainting_model: Optional[str] = None
    llm_model: Optional[str] = None
    long_video: bool = False
    dilate_size: int = -1

    # Waymo related
    waymo_dir: str = None
    waymo_img_pair_dir: str = None
    caption_offline_dir: str = None
    gain_th: float = 1.0

def initialize_pipeline_raw(
    model_path: str,
    inpainting_branch: str = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Initialize CogVideoX inference Pipeline and return the configured pipeline object.

    Args:
        model_path: Path to pretrained model (e.g. "THUDM/CogVideoX-5b").
        inpainting_branch: Path to control branch model (use default branch if None).
        id_adapter_resample_learnable_path: LoRA weight path (used for ID adapter).
        dtype: Model computation precision (torch.bfloat16 or torch.float16).
        lora_rank: LoRA rank (only effective when loading LoRA).
        long_video: Whether to enable long video optimization (slice/chunk inference).

    Returns:
        CogVideoXI2VDualInpaintAnyLPipeline: Initialized inference pipeline.
    """
    print(f"Using specified inpainting branch: {inpainting_branch}")
    branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to(dtype=dtype).cuda()
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
            model_path,
            branch=branch,
            torch_dtype=dtype,
        )
        
    # Freeze modules that don't need training (accelerate inference)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)
    
    # Configure scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    return pipe

def _visualize_video(pipe, mask_background, original_video, video, masks, video_ref):
    
    original_video = pipe.video_processor.preprocess_video(original_video, height=video.shape[1], width=video.shape[2])
    masks = pipe.gain_video_processor.preprocess_video(masks, height=video.shape[1], width=video.shape[2])
    if mask_background:
        masked_video = original_video * (masks >= 0.8)
    else:
        masked_video = original_video * (masks < 0.8)
   
    original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
    
    masks = masks.squeeze(0).squeeze(0).numpy()
    masks = masks[..., np.newaxis].repeat(3, axis=-1)
    video_with_ref = []
    frame_height, frame_width = video.shape[1], video.shape[2]
    for i in range(len(video)):
        v_frame = video[i]  # Shape: (H, W, 3), value range 0~1
        v_arr = (v_frame * 255).astype(np.uint8)  # Convert to 0~255
        
        ref_frame = video_ref[i]
        ref_resized = ref_frame.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        ref_arr = np.array(ref_resized, dtype=np.uint8)  # Shape: (H, W, 3)
        
        # Generate mask for non-255 regions (regions where all three channels are not 255)
        non_255_mask = ~np.all(ref_arr == 255, axis=-1)  # Shape: (H, W)
        
        # Overlay non-255 regions onto video frame
        v_arr[non_255_mask] = ref_arr[non_255_mask]
        
        # Convert back to 0~1 array (keep consistent with original video format)
        v_frame_with_ref = v_arr.astype(np.float32) / 255.0
        video_with_ref.append(v_frame_with_ref)

    video_ = concatenate_images_horizontally(
        [original_video, masks, video_with_ref, video],
    )
    return video_

def concatenate_images_horizontally(images_list, output_type="np"):

    concatenated_images = []

    length = len(images_list[0])
    for i in range(length):
        tmp_tuple = ()
        for item in images_list:
            tmp_tuple += (np.array(item[i]), )

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate(tmp_tuple, axis=1)

        # Convert back to PIL Image
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images

def save_video(frames, output_path, fps=30):
    '''
    Save list of PIL image frames as video file
    Args:
        frames: List[Image.Image], List of PIL image frames
        output_path: str, Output video path
        fps: int, Video frame rate
    '''
    if not frames:
        print("No frames to save")
        return
    
    # Get dimensions of first frame
    width, height = frames[0].size
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Convert PIL images to OpenCV format and write to video
    for frame in frames:
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
    
    # Release resources
    out.release()


def save_frames_as_png(frames, output_folder, prefix='EIGent', start_idx=0):
    """
    Save video frames as PNG images
    
    Parameters:
    - frames: Video frame array with shape (number of frames, height, width, 3)
    - output_folder: Output folder path
    - prefix: Image filename prefix
    - start_idx: Start index
    
    Returns:
    - Number of successfully saved images
    """
    # Create output folder (if not exists)
    os.makedirs(output_folder, exist_ok=True)
    
    count = 0
    for i, frame in enumerate(frames):
        frame_idx = start_idx + i
        # if i == 0:
        #     frame_idx = start_idx + i
        # elif (i - 1) % 8 == 0:
        #     frame_idx = start_idx + int((i - 1) / 8) + 1
        # else:
        #     continue
        filename = f"{prefix}_{frame_idx}.png"
        file_path = os.path.join(output_folder, filename)
        frame = (frame * 255).astype(np.uint8)
        
        # Save frame as PNG image
        # Note: If frames are in RGB format, need to convert to BGR first (since OpenCV uses BGR by default)
        if frame.shape[-1] == 3:  # Check if there are 3 channels (RGB/BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(file_path, frame)
        count += 1
    
    return count

def infer_single_video_raw(
    pipe,
    video: np.ndarray,
    video_rendering: list,
    video_ref: list,
    output_name: str,
    raw_gains: np.ndarray,
    prompt: str,
    output_path: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    mask_background: bool = False,
    replace_gt: bool = False,
    mask_add: bool = False,
    overlap_frames: int = 0,
    prev_clip_weight: float = 0.0,
    inpainting_frames: int = None,
    gain_th: float = 0.1,
    strength: float = 1.0,
):
    """
    Perform inference on single video and save results.

    Args:
        pipe: Initialized CogVideoX Pipeline.
        video: Original video array with shape (number of frames, height, width, channels).
        video_rendering: List of video frames to process (PIL.Image format).
        video_ref: List of reference video frames (PIL.Image format).
        raw_gains: Gain mask array with shape matching video.
        prompt: Text prompt.
        output_path: Save path for generated video.
        num_inference_steps: Number of inference steps (higher = better quality, slower speed).
        guidance_scale: Guidance scale (higher = more prompt-following, lower = more creative).
        seed: Random seed (for reproducibility).
        mask_background: Whether to mask background.
        replace_gt: Whether to replace ground truth frames.
        mask_add: Whether to add mask.
        overlap_frames: Number of overlapping frames (for long video stitching).
        prev_clip_weight: Weight of previous clip.
        inpainting_frames: Target number of generated frames (video will be truncated if too long).
        gain_th: Gain threshold (for mask filtering).

    Returns:
        np.ndarray: Generated video frame array (shape: (number of frames, height, width, 3)).
    """
    frames, height, width, c = video.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_video_path = os.path.join(output_path, output_name + ".mp4")
    # print("path: ", output_name)

    # Process video frame count to meet model requirements
    if inpainting_frames is not None:
        if frames > inpainting_frames:
            begin_idx = 0
            end_idx = begin_idx + inpainting_frames
            raw_gains = raw_gains[begin_idx:end_idx]
            video = video[begin_idx:end_idx]
            video_rendering = video_rendering[begin_idx:end_idx]
            video_ref = video_ref[begin_idx:end_idx]
            frames = end_idx - begin_idx
        elif frames <= inpainting_frames:
            remainder = (3 + (frames % 4)) % 4
            if remainder != 0:
                video = video[:-remainder]
                video_rendering = video_rendering[:-remainder]
                video_ref = video_ref[:-remainder]
                raw_gains = raw_gains[:-remainder]
            frames = video.shape[0]

    # Convert gain mask to PIL format
    video_gain = np.repeat(raw_gains[:, :, :, np.newaxis], 3, axis=3)
    assert len(raw_gains) == len(video_rendering), "Mask and video frame count mismatch"
    video_gain = [Image.fromarray(video_gain[i]) for i in range(frames)]

    # Ensure video frames are in RGB format
    video = [Image.fromarray(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)) for i in range(frames)]
    video_rendering = [Image.fromarray(cv2.cvtColor(video_rendering[i], cv2.COLOR_BGR2RGB)) for i in range(frames)]
    video_ref = [Image.fromarray(cv2.cvtColor(video_ref[i], cv2.COLOR_BGR2RGB)) for i in range(frames)]

    # Process first frame (if ground truth first frame is needed)
    video_rendering[0] = video[0]
    gt_video_first_frame = video[0]

    if len(video) < frames:
        raise ValueError(f"Video length is less than {frames} frames, actual length: {len(video)}")

    # Process mask for first frame
    gt_mask_first_frame = video_gain[0]
    if mask_background:
        video_gain[0] = Image.fromarray(np.ones_like(np.array(video_gain[0])) * 255).convert("RGB")
    else:
        video_gain[0] = Image.fromarray(np.zeros_like(np.array(video_gain[0]))).convert("RGB")
    image = video_rendering[0]

    # Negative prompt (avoid low quality content)
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
        "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
        "fused fingers, still picture, messy background, many people in the background, walking backwards, "
        "unrealistic, implausible"
    )

    # Perform inference
    inpaint_outputs = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        video=video_rendering,
        masks=video_gain,
        ref_video=video_ref,
        strength=strength,
        replace_gt=replace_gt,
        mask_add=mask_add,
        stride=int(frames - overlap_frames),
        prev_clip_weight=prev_clip_weight,
        id_pool_resample_learnable=False,
        output_type="np"
    ).frames[0]

    video_generate = inpaint_outputs
    count = save_frames_as_png(inpaint_outputs, output_path)
    video_gain[0] = gt_mask_first_frame
    video[0] = gt_video_first_frame
    round_video = _visualize_video(pipe, mask_background, video[:len(video_generate)], video_generate, video_gain[:len(video_generate)], video_ref)
    export_to_video(round_video, output_video_path, fps=8)

def initialize_pipeline(config: InferenceConfig):
    """Initialize Pipeline with configuration class"""
    return initialize_pipeline_raw(  # Reuse previous core logic
        model_path=config.model_path,
        inpainting_branch=config.inpainting_branch,
        dtype=config.dtype,
    )

def infer_single_video(
    pipe,
    video: np.ndarray,
    video_rendering: list,
    video_ref: list,
    output_name: str,
    raw_gains: np.ndarray,
    config: InferenceConfig,  # Use configuration class instead of scattered parameters
):
    """Perform single video inference with configuration class"""
    return infer_single_video_raw(  # Reuse previous core logic
        pipe=pipe,
        video=video,
        video_rendering=video_rendering,
        video_ref=video_ref,
        output_name=output_name,
        raw_gains=raw_gains,
        prompt=config.prompt,
        output_path=config.image_or_video_path,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        seed=config.seed,
        mask_background=config.mask_background,
        replace_gt=config.replace_gt,
        mask_add=config.mask_add,
        overlap_frames=config.overlap_frames,
        prev_clip_weight=config.prev_clip_weight,
        inpainting_frames=config.inpainting_frames,
        gain_th=config.gain_th,
        strength=config.strength
    )