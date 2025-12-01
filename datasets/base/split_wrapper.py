from typing import List, Tuple
import torch
import random
import os
from PIL import Image
import numpy as np
import imageio

from .pixel_source import ScenePixelSource, sparse_lidar_map_downsampler

import open3d as o3d
        
class SplitWrapper(torch.utils.data.Dataset):

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: ScenePixelSource,
        split_indices: List[int] = None,
        split: str = "train",
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.split = split
        self.fix_init = False

    def get_image(self, idx, camera_downscale) -> dict:
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(self.split_indices[idx])
        self.datasource.reset_downscale_factor()
        return image_infos, cam_infos

    def get_image_mock(self, idx, camera_downscale, mock_id, mock_dist) -> dict:
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image_mock(self.split_indices[idx], mock_id, mock_dist)
        self.datasource.reset_downscale_factor()
        return image_infos, cam_infos

    def proj_depth_img(self, image_infos, cam_infos):
        # static_proj
        # 1. Process intrinsics and extrinsics (maintain torch tensor operations)
        # Pad intrinsics into a 4x4 matrix
        intrinsic_4x4 = torch.nn.functional.pad(
            cam_infos["intrinsics"], (0, 1, 0, 1)
        )
        intrinsic_4x4[3, 3] = 1.0

        
        # Lidar to camera transformation matrix (extrinsics)
        lidar2cam = cam_infos["camera_to_world"].inverse()
        # Point cloud transformation from world coordinates to camera coordinates (N, 3)
        points_cam = (
            lidar2cam[:3, :3] @ self.static_pts.T + lidar2cam[:3, 3:4]
        ).T  # (num_pts, 3)

        # 2. Filter points behind the camera (z > 1e-6)
        valid_depth_mask = points_cam[:, 2] > 1e-6
        points_cam = points_cam[valid_depth_mask]
        if points_cam.numel() == 0:  # No valid points
            h, w = cam_infos["height"], cam_infos["width"]
            return torch.zeros((h, w, 3), dtype=torch.uint8, device=points_cam.device), \
                torch.full((h, w), torch.inf, device=points_cam.device)
        
        # 3. Perspective projection to calculate pixel coordinates (u, v)
        fx, fy = intrinsic_4x4[0, 0], intrinsic_4x4[1, 1]
        cx, cy = intrinsic_4x4[0, 2], intrinsic_4x4[1, 2]
        z_cam = points_cam[:, 2]  # Depth value (z in camera coordinates)
        
        # Calculate u and v (maintain torch operations, avoid converting to numpy)
        u = torch.round((fx * points_cam[:, 0] / z_cam) + cx).to(torch.long)
        v = torch.round((fy * points_cam[:, 1] / z_cam) + cy).to(torch.long)
        
        # 4. Filter pixels outside the image boundary
        h, w = cam_infos["height"], cam_infos["width"]
        img_bound_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z_cam = u[img_bound_mask], v[img_bound_mask], z_cam[img_bound_mask]
        if u.numel() == 0:  # No valid points after filtering
            return torch.zeros((h, w, 3), dtype=torch.uint8, device=u.device), \
                torch.full((h, w), torch.inf, device=u.device)
        
        # 5. Z-buffer core: keep the closest point for each pixel (PyTorch version)
        # Generate 1D pixel indices (v*w + u)
        pixel_indices = v * w + u
        
        # Sort by depth in ascending order (near → far)
        sorted_idx = torch.argsort(z_cam)
        sorted_u = u[sorted_idx]
        sorted_v = v[sorted_idx]
        sorted_z = z_cam[sorted_idx]
        sorted_pixel_indices = pixel_indices[sorted_idx]
        
        # Keep the first occurrence for each pixel (closest point)
        # Use torch.unique with return_inverse and return_counts to find the first occurrence index
        unique_indices, inverse_indices, counts = torch.unique(
            sorted_pixel_indices, return_inverse=True, return_counts=True
        )
        # Construct a mask for the first occurrence (first element of each group)
        first_occurrence_mask = torch.zeros_like(sorted_pixel_indices, dtype=torch.bool)
        # Calculate the starting index of each group (cumulative count)
        cumsum_counts = torch.cat([torch.tensor([0], device=counts.device), counts.cumsum(dim=0)[:-1]])
        first_occurrence_mask[cumsum_counts] = True
        # Extract valid points
        final_u = sorted_u[first_occurrence_mask]
        final_v = sorted_v[first_occurrence_mask]
        final_z = sorted_z[first_occurrence_mask]
        
        # 6. Generate depth map and color projection map (PyTorch tensor operations)
        # Initialize depth map and projection map
        depth_map = torch.zeros(cam_infos["height"], cam_infos["width"]).to(self.static_pts.device)     
        # Fill depth map and projection map (using tensor indexing for batch assignment)
        depth_map[final_v, final_u] = final_z

        # dynamic proj
        # sort_depth (render from far to near)
        instance_ids = list(self.dynamic_pts_dic.keys())
        instance_depth_list = []
        
        frame_idx = int(image_infos["frame_idx"][0][0])
        for instance_id in instance_ids:
            is_valid_instance = self.datasource.per_frame_instance_mask[frame_idx, instance_id]
            if not is_valid_instance:
                continue
            # get the pose of the instance at the given frame
            o2w = self.datasource.instances_pose[frame_idx, instance_id]
            o2c = cam_infos["camera_to_world"].inverse() @ o2w
            instance_depth_list.append((instance_id, abs(o2c[2, 3])))

        sorted_instances = sorted(instance_depth_list, key=lambda x: x[1], reverse=True)
        sorted_instance_ids = [item[0] for item in sorted_instances]
        for instance_id in sorted_instance_ids:
            o2w = self.datasource.instances_pose[frame_idx, instance_id]
            o2c = cam_infos["camera_to_world"].inverse() @ o2w
            obj2img = intrinsic_4x4 @ o2c
            
            fake_pts = self.dynamic_fake_pts_dic[instance_id]  # Point cloud in world coordinates
            fake_img_points = (
                obj2img[:3, :3] @ fake_pts.T + obj2img[:3, 3:4] ).T # (num_pts, 3)
        
            fake_depth = fake_img_points[:, 2]  # Depth in camera coordinates (N,)
            fake_cam_points = fake_img_points[:, :2] / (fake_depth.unsqueeze(-1) + 1e-6)  # (N, 2)

            valid_fake = (
                (fake_cam_points[:, 0] >= 0) & (fake_cam_points[:, 0] < cam_infos["width"]) &
                (fake_cam_points[:, 1] >= 0) & (fake_cam_points[:, 1] < cam_infos["height"]) &
                (fake_depth > 0)
            )
            valid_fake_pts = fake_cam_points[valid_fake].long()  # Valid pixel coordinates (M, 2), converted to integers
            M = valid_fake_pts.shape[0]

            if M > 0:
                # Generate 11x11 neighborhood offsets (±5 pixels around the center, including -5 to +5)
                offsets = torch.arange(-7, 6, device=self.datasource.device)  # [-5,-4,-3,-2,-1,0,1,2,3,4,5], total 11 values
                dx, dy = torch.meshgrid(offsets, offsets, indexing="xy")  # (11,11) grid offsets
                dx = dx.flatten()  # (121,) flattened to 1D offsets
                dy = dy.flatten()  # (121,) flattened to 1D offsets
                # # Generate 5x5 neighborhood offsets, used to draw the depth of the vehicle background that needs to be masked
                # offsets = torch.arange(-2, 3, device=self.datasource.device)  # [-2,-1,0,1,2]
                # dx, dy = torch.meshgrid(offsets, offsets, indexing="xy")  # (5,5) grid offsets
                # dx = dx.flatten()  # (25,)
                # dy = dy.flatten()  # (25,)

                # Batch calculate all neighborhood pixel coordinates (M*25, 2)
                neighbor_x = valid_fake_pts[:, 0].unsqueeze(1) + dx.unsqueeze(0)  # (M,25)
                neighbor_y = valid_fake_pts[:, 1].unsqueeze(1) + dy.unsqueeze(0)  # (M,25)
                neighbor_x = neighbor_x.flatten()
                neighbor_y = neighbor_y.flatten()

                # Filter pixels in the neighborhood that are outside the image boundaries
                valid_neighbor = (
                    (neighbor_x >= 0) & (neighbor_x < cam_infos["width"]) &
                    (neighbor_y >= 0) & (neighbor_y < cam_infos["height"])
                )
                neighbor_x = neighbor_x[valid_neighbor]
                neighbor_y = neighbor_y[valid_neighbor]

                # Batch set neighborhood pixels to 0 (vectorized index assignment)
                depth_map[neighbor_y, neighbor_x] = 0.0

            real_pts = self.dynamic_pts_dic[instance_id]
            real_img_points = (
                obj2img[:3, :3] @ real_pts.T + obj2img[:3, 3:4] ).T # (num_pts, 3)
        
            real_depth = real_img_points[:, 2]  # Depth in camera coordinates (N,)
            real_cam_points = real_img_points[:, :2] / (real_depth.unsqueeze(-1) + 1e-6)  # (N, 2)

            valid_real = (
                (real_cam_points[:, 0] >= 0) & (real_cam_points[:, 0] < cam_infos["width"]) &
                (real_cam_points[:, 1] >= 0) & (real_cam_points[:, 1] < cam_infos["height"]) &
                (real_depth > 0)
            )

            valid_real_pts = real_cam_points[valid_real]
            depth_map[
                valid_real_pts[:, 1].long(), valid_real_pts[:, 0].long() ] = real_depth[valid_real].squeeze(-1)
        return depth_map
    
    def next(self, camera_downscale) -> Tuple[dict, dict]:
        assert self.split == "train", "Only train split supports next()"
        
        img_idx = self.datasource.propose_training_image(
            candidate_indices=self.split_indices
        )
        
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(img_idx)
        self.datasource.reset_downscale_factor()
        
        return image_infos, cam_infos
    
    def next_with_diffusion(self, camera_downscale, step, fix_type="difix", cfg=None) -> Tuple[dict, dict]:
        assert self.split == "train", "Only train split supports next()"
        
        img_idx = self.datasource.propose_training_image(
            candidate_indices=self.split_indices
        )

        start_step = cfg.diffusion.difix_start_step
        fix_inter = cfg.diffusion.difix_fix_interval
        need_shift = False
        if step >= start_step:
            difix_epoch_used_time_cur = int(min(
                ((step - start_step) // fix_inter) + 1,  # Use // for integer division to avoid float
                cfg.diffusion.epoch_used_times  # Upper limit is 6
            ))
            first_init=False
            if not self.fix_init:
                static_ply_path = f"{cfg.log_dir}/aggregate_static_world_pts/aggregate_static_world_pts.ply"    
                static_pcd = o3d.io.read_point_cloud(static_ply_path)
                self.static_pts = torch.tensor(np.asarray(static_pcd.points), dtype=torch.float32, device=self.datasource.device)
                self.dynamic_pts_dic = {}
                self.dynamic_fake_pts_dic = {}
                dynamic_ply_folder = f"{cfg.log_dir}/aggregated_instance_lidar_pts"
                for ins_id_ply in os.listdir(dynamic_ply_folder):
                    if "fake" in ins_id_ply:
                        continue
                    ins_pcd = o3d.io.read_point_cloud(os.path.join(dynamic_ply_folder, ins_id_ply))
                    ins_id = int(ins_id_ply.split(".ply")[0])
                    fake_pcd = o3d.io.read_point_cloud(os.path.join(dynamic_ply_folder, f"{ins_id}_fake.ply"))
                    self.dynamic_pts_dic[ins_id] = torch.tensor(np.asarray(ins_pcd.points), dtype=torch.float32, device=self.datasource.device)
                    self.dynamic_fake_pts_dic[ins_id] = torch.tensor(np.asarray(fake_pcd.points), dtype=torch.float32, device=self.datasource.device)
                self.fix_init = True
                first_init = True
                self.fusion_gain = {}

            if (step - start_step) % fix_inter == 0 or first_init:
                difix_epoch_used_time_list = [difix_epoch_used_time_cur]
                
                for difix_epoch_used_time in difix_epoch_used_time_list:
                    #re-init 
                    self.diffusion_rgb = {}
                    self.diffusion_sky_mask = {}
                    self.diffusion_depth_map = {}
                    self.fusion_gain = {}

                    video_output_dir = f"{cfg.log_dir}/lane_shift_diffuison/dist_{difix_epoch_used_time}"

                    if fix_type == "difix":
                        pre_frame = 0
                    else:
                        pre_frame = 9
                    
                    for img_idx in self.split_indices:
                        if img_idx not in self.diffusion_rgb:
                            self.diffusion_rgb[img_idx] = {}
                            self.diffusion_sky_mask[img_idx] = {}
                            self.diffusion_depth_map[img_idx] = {}
                            self.fusion_gain[img_idx] = {}

                        if fix_type == "difix":
                            difix_image_path = os.path.join(video_output_dir, f"difix_{img_idx + pre_frame}.png")
                        else:
                            difix_image_path = os.path.join(video_output_dir, f"EIGent_{img_idx + pre_frame}.png")
                            gain_path = os.path.join(video_output_dir, f"gain_{img_idx + pre_frame}.pt")
                            gain_tensor = torch.load(gain_path, weights_only=True, map_location="cpu")
                            self.fusion_gain[img_idx][difix_epoch_used_time] = gain_tensor.to(self.datasource.device)
        
                        sky_image_path = os.path.join(video_output_dir, f"skymask_{img_idx + pre_frame}.png")
                        infer_rgb = Image.open(difix_image_path).convert("RGB")
                        infer_rgb = (torch.from_numpy(np.array(infer_rgb)) / 255)
                        self.diffusion_rgb[img_idx][difix_epoch_used_time] = infer_rgb.to(self.datasource.device)
                        sky_mask = Image.open(sky_image_path).convert("L")
                        sky_mask = (torch.from_numpy(np.array(sky_mask) > 0)).float()
                        self.diffusion_sky_mask[img_idx][difix_epoch_used_time] = sky_mask.to(self.datasource.device)

                        if difix_epoch_used_time not in self.diffusion_depth_map[img_idx]:
                            tmp_image_infos, tmp_cam_infos = self.datasource.get_image_mock(img_idx, mock_id=difix_epoch_used_time, mock_dist=cfg.diffusion.dist_max)
                            depth_map = self.proj_depth_img(tmp_image_infos, tmp_cam_infos)
                            self.diffusion_depth_map[img_idx][difix_epoch_used_time] = depth_map.to(self.datasource.device)
                            ## only debug
                            # depth_np = depth_map.cpu().numpy()  # Convert to CPU numpy array
                            # valid_mask = depth_np > 0  # Non-zero depth region mask
                            # raw_img_array = (infer_rgb.clone().cpu().numpy() * 255).astype(np.uint8)
                            # raw_img_array[valid_mask] = [0, 0, 255]

                            # depth_image = Image.fromarray(raw_img_array)
                            # depth_image.save(os.path.join("debug/undis", f'{img_idx}_with_blue_depth.png'))
                        
                    
            if random.random() > 0.6:
                need_shift = True
                mock_dist = cfg.diffusion.dist_max
                mock_id = difix_epoch_used_time_cur

        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        if need_shift:
            image_infos, cam_infos = self.datasource.get_image_mock(img_idx, mock_id=mock_id, mock_dist=mock_dist)
            image_infos["diffusion_type"] = fix_type
            if downscale_factor != 1.0:
                down_rgb = (
                    torch.nn.functional.interpolate(
                        self.diffusion_rgb[img_idx][mock_id].unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                down_sky_mask = (
                    torch.nn.functional.interpolate(
                        self.diffusion_sky_mask[img_idx][mock_id].unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2),
                        scale_factor=downscale_factor,
                        mode="nearest",
                        antialias=False,
                    ).squeeze(0).squeeze(0))
                down_depth_map = sparse_lidar_map_downsampler(self.diffusion_depth_map[img_idx][mock_id], downscale_factor)
                image_infos['pixels'] = down_rgb
                image_infos['sky_masks'] = down_sky_mask
                image_infos['lidar_depth_map'] = down_depth_map
                image_infos['dynamic_masks'] = None
                if fix_type == "difix":
                    image_infos['gain'] = None
                else:
                    down_gain = (
                        torch.nn.functional.interpolate(
                            self.fusion_gain[img_idx][mock_id].unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2),
                            scale_factor=downscale_factor,
                            mode="nearest",
                            antialias=False,
                        ).squeeze(0).squeeze(0))
                    image_infos['gain'] = down_gain
            else:
                image_infos['pixels'] = self.diffusion_rgb[img_idx][mock_id]
                image_infos['sky_masks'] = self.diffusion_sky_mask[img_idx][mock_id]
                image_infos['lidar_depth_map'] = self.diffusion_depth_map[img_idx][mock_id]
                image_infos['dynamic_masks'] = None
                if fix_type == "difix":
                    image_infos['gain'] = None
                else:
                    image_infos['gain'] = self.fusion_gain[img_idx][mock_id]

            # image_infos['dynamic_masks'] = None
            image_infos['human_masks'] = None
            image_infos['vehicle_masks'] = None
            image_infos['diffusion_img_idx'] = torch.full(image_infos['img_idx'].shape, img_idx + 1, dtype=torch.long)
        else:
            image_infos, cam_infos = self.datasource.get_image(img_idx)
            image_infos['diffusion_img_idx'] = torch.full(image_infos['img_idx'].shape, 0, dtype=torch.long)
        self.datasource.reset_downscale_factor()
        
        return image_infos, cam_infos
    
    def __getitem__(self, idx) -> dict:
        return self.get_image(idx, camera_downscale=1.0)

    def __len__(self) -> int:
        return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
