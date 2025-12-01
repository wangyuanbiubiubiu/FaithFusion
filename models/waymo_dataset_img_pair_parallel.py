import os
import random
import numpy as np
import cv2
import json
from tqdm import tqdm
from matplotlib import cm
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch
from torchvision.transforms import v2
from PIL import Image
from einops import rearrange
import cv2
import shutil
import math

OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

class WaymoImgPairProcessor:
    def __init__(self, base_dir, img_pair_dir, caption_offline_dir, output_dir,
                max_num_frames=81, frame_interval=1, 
                num_frames=81, height=480, width=720):
        self.base_dir = base_dir
        self.img_pair_dir = img_pair_dir
        self.caption_offline_dir = caption_offline_dir
        self.output_dir = output_dir
        self.gain_out = True

        #
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_process = v2.Compose([
            v2.ToTensor(),
        ])
        self.render_ori_prob = 0.05
    
    def view_concate_video(self, video_frames, gain_frames, expand_video_frames, ref_video_frames, video_path):
        video_frames = video_frames.astype(np.uint8)
        gain_frames = gain_frames.astype(np.uint8)
        expand_video_frames = expand_video_frames.astype(np.uint8)
        ref_video_frames = ref_video_frames.astype(np.uint8)

        frames_count, height, width, _ = video_frames.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(self.output_dir, video_path)

        fps = 10 
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for i in range(frames_count):
            frame1 = video_frames[i]
            frame2 = expand_video_frames[i]
            frame3 = gain_frames[i]
            frame4 = ref_video_frames[i]
            
            half_height = height // 2
            half_width = width // 2
            
            frame1_resized = cv2.resize(frame1, (half_width, half_height))
            frame2_resized = cv2.resize(frame2, (half_width, half_height))
            frame3_resized = cv2.resize(frame3, (half_width, half_height))
            frame3_resized = np.stack([frame3_resized] * 3, axis=-1) 
    
            frame4_resized = cv2.resize(frame4, (half_width, half_height))
            
            top_row = np.hstack([frame1_resized, frame2_resized])
            bottom_row = np.hstack([frame3_resized, frame4_resized])
            
            combined_frame = np.vstack([top_row, bottom_row])
            
            if combined_frame.shape != (height, width, 3):
                combined_frame = cv2.resize(combined_frame, (width, height))
            
            out.write(combined_frame)

        out.release()
        print(f"saved: {output_video_path}")

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_matrix_from_txt(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            matrix = np.array([list(map(float, line.strip().split())) for line in lines])
        return matrix
    
    def load_intrinsics(self, intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
            values = [float(line.strip()) for line in lines]
        # 假设内参顺序为：fx, fy, cx, cy, k1, k2, p1, p2, k3
        intrinsics = {'fx': values[0], 'fy': values[1], 'cx': values[2], 'cy': values[3], 'distortion_coeffs': values[4:9]}
        return intrinsics
    
    def count_frames(self, segment_dir):
        images_dir = os.path.join(segment_dir, "images")
        image_files = [f for f in os.listdir(images_dir) if f.endswith("_0.jpg")]
        return len(image_files)
 
    def process_segment(self, segment_id, start_frame, num_frames=30, alpha=0.5,
                        points_cam_coords=None, point_size=8, output_dir="output", img_pair_name="", cam_id=0, 
                        data_type="train", traj_mode="default", extra_cam_id=-1, gain_th=1.0):
        segment_id_str = f"{segment_id:03d}"
        segment_dir = os.path.join(self.base_dir, data_type, segment_id_str)
        
        if not os.path.exists(segment_dir):
            raise FileNotFoundError(f"Segment directory does not exist: {segment_dir}")
        
        start_idx_map = {
            '005':120,
            "018":0,
            "027":110,
            "065":0,
            "081":0,
            "096":90,
            "121":40,
            "164":70
        }
        os.makedirs(output_dir, exist_ok=True)
        
        images_dir = os.path.join(segment_dir, "images")
        img_pair_json = os.path.join(self.img_pair_dir, "pair_path.json")
        intrinsics_path = os.path.join(segment_dir, "intrinsics", str(cam_id) + ".txt")
        extrinsics_path = os.path.join(segment_dir, "extrinsics", str(cam_id) + ".txt")
        instances_dir = os.path.join(segment_dir, "instances")
        poses_dir = os.path.join(segment_dir, "ego_pose")
        frame_infos = json.load(
            open(f'{instances_dir}/frame_instances.json')
        )
        instances_meta = json.load(
            open(f'{instances_dir}/instances_info.json')
        )
        
        intrinsics = self.load_intrinsics(intrinsics_path)
        intrinsic = np.array([[intrinsics["fx"], 0, intrinsics["cx"]], [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]], dtype=np.float32)
        extrinsics = self.load_matrix_from_txt(extrinsics_path)

        img_pair_dic = {}
        if os.path.exists(img_pair_json):
            with open(img_pair_json, 'r') as fp:
                pair_json = json.load(fp)

            for tmp_id, item in enumerate(pair_json):
                img_id = item["rgb"].split(".")[0].split("_")[-1]
                img_pair_dic[int(img_id)] = {
                    "rgb": item["rgb"],
                    "gain": item["gain"],
                    "gt": item["gt"]
                }
        else:
            print("No img_pair_json found, return None")
            return None
        total_frames = len(img_pair_dic)
        end_frame = start_frame + num_frames
        
        video_frames = []
        expand_video_frames = []
        gain_frames = []
        gain_frames_vis = []
        raw_masks = []
        ref_video_frames = []
        refs_image = None
        refs_gain = None
        refs_rgb = None
        refs_reference = None

        step = 40
        num_x_dist = 49 - step
        dist_max = float(img_pair_name.split("_")[-1])
        frame_step = 51
        if (int(start_frame / frame_step) <= 1):
            mock_dist = dist_max / num_x_dist
        else:
            mock_dist = -dist_max / num_x_dist
        
        for frame_idx, i in (enumerate(range(start_frame, end_frame))):
            expanded_img = None
            if i not in img_pair_dic:
                img = refs_image.copy()
                gain_img = refs_gain.copy()
                render_img = refs_rgb.copy()
                ref_img = refs_reference.copy()
            else:
                img_path =  os.path.join(self.img_pair_dir, img_pair_dic[i]["gt"])
                render_path = os.path.join(self.img_pair_dir, img_pair_dic[i]["rgb"])
                gain_path = os.path.join(self.img_pair_dir, img_pair_dic[i]["gain"])
                difix_path = render_path.replace("rgb", "difix")

                if not os.path.exists(img_path) or not os.path.exists(gain_path):
                    img = refs_image.copy()
                    gain_img = refs_gain.copy()
                    render_img = refs_rgb.copy()
                    ref_img = refs_reference.copy()
                else:
                    img = cv2.imread(img_path)
                    render_img = cv2.imread(render_path)
                    gain_tensor = torch.load(gain_path, weights_only=True, map_location="cpu")#xf
                    difix_img = cv2.imread(difix_path)
                    no_opacity = gain_tensor > 99

                    if self.gain_out:
                        scaled_tensor = torch.clamp(gain_tensor, max=gain_th) / gain_th
                        scaled_tensor *= 1.0
                        scaled_tensor[no_opacity] = 1.0
                        scaled_tensor *= 255
                    else:
                        scaled_tensor = (gain_tensor > gain_th) * 255
                    gain_img = scaled_tensor.detach().cpu().numpy()
                    gain_img = gain_img.astype(np.uint8)

                                            
                    start_frame_id = start_idx_map[segment_id_str]
                    if frame_idx < num_x_dist:
                        mock_times = frame_idx
                        usr_frame_id = start_frame_id
                    elif frame_idx < 49:
                        mock_times = num_x_dist
                        usr_frame_id = start_frame_id + (frame_idx - num_x_dist)
                    else:
                        mock_times = num_x_dist
                        usr_frame_id = start_frame_id + (49 - num_x_dist)
                    

                    frame_ins_list = frame_infos[str(usr_frame_id)]
                    cam_to_ego = extrinsics @ OPENCV2DATASET
                    ego_to_world = np.loadtxt(os.path.join(poses_dir, f"{str(usr_frame_id).zfill(3)}.txt"))
                    # print(mock_times * mock_dist)
                    render_shift = np.array([[1., 0., 0., 0.],
                                [0., 1., 0., mock_dist * mock_times],
                                [0., 0., 1., 0],
                                [0., 0., 0., 1.]])
                    new_ego_to_world = ego_to_world @ render_shift
                    cam2world = new_ego_to_world @ cam_to_ego            
                    ref_img = difix_img.copy()
            if frame_idx == 0:
                render_img = img.copy() #good img
                gain_img *= 0
            
            video_frames.append(img)
            expand_video_frames.append(render_img)
            gain_frames.append(gain_img)
            ref_video_frames.append(ref_img)

        video_frames = np.array(video_frames)
        gain_frames = np.array(gain_frames)
        expand_video_frames = np.array(expand_video_frames)
        ref_video_frames = np.array(ref_video_frames)
        # self.view_concate_video(video_frames, gain_frames, expand_video_frames, ref_video_frames, "concate_test.mp4")
        #BGR ouput
        return {
            "prompt": None,
            "video": expand_video_frames,#gt
            "video_rendering": expand_video_frames, #NVS
            "video_ref": ref_video_frames,
            "raw_gain" : gain_frames,
            "path": 'segment_id-{}-cam_id-{}-start_frame-{}'.format(segment_id, cam_id, start_frame),
        }


class WaymoImgPairDataset(Dataset):
    def __init__(self, base_dir, img_pair_dir, caption_offline_dir, scene_list, pair_list, num_frames=81, min_start_frame=30, 
        max_start_frame=110,  frame_interval=1, point_size=2, output_dir="waymo_output", cam_id=0, height=480, width=720,traj_mode="default"):
        self.base_dir = base_dir
        self.num_frames = num_frames
        self.max_start_frame = max_start_frame
        self.point_size = point_size
        self.output_dir = output_dir
        self.scene_list = scene_list
        self.pair_list = pair_list
        self.processor = WaymoImgPairProcessor(base_dir,img_pair_dir, caption_offline_dir, output_dir, max_start_frame, frame_interval, num_frames, height, width)
        self.MAX_RETRIES = 20
        self.cam_id = cam_id
        self.traj_mode = traj_mode

    def __len__(self):
        return len(self.scene_list)
    
    def mock_data(self, segment_id, start_frame, img_pair_name, data_type):
        print(f"Processing segment {segment_id}, start_frame {start_frame}, img_pair_name {img_pair_name}")
        cur_cam_id = int(img_pair_name[-1])
        result = self.processor.process_segment(
                    segment_id=segment_id,
                    start_frame=start_frame,
                    num_frames=self.num_frames,
                    points_cam_coords=self.points_cam_coords,
                    point_size=self.point_size,
                    output_dir=self.output_dir,
                    img_pair_name=img_pair_name,
                    cam_id=cur_cam_id,
                    data_type=data_type,
                    traj_mode=self.traj_mode
                )
        return result        
    
    def __getitem__(self, i):
        segment_info = self.scene_list[i]
        pair_name  = segment_info['pair_name']
        segment_id = segment_info["scene_id"]
        start_idx_list = segment_info["start_idx_list"]
        data_type = segment_info["data_type"]
        img_pair_name = os.path.join(data_type, f"{segment_id:03d}", "0_-1/switch_camera/", pair_name)
        extra_cam_id = -1
        video_out = segment_info.get("video_out", False)
        gain_th = segment_info["gain_th"]

        start_frame = random.choice(start_idx_list)
        cur_cam_id = pair_name.split("_")[0][-1]
        result = self.processor.process_segment(
                    segment_id=segment_id,
                    start_frame=start_frame,
                    num_frames=self.num_frames,
                    alpha=1.0,
                    output_dir=self.output_dir,
                    img_pair_name=img_pair_name,
                    cam_id=cur_cam_id,
                    data_type=data_type,
                    traj_mode=self.traj_mode,
                    extra_cam_id=extra_cam_id,
                    gain_th=gain_th)
        return result
