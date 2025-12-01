#!/bin/bash

ROOT_DIR="$PWD"
scene_idx=5
data_type="valid"
start_timestep=120 # start frame index for training
end_timestep=159 # end frame index, -1 for the last frame
output_root="work_dirs/faithfusion/"
project="train_with_EIG_demo"
expname="id_0_cam_${scene_idx}_${start_timestep}_${end_timestep}"
moving_step_size=1.0
fix_start_step=3000
fix_interval=2000

python $ROOT_DIR/tools/train_with_gen.py \
    --config_file configs/faithfusion/streetgs_with_gen.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/1cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    data.pixel_source.load_smpl=False \
    data.data_root="data/waymo/processed/${data_type}" \
    diffusion.dist_max=$moving_step_size \
    difix_start_step=$fix_start_step \
    difix_fix_interval=$fix_interval
