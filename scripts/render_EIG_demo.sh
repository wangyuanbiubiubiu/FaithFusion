#!/bin/bash
ROOT_DIR="$PWD"
scene_idx=5
data_type="valid"
start_timestep=120 # start frame index for training
end_timestep=159 # end frame index, -1 for the last frame
output_root="work_dirs/faithfusion/"
project="EIG_demo"
expname="id_0_cam_${scene_idx}_${start_timestep}_${end_timestep}"

python $ROOT_DIR/tools/train.py \
    --config_file configs/faithfusion/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/1cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    data.pixel_source.load_smpl=False \
    data.data_root="data/waymo/processed/${data_type}" \
    trainer.optim.num_iters=30000
