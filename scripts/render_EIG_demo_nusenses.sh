#!/bin/bash
ROOT_DIR="$PWD"

scene_idx=87
data_type="valid"
start_timestep=130 # start frame index for training
end_timestep=169 # end frame index, -1 for the last frame
output_root="work_dirs/faithfusion/"
project="EIG_demo_nusenses"
expname="id_0_cam_${scene_idx}_${start_timestep}_${end_timestep}"

python $ROOT_DIR/tools/train.py \
    --config_file configs/faithfusion/streetgs_multicam.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=nuscenes/6cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    data.pixel_source.load_smpl=False \
    data.data_root="data/nusenses/processed/${data_type}" \
    trainer.optim.num_iters=40000
