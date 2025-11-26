#!/bin/bash

# List of scenes to process
scene_list=($(seq 1 2))
start_idx=0
end_idx=-1

task_index=0
ROOT_DIR="$PWD"

num_gpus=$(nvidia-smi --list-gpus | wc -l)

declare -a gpu_in_use
for ((i = 0; i < num_gpus; i++)); do
    gpu_in_use[$i]=0
done

# Record the Process ID (PID) of each task
declare -a task_pids
for ((i = 0; i < num_gpus; i++)); do
    task_pids[$i]=0
done

# Waymo dataset configuration
# Processed cameras:
#   idx    camera           original size
#    0    front_camera       (1920, 1280)
#    1    front_left_camera  (1920, 1280)
#    2    front_right_camera (1920, 1280)
#    3    left_camera        (1920, 866)
#    4    right_camera       (1920, 866)

# Constructing infinite pairing...
# declare -a train_cam_ids=(1 2 0 0 1 3 2 4 ...)
# declare -a render_cam_ids=(0 0 1 2 3 1 4 2 ...)

declare -a train_cam_ids=(1)
declare -a render_cam_ids=(0)

while [ $task_index -lt ${#scene_list[@]} ]; do
    for ((gpu_index = 0; gpu_index < num_gpus; gpu_index++)); do
        # Check if the task on this GPU has completed
        if [ ${task_pids[$gpu_index]} -ne 0 ] &&! kill -0 ${task_pids[$gpu_index]} 2>/dev/null; then
            task_pids[$gpu_index]=0
            gpu_in_use[$gpu_index]=0
        fi

        # Skip if the GPU is currently in use
        if [ ${gpu_in_use[$gpu_index]} -eq 1 ]; then
            continue
        fi

        # Get current scene index
        scene_idx_str=${scene_list[$task_index]}
        # Base command
        base_command=""
        printf -v scene_idx_str_fill '%03d' $scene_idx_str
        for ((j = 0; j < ${#train_cam_ids[@]}; j++)); do
            train_cam_id=${train_cam_ids[$j]}
            render_cam_id=${render_cam_ids[$j]}
            train_cameras_str="[${train_cam_id}]"
            train_cameras_str=$(echo "$train_cameras_str" | sed 's/[0-9]/2/g')
            render_cameras_str="[${train_cam_id},${render_cam_id}]"
            render_cameras_str=$(echo "$render_cameras_str" | sed 's/[0-9]/2/g')

            source_dir="work_dirs/train/diffusion_pair/id_${scene_idx_str}_cam_${train_cam_id}_${start_idx}_${end_idx}/"
            source_switch_dir="work_dirs/train/diffusion_pair/id_${scene_idx_str}_cam_${train_cam_id}_${start_idx}_${end_idx}/switch_camera/"
            
            if [ ! -e  "$source_dir/checkpoint_final.pth" ]; then
                base_command+="CUDA_VISIBLE_DEVICES=$gpu_index python $ROOT_DIR/tools/train.py --config_file configs/faithfusion/streetgs_trans_cam.yaml \
                    --output_root work_dirs/train/ --project diffusion_pair --skip_video_output \
                    --run_name id_${scene_idx_str}_cam_${train_cam_id}_${start_idx}_${end_idx} data.pixel_source.load_smpl=False \
                    data.scene_idx=${scene_idx_str} data.start_timestep=${start_idx} data.end_timestep=${end_idx} \
                    data.pixel_source.downscale_when_loading=${render_cameras_str} data.pixel_source.cameras=[${train_cam_id}] \
                    data.data_root=data/waymo/processed/train trainer.optim.num_iters=30000 &&" 
            fi
            
            if [ ! -e  "$source_switch_dir/cam${train_cam_id}_to_cam${render_cam_id}/*.pt" ]; then
                base_command+="CUDA_VISIBLE_DEVICES=$gpu_index python $ROOT_DIR/tools/eval.py \
                    --resume_from work_dirs/train/diffusion_pair/id_${scene_idx_str}_cam_${train_cam_id}_${start_idx}_${end_idx}/checkpoint_final.pth \
                    render.render_full=False render.skip_object=False data.pixel_source.cameras=[${train_cam_id},${render_cam_id}] \
                    data.pixel_source.train_cameras=[${train_cam_id}] data.pixel_source.test_cameras=[${render_cam_id}] \
                    data.pixel_source.downscale_when_loading=${render_cameras_str} render.render_novel.traj_types=['switch_camera'] \
                    render.render_novel.render_EIG=True &&"
            fi
        done
        base_command+="echo 'finished'"
        
        # Set CUDA_VISIBLE_DEVICES environment variable (The setting here can be kept, even though the command already contains it)
        command="$base_command"
        echo "Starting task on GPU $gpu_index: $command"
        # Start the task
        bash -c "$command" >> ${scene_idx_str_fill}.log 2>&1 &
        # Record the task's Process ID
        task_pids[$gpu_index]=$!
        # Mark the GPU as in use
        gpu_in_use[$gpu_index]=1
        task_index=$((task_index + 1))

        # If the task index has reached the end of the scene list, break the inner loop
        if [ $task_index -ge ${#scene_list[@]} ]; then
            break
        fi
    done

    # Check if all tasks have finished; if so, exit the loop
    all_tasks_finished=true
    for ((i = 0; i < num_gpus; i++)); do
        if [ ${task_pids[$i]} -ne 0 ]; then
            all_tasks_finished=false
            break
        fi
    done
    if $all_tasks_finished; then
        break
    fi

    # Check every 10 seconds
    sleep 10
done

# Wait for all remaining tasks to complete
for ((i = 0; i < num_gpus; i++)); do
    if [ ${task_pids[$i]} -ne 0 ]; then
        wait ${task_pids[$i]}
    fi
done