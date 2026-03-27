#!/bin/bash

model=legs
dataset=test
motion_id=0
headless=False
exp_name=kinesis-imitation

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model=$2
            shift
            shift
            ;;
        --dataset)
            dataset=$2
            shift
            shift
            ;;
        --motion_id)
            motion_id=$2
            shift
            shift
            ;;
        --headless)
            headless=$2
            shift
            shift
            ;;
        --exp_name)
            exp_name=$2
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ $model == "legs" ]]; then
    config_name="config_legs.yaml"
    run_config="play_run_legs"
    initial_pose_dir="data/initial_pose/legs"
elif [[ $model == "legs_abs" ]]; then
    config_name="config_legs_abs.yaml"
    run_config="play_run_legs_abs"
    initial_pose_dir="data/initial_pose/legs_abs"
elif [[ $model == "legs_back" ]]; then
    config_name="config_legs_back.yaml"
    run_config="play_run_legs_back"
    initial_pose_dir="data/initial_pose/legs_back"
else
    echo "Invalid model: $model. Use 'legs', 'legs_abs', or 'legs_back'."
    exit 1
fi

if [[ $dataset == "train" ]]; then
    motion_file="data/kit_train_motion_dict.pkl"
    initial_pose_file="${initial_pose_dir}/initial_pose_train.pkl"
elif [[ $dataset == "test" ]]; then
    motion_file="data/kit_test_motion_dict.pkl"
    initial_pose_file="${initial_pose_dir}/initial_pose_test.pkl"
else
    echo "Invalid dataset: $dataset. Use 'train' or 'test'."
    exit 1
fi

python src/run.py \
    --config-name ${config_name} \
    exp_name=${exp_name} \
    epoch=-1 \
    run=${run_config} \
    run.headless=${headless} \
    run.motion_id=${motion_id} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file}
