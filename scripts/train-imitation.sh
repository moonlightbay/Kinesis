#!/bin/bash

model=legs
exp_name=kinesis-imitation
num_threads=1
headless=True

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model=$2
            shift
            shift
            ;;
        --exp_name)
            exp_name=$2
            shift
            shift
            ;;
        --num_threads)
            num_threads=$2
            shift
            shift
            ;;
        --headless)
            headless=$2
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
elif [[ $model == "legs_abs" ]]; then
    config_name="config_legs_abs.yaml"
elif [[ $model == "legs_back" ]]; then
    config_name="config_legs_back.yaml"
else
    echo "Invalid model: $model. Use 'legs', 'legs_abs', or 'legs_back'."
    exit 1
fi

python src/run.py \
    --config-name ${config_name} \
    exp_name=${exp_name} \
    run.num_threads=${num_threads} \
    run.headless=${headless}
