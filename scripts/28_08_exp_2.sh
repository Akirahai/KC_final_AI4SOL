#!/bin/bash

# Define the models and seeds
models=(
  # "google-bert/bert-base-cased"
  # "google-bert/bert-large-cased"
  "FacebookAI/roberta-base"
  "FacebookAI/roberta-large"
  "google/flan-t5-base"
)
seeds=(12 24 42 84 168)
gpus=(1 1 1 1 1)

for model in "${models[@]}"; do
  # Loop through each seed and corresponding GPU
  for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    gpu=${gpus[i]}
    
    python main.py --use-gpu --gpus $gpu --phase train --batch-size 16 --lr 0.00001 --epochs 150 --model $model --seed $seed \
    --experiment 28_08_deliver --training-folder Training_data_12345 --num-of-labels 50

    find "./result/28_08_deliver/seed_$seed/$model" -type d -name "checkpoint-*" -exec rm -r {} +

    sleep 300

  done
done