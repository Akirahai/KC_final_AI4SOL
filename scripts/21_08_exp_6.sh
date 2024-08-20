#!/bin/bash

# Define the models and seeds
models=(
  # "google-bert/bert-base-cased"
  "FacebookAI/roberta-base"
  "google/flan-t5-base"
  # "FacebookAI/roberta-large"
)
seeds=(12 24 42 84 168)
gpus=(0 1 0 1 0)

# Loop through each model and seed
for model in "${models[@]}"; do
  for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    gpu=${gpus[i]}

    python main.py --use-gpu --gpus $gpu --phase train --batch-size 16 --lr 0.00001 --epochs 70 --model $model --seed $seed --experiment 21_08_deliver_ver_6
    find 21_08_deliver_ver_6 -type d -name "checkpoint-*" -exec rm -r {} +
    
  done
done
