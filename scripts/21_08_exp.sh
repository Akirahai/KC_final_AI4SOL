#!/bin/bash

# Define the models and seeds
models=(
  # "google-bert/bert-base-cased"
  # "FacebookAI/roberta-base"
  # "google/flan-t5-base"
  "google-bert/bert-large-cased"
  "FacebookAI/roberta-large"
)
seeds=(12 24 42 84 168)
# seeds=(84 168)

# Loop through each model and seed
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python main.py --use-gpu --gpus 0 1 --phase train --batch-size 16 --lr 0.0001 --epochs 70 --model $model --seed $seed --experiment 21_08_deliver_ver_4
    sleep 300  # 600 seconds = 10 minutes

  done
done
