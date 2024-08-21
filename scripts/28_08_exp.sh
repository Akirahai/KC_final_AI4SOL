#!/bin/bash

# Define the models and seeds
models=(
  "google-bert/bert-base-cased"
  "google-bert/bert-large-cased"
  # "FacebookAI/roberta-base"
  # "FacebookAI/roberta-large"
  # "google/flan-t5-base"
)
seeds=(12 24 42 84 168)
gpus=(0 1 0 1 0)

# Loop through each model and seed
for model in "${models[@]}"; do
  for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    gpu=${gpus[i]}

    python main.py --use-gpu --gpus $gpu --phase train --batch-size 16 --lr 0.00001 --epochs 70 --model $model --seed $seed --experiment 28_08_deliver
    find "./result/28_08_deliver/seed_$seed/$model" -type d -name "checkpoint-*" -exec rm -r {} +
    
  done
done
