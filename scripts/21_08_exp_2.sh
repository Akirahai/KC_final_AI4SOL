#!/bin/bash

# Define the models and seeds
models=(
  # "google-bert/bert-base-cased"
  # "FacebookAI/roberta-base"
  # "google/flan-t5-base"
  "google-bert/bert-large-cased"
  # "FacebookAI/roberta-large"
)
seeds=(42 84 168)
gpus = (0 1 2 3 0)

for model in "${models[@]}"; do
  # Loop through each seed and corresponding GPU
  for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    gpu=${gpus[i]}
    
    python main.py --use-gpu --gpus $gpu --phase train --batch-size 16 --lr 0.00001 --epochs 70 --model $model --seed $seed --experiment 21_08_deliver_ver_6

  done
done