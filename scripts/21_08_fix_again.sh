#!/bin/bash

# Define the models and seeds
models=(
  "FacebookAI/roberta-large"
  "google-bert/bert-large-uncased"
)
seeds=(5 10 15 20 42)

# Loop through each model and seed
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python main.py --use-gpu --gpus 0 1 2 3 --phase train --batch-size 16 --lr 0.0001 --epochs 70 --model $model --seed $seed --experiment 21_08_deliver
  done
done
