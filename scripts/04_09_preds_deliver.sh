#!/bin/bash

# Define the models and seeds
models=(
  "google-bert/bert-base-cased"
  "google-bert/bert-large-cased"
  "FacebookAI/roberta-base"
  "FacebookAI/roberta-large"
  "google/flan-t5-base"
)
seeds=(12 24 42 84 168)

# Loop through each model and seed
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python main.py --use-gpu --gpus 1 2 3 --phase test --batch-size 16 --lr 0.0001 --epochs 70 --model "result/28_08_deliver/seed_$seed/$model" \
    --seed $seed --experiment 04_09_deliver_train_set_12345 --top-k 5 --training-folder Training_data_12345 --num-of-labels 50 --testing-set ASDiv-600-6th_test.csv
  done
done
