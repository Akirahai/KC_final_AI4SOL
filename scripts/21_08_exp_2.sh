#!/bin/bash


# Training with batch 1,2,3 data set, evaluate with batch 4 data set, using larger model and smaller batch size (8)
# Define the models and seeds
models=(
  # "google-bert/bert-base-cased"
  # "google-bert/bert-large-cased"
  # "FacebookAI/roberta-base"
  # "FacebookAI/roberta-large"
  # "google/flan-t5-base"
  "Qwen/Qwen2-Math-1.5B"
  "Qwen/Qwen2-1.5B"
  "microsoft/phi-1_5"
)
seeds=(12 24 42 84 168)
gpus=(3 3 3 3 3)

# Loop through each model and seed
for model in "${models[@]}"; do
  for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    gpu=${gpus[i]}

    python main.py --use-gpu --gpus $gpu --phase train --batch-size 8 --lr 0.00001 --epochs 150 --model $model --seed $seed \
    --experiment 21_08_deliver --training-folder Training_data_1234 --num-of-labels 29
    
    find "./result/21_08_deliver/seed_$seed/$model" -type d -name "checkpoint-*" -exec rm -r {} +

    sleep 300
    
  done
done
