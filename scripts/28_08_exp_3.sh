#!/bin/bash


# Training with batch 1,2,3, 5 data set, evaluate with batch 4 data set, using larger model and smaller batch size (8)

# Define the models and seeds
models=(
  # "google-bert/bert-base-cased"
  # "google-bert/bert-large-cased"
  # "FacebookAI/roberta-base"
  # "FacebookAI/roberta-large"
  # "google/flan-t5-base"
  # "Qwen/Qwen2-Math-1.5B"
  # "Qwen/Qwen2-1.5B"
  "microsoft/phi-1_5"
  'HuggingFaceTB/SmolLM-1.7B'
)
seeds=(12 24 42 84 168)
gpus=(2 2 2 2 2 )

for model in "${models[@]}"; do
  # Loop through each seed and corresponding GPU
  for i in "${!seeds[@]}"; do
    seed=${seeds[i]}
    gpu=${gpus[i]}
    
    python main.py --use-gpu --gpus $gpu --phase train --batch-size 8 --lr 0.00001 --epochs 150 --model $model --seed $seed \
    --experiment 28_08_deliver --training-folder Training_data_12345 --num-of-labels 50

    find "./result/28_08_deliver/seed_$seed/$model" -type d -name "checkpoint-*" -exec rm -r {} +

    sleep 300

  done
done