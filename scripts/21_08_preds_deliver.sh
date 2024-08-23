// This is a shell script written in Bash. The script defines an array of models and an array of seeds.
// It then loops through each model and seed combination and runs a Python script `main.py` with
// specific arguments like `--use-gpu`, `--gpus`, `--phase`, `--batch-size`, `--lr`, `--epochs`,
// `--model`, `--seed`, `--experiment`, and `--top-k`.
#!/bin/bash

# Define the models and seeds
models=(
  "google-bert/bert-base-cased"
  "FacebookAI/roberta-base"
  "google/flan-t5-base"
  "FacebookAI/roberta-large"
  "google-bert/bert-large-uncased"
)
seeds=(12 24 42 84 168)

# Loop through each model and seed
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python main.py --use-gpu --gpus 1 2 3 --phase test --batch-size 16 --lr 0.0001 --epochs 70 --model "result/21_08_deliver/seed_$seed/$model" \
    --seed $seed --experiment 21_08_deliver --top-k 5 --training-folder Training_data_1234 --num-of-labels 29
  done
done
