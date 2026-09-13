#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/vae_train_%j.out
#SBATCH --error=logs/vae_train_%j.err

mkdir -p logs checkpoints

source .venv/bin/activate

python3 -m src.train_vae \
    --oasis-root /home/groups/comp3710/OASIS \
    --epochs 35 \
    --anneal-epochs 10 \
    --beta-max 1.0 \
    --checkpoint-path checkpoints/vae.pth