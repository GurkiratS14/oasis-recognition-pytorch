#!/bin/bash
#SBATCH --job-name=gan_train
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/gan_train_%j.out
#SBATCH --error=logs/gan_train_%j.err

mkdir -p logs checkpoints

source .venv/bin/activate

python3 -u -m src.train_gan \
    --oasis-root /home/groups/comp3710/OASIS \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-4 \
    --sample-every 5 \
    --checkpoint-dir checkpoints/gan