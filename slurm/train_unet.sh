#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/unet_train_%j.out
#SBATCH --error=logs/unet_train_%j.err

mkdir -p logs checkpoints

source .venv/bin/activate

python3 -m src.train_unet \
    --oasis-root /home/groups/comp3710/OASIS \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-3 \
    --dice-weight 0.5 \
    --checkpoint-path checkpoints/unet.pth