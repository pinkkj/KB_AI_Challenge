#!/usr/bin/bash

#SBATCH -J ksy-love
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -o logs/slurm-%A.out

python app.py
