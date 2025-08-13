#!/usr/bin/bash

#SBATCH -J ksy-love
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out

python app.py
