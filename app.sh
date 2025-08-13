#!/usr/bin/bash

#SBATCH srun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=10G
#SBATCH -p debug_ugrad
#SBATCH -w aurora-g3
#SBATCH --pty $SHELL
#SBATCH -o logs/slurm-%A.out

python app.py
