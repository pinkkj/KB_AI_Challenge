#!/usr/bin/bash

#SBATCH -J kb-app
#SBATCH -p debug_ugrad
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH -o logs/slurm-%A.out

python app.py
