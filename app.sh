#!/usr/bin/bash

#SBATCH -J ksy-love
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out
set -euo pipefail

PY="/data/juventa23/anaconda3/envs/kb_env/bin/python"

echo "HOST: $HOSTNAME"
echo "PWD : $(pwd)"
echo "Using PY: $PY"
"$PY" -V

# train.py가 없으면 app.py로 바꾸세요
"$PY" -u app.py
