#!/bin/bash
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="apps"
#SBATCH --time=0-12:00
#SBATCH --gpus-per-node=2
#SBATCH --output="F-"%j.out
#SBATCH --error="F-"%j.err
#SBATCH --constraint="type_e"
#SBATCH --nodes=4


source ~/.bashrc
module restore default

source activate apps
/opt/hse/slurm/scripts/get_gpu_indexes.sh

#env
type python
nvcc --version
echo "PATH=${PATH}" > ~/.deepspeed_env

USE_TF=NO deepspeed tune_apps_gpt.py
