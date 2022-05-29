#!/bin/bash
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="apps"
#SBATCH --time=0-12:00
#SBATCH --gpus-per-node=2
#SBATCH --output="F-"%j.out
#SBATCH --error="F-"%j.err
#SBATCH --constraint="type_e"
#SBATCH --nodes=2


source ~/.bashrc
module restore default

conda activate apps
/opt/hse/slurm/scripts/get_gpu_indexes.sh

#env
echo '----------------------------------------------------------------------\n'
type python
echo '----------------------------------------------------------------------\n'
conda list
echo '----------------------------------------------------------------------\n'
nvcc --version
echo '----------------------------------------------------------------------\n'
echo "PATH=${PATH}" > ~/.deepspeed_env
echo '----------------------------------------------------------------------\n'
ds_report
echo '----------------------------------------------------------------------\n'
python -m torch.utils.collect_env
echo '----------------------------------------------------------------------\n'
python test_cuda.py
echo '----------------------------------------------------------------------\n'

#Executable

USE_TF=NO deepspeed --hostfile=hostfile train/tune_apps_gpt.py
