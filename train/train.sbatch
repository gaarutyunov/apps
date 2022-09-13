#!/bin/bash
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="apps"
#SBATCH --time=1-12:00
#SBATCH --gpus-per-node=2
#SBATCH --output="F-"%j.out
#SBATCH --error="F-"%j.err
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --exclude="cn-043"
#SBATCH --nodes=4


source ~/.bashrc
module restore default

conda activate apps
chmod +x ./get_gpu_indexes_safe_new.sh
./get_gpu_indexes_safe_new.sh

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

deepspeed --hostfile=hostfile_${SLURM_JOB_ID} tune_apps_gpt.py
