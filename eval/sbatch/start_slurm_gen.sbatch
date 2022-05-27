#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=R-%j.out
#SBATCH --constraint="type_e"

source ~/.bashrc
module restore default

source activate apps

#env
type python
nvcc --version
echo "PATH=${PATH}" > ~/.deepspeed_env

echo $1 $2 $3 $4 $5 $6
pushd .

cd $1
echo "python generate_gpt_codes.py -s $2 -e $3 --save $4 --load $5 --test_loc $6"
python generate_gpt_codes.py -s $2 -e $3 --save $4 --load $5 --test_loc $6 --debug

popd
mkdir slurm-output
mv R-${SLURM_JOB_ID}.out slurm-output
