#!/bin/bash

#SBATCH --job-name="T-apps"
#SBATCH --time=1-12:00
#SBATCH --output=T-%A-%a.out
#SBATCH --error=T-%A-%a.err
#SBATCH --mail-user=gaarutyunov@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --array=0-26
#SBATCH --constraint="type_a|type_b|type_c|type_d"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

source ~/.bashrc
module restore default

conda activate apps

TOTAL=3011
SKIP=10

i=$((SLURM_ARRAY_TASK_ID*SKIP))

START=$i
END=$((i+SKIP))

SAVE_LOC="~/apps/results"
TEST_LOC="~/apps/data_split/test.json"

python test_one_solution.py -s $START -e $END --save $SAVE_LOC --test_loc $TEST_LOC