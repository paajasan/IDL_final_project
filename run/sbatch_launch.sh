#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 15:00:00
#SBATCH --mem=5000
#SBATCH --gres=gpu:v100:1
#SBATCH -J IDL_fp
#SBATCH -o output.txt
#SBATCH --account=project_2002605
#SBATCH --mail-type=END
#SBATCH --mail-user=santeri.e.paajanen@helsinki.fi

module purge
module load pytorch/1.13

#python ../src/train_model.py  --maxepoch 200 -n  ${SLURM_ARRAY_TASK_ID} --slurm-mode
python ../src/train_model.py -e 30 -n 2 --pretrained --train-all -lr 0.00005 -wd 0 --slurm-mode
