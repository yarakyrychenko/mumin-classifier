#!/bin/bash
#SBATCH --job-name=xlm-roberta
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yk1937/%j_%x.out
#SBATCH --error=/scratch/yk1937/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

singularity exec --nv --bind /scratch/yk1937 --overlay /scratch/yk1937/overlay-25GB-500K.ext3:ro /scratch/yk1937/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/yk1937/train_classification.py"