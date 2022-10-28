#!/bin/bash

#SBATCH --job-name=fss-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100,v100,rtx8000

# job info
config_path=$1
exp_id=$2

# Singularity path
coco_path=/scratch/$USER/dataset/coco2014.sqf
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${coco_path}:ro \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/FSS-Codebase
python -m src.train --config ${config_path} --exp_id ${exp_id}
"


