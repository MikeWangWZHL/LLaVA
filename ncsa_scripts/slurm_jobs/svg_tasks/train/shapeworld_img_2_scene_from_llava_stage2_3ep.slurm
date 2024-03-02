#!/bin/bash

#SBATCH --job-name=shapeworld_img_2_scene_fullfinetune_lava
#SBATCH --output="ncsa_logs/%j.%N.llava_fullfinetune.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch&projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcdq-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 10:00:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"


conda init bash
source ~/.bashrc
conda activate llava

export CODE_DIR="/scratch/bcdq/wangz3/ecole-gvs-method/third_party/LLaVA"
export DATA_DIR="/scratch/bcdq/wangz3/llava_data"
echo "CODE_DIR: $CODE_DIR"
echo "DATA_DIR: $DATA_DIR"

### run script ###
cd $CODE_DIR
bash ncsa_scripts/train_scripts/svg_tasks/train_jobs/shapeworld_2_scene_from_llava_stage2_3epochs.sh

### run eval ###
cd "/scratch/bcdq/wangz3/ecole-gvs-method"
bash scripts/image-scene_eval/_run_jobs_img_scene_from_llava_stage2.sh


# sbatch scripts/slurm/configs/finetune_4xA100_4tp.slurm
# squeue -u $USER