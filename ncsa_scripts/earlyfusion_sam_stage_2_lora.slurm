#!/bin/bash

#SBATCH --job-name=earlyfusion_sam_stage_2_lora
#SBATCH --output="ncsa_logs/%j.%N.earlyfusion_sam_stage_2_lora.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch&projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcdq-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 20:00:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"


conda init bash
source ~/.bashrc
conda activate llava

### run script ###
cd /scratch/bcdq/wangz3/ecole-gvs-method/third_party/LLaVA
# interative
bash ncsa_scripts/llava_geo/earlyfusion_sam_stage_2_lora.sh

# sbatch scripts/slurm/configs/finetune_4xA100_4tp.slurm
# squeue -u $USER