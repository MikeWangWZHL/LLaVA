#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --output="ncsa_logs/%j.%N.test_job.out"
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
#SBATCH -t 00:01:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"


conda init bash
source ~/.bashrc
conda activate llava

srun python3 test_ncsa.py # init an interactive session


# run the container binary with arguments: python3 <program.py>
# --bind /projects/bbXX  # add to apptainer arguments to mount directory inside container

# MODEL_DIR=/projects/bcbf/xingyao6/models
# WORK_DIR=/scratch/bcbf/xingyao6/llm-agent
# IMAGE=/projects/bcbf/xingyao6/apptainer-images/pt-megatron-llm_v1.0.sif
# echo "MODEL_DIR=$MODEL_DIR"
# echo "WORK_DIR=$WORK_DIR"
# echo "IMAGE=$IMAGE"

# apptainer run --nv \
#     --no-home \
#     --no-mount bind-paths \
#     --cleanenv \
#     --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
#     --env "WANDB_API_KEY=$WANDB_API_KEY" \
#     --writable-tmpfs \
#     --bind $WORK_DIR:/workspace \
#     --bind $MODEL_DIR:/models \
#     $IMAGE \
#     /bin/bash -c "cd /workspace && scripts/slurm/configs/finetune_4xA100_4tp.sh"

# sbatch scripts/slurm/configs/finetune_4xA100_4tp.slurm
# squeue -u $USER
# scancel jobid