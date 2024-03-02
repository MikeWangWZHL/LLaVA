#!/bin/bash

# ### one gpu ###
srun \
    -A bcdq-delta-gpu \
    --time=01:00:00 \
    --nodes=1 \
    --ntasks-per-node=16 \
    --tasks=1 \
    --cpus-per-task=16 \
    --partition=gpuA40x4 \
    --gpus=1 \
    --mem=208g \
    --pty /bin/bash


### multiple gpu ###
# srun \
#     -A bcdq-delta-gpu \
#     --time=00:30:00 \
#     --nodes=1 \
#     --ntasks-per-node=16 \
#     --tasks=1 \
#     --cpus-per-task=16 \
#     --partition=gpuA40x4 \
#     --gpus=4 \
#     --mem=208g \
#     --pty /bin/bash

# # ### multiple gpu 8 A100 ###
# srun \
#     -A bcdq-delta-gpu \
#     --time=00:30:00 \
#     --nodes=1 \
#     --ntasks-per-node=16 \
#     --tasks=1 \
#     --cpus-per-task=16 \
#     --partition=gpuA100x8 \
#     --gpus=8 \
#     --mem=208g \
#     --pty /bin/bash