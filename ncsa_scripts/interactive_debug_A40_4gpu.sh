#!/bin/bash

### multiple gpu ###
srun \
    -A bcdq-delta-gpu \
    --time=00:30:00 \
    --nodes=1 \
    --ntasks-per-node=16 \
    --tasks=1 \
    --cpus-per-task=16 \
    --partition=gpuA40x4 \
    --gpus=4 \
    --mem=208g \
    --pty /bin/bash