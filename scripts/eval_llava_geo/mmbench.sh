#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

SPLIT="mmbench_dev_20230712"

MODEL_PATH="liuhaotian/llava-v1.5-7b"
OUTPUT_MODEL_NAME="llava-v1.5-7b"

python llava/eval/eval_llava_geo/model_vqa_mmbench.py \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${OUTPUT_MODEL_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $OUTPUT_MODEL_NAME
