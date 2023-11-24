#!/bin/bash

source scripts/setup_openai_api_key.sh
echo $OPENAI_API_KEY

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


MODEL_PATH="liuhaotian/llava-v1.5-7b"
OUTPUT_MODEL_NAME="llava-v1.5-7b"


# python llava/eval/model_vqa_loader.py \
python llava/eval/eval_llava_geo/model_vqa.py \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$OUTPUT_MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --max-new-tokens 128

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $OUTPUT_MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$OUTPUT_MODEL_NAME
