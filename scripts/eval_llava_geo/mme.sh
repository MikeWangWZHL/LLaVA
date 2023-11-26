#!/bin/bash
source scripts/setup_openai_api_key.sh
echo $OPENAI_API_KEY


# MODEL_PATH="liuhaotian/llava-v1.5-7b"
# OUTPUT_MODEL_NAME="llava-v1.5-7b"

CUDA_ID="$1"
MODEL_PATH="$2"
MODEL_BASE="$3"
OUTPUT_MODEL_NAME="$4"

export CUDA_VISIBLE_DEVICES=$CUDA_ID
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

if [ "$MODEL_BASE" == "None" ]; then
    python llava/eval/eval_llava_geo/model_vqa.py \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$OUTPUT_MODEL_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --max-new-tokens 128
else
    python llava/eval/eval_llava_geo/model_vqa.py \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$OUTPUT_MODEL_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --max-new-tokens 128
fi

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $OUTPUT_MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$OUTPUT_MODEL_NAME
