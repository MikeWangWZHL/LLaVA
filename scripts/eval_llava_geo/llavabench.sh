#!/bin/bash

source scripts/setup_openai_api_key.sh
echo $OPENAI_API_KEY


CUDA_ID="$1"
MODEL_PATH="$2"
MODEL_BASE="$3"
OUTPUT_MODEL_NAME="$4"

export CUDA_VISIBLE_DEVICES=$CUDA_ID
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "model path: ${MODEL_PATH}"
echo "model base: ${MODEL_BASE}"
echo "output model name: ${OUTPUT_MODEL_NAME}"

# MODEL_PATH="liuhaotian/llava-v1.5-7b"
# OUTPUT_MODEL_NAME="llava-v1.5-7b"

if [ "$MODEL_BASE" == "None" ]; then
    echo "Evaluating non-lora models."
    python llava/eval/eval_llava_geo/model_vqa.py \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$OUTPUT_MODEL_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
else
    echo "Evaluating lora models."
    python llava/eval/eval_llava_geo/model_vqa.py \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$OUTPUT_MODEL_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
fi

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

OUTPUT_GPT_REVIEW_NAME="${OUTPUT_MODEL_NAME}__gpt4_turbo_1106"
python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$OUTPUT_MODEL_NAME.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$OUTPUT_GPT_REVIEW_NAME.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$OUTPUT_GPT_REVIEW_NAME.jsonl
