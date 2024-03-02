#!/bin/bash

CODE_DIR="/scratch/bcdq/wangz3/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/scratch/bcdq/wangz3/llava_data"

MODEL_PATH="liuhaotian/llava-v1.5-7b"
MODEL_TYPE="llava"
BATCH_SIZE=1 # 16
NUM_GPU=4
GRAD_ACC_STEP=8 # 1
NUM_EPOCH=1
LR=1e-5


DATA_PATH=$1
echo "DATA_PATH: ${DATA_PATH}"
OUTPUT_DIR=$2
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
NUM_INSTANCES=$3
NUM_ITERATIONS_PER_EPOCH=$((NUM_INSTANCES / (BATCH_SIZE * NUM_GPU)  / GRAD_ACC_STEP))
SAVE_PER_STEPS=$NUM_ITERATIONS_PER_EPOCH
echo "NUM_ITERATIONS_PER_EPOCH: ${NUM_ITERATIONS_PER_EPOCH}"
echo "SAVE_PER_STEPS: ${SAVE_PER_STEPS}"

# mm_projector_path=${CODE_DIR}/checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin
# deepspeed --include localhost:0 --master_port 29502 llava/train/train_mem.py \
deepspeed --include localhost:0,1,2,3 --master_port 29502 llava/train/train_mem.py \
    --deepspeed ${CODE_DIR}/scripts/zero3.json \
    --tune_mm_mlp_adapter True \
    --tune_llm True \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${DATA_DIR} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs $NUM_EPOCH \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACC_STEP} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_PER_STEPS} \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --model_type ${MODEL_TYPE}
    # --pretrain_mm_mlp_adapter ${mm_projector_path} \
    # --group_by_modality_length True \