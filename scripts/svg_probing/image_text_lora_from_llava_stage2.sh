#!/bin/bash

CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"
DEEPSPEED="zero2"
SAVE_PER_STEPS=500

MODEL_PATH="liuhaotian/llava-v1.5-7b"
MODEL_TYPE="llava"
BATCH_SIZE=8 # 16
GRAD_ACC_STEP=2 # 1

for TASK_NAME in "line_or_angle" "single_angle" "lines" "intersect_horizontal" "clevr_easy"
do
    echo "run task: ${TASK_NAME}"
    DATA_PATH=${DATA_DIR}/svg_probing/${TASK_NAME}/train_image.json
    OUTPUT_DIR=${CODE_DIR}/checkpoints/svg_probing/image_text_lora/${TASK_NAME}_from_llava_stage2
    # mm_projector_path=${CODE_DIR}/checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin

    LR=2e-5
    deepspeed --include localhost:2,7 --master_port 29502  llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed ${CODE_DIR}/scripts/${DEEPSPEED}.json \
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
        --num_train_epochs 1 \
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
done