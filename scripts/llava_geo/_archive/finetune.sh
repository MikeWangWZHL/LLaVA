#!/bin/bash

LLAVA_FINETUNE_DATA_DIR="/data/wangz3/projects/llava_data"

# in this setting we freeze llm and only finetune the mae decoder and mm projector
SAVE_PER_STEPS=2000 # 50000
TRAIN_BATCH=8 # 16: for 8 gpu
GRADIENT_ACCUMULATE=2 # 1 for 8 gpu
LR=5e-5 # 2e-5
# llm frozen version
deepspeed --include localhost:1,2,3,4 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ${LLAVA_FINETUNE_DATA_DIR}/llava_v1_5_mix665k.json \
    --image_folder ${LLAVA_FINETUNE_DATA_DIR} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava_geo_7b/instruction_finetune_v0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${TRAIN_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
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
    --model_type llava_geo \
    --llava_geo_config_path ./llava/train/llava_geo_configs/llava_geo_v1.json \
    --tune_mae_decoder True

# # llm open version
# deepspeed --include localhost:1,2,3,4 llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path liuhaotian/llava-v1.5-7b \
#     --version v1 \
#     --data_path ${LLAVA_FINETUNE_DATA_DIR}/llava_v1_5_mix665k.json \
#     --image_folder ${LLAVA_FINETUNE_DATA_DIR} \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava_geo_7b/instruction_finetune_v0 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size ${TRAIN_BATCH} \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps ${SAVE_PER_STEPS} \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --model_type llava_geo \
#     --llava_geo_config_path ./llava/train/llava_geo_configs/llava_geo_v1.json \
#     --tune_mae_decoder True
#     # --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \

# # original llava
# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb
