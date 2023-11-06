#!/bin/bash

LLAVA_FINETUNE_DATA_DIR="/data/wangz3/projects/llava_data"

# in this setting we freeze llm and only finetune the mae decoder and mm projector
SAVE_PER_STEPS=5000 # 50000
TRAIN_BATCH=32 # 16: for 8 gpu
GRADIENT_ACCUMULATE=1 # 1 for 8 gpu
LR=1e-4 # 2e-5
# llm frozen version

# MODEL_PATH="lmsys/vicuna-7b-v1.5" # liuhaotian/llava-v1.5-7b
# MODEL_PATH="liuhaotian/llava-v1.5-7b"
# OUTPUT_DIR="./checkpoints/llava_geo_7b/debug_reconstruction_only_no_projection_3epochs"
# EPOCH=3
# MODEL_TYPE="debug"
# IF_FREEZE_PROJECTION="True"
# TUNE_VISION_TOWER="False"


MODEL_PATH="liuhaotian/llava-v1.5-7b"
OUTPUT_DIR="./checkpoints/llava_geo_7b/debug_reconstruction_only_no_projection_tune_clip_encoder_3epochs"
EPOCH=3
MODEL_TYPE="debug"
IF_FREEZE_PROJECTION="True"
TUNE_VISION_TOWER="True"


# deepspeed --include localhost:1 llava/train/train_mem.py \
deepspeed --include localhost:1,2,3,4 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
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
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCH} \
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
    --model_type ${MODEL_TYPE} \
    --llava_geo_config_path ./llava/train/llava_geo_configs/llava_geo_debug_reconstruction.json \
    --tune_mae_decoder True \
    --freeze_mm_mlp_adapter ${IF_FREEZE_PROJECTION} \
    --tune_vision_tower ${TUNE_VISION_TOWER}
