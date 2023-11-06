#!/bin/bash
LLAVA_FINETUNE_DATA_DIR="/data/wangz3/projects/llava_data/LLaVA-Pretrain"

SAVE_PER_STEPS=5000 # 5000
TRAIN_BATCH=32 # 16: for 8 gpu
GRADIENT_ACCUMULATE=1 # 1 for 8 gpu
LR=5e-5 # 2e-5
MODEL_PATH="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_7b/using_pretrained_mae/stage_1__pretrain_mae_adapter" 


# deepspeed --include localhost:1 llava/train/train_mem.py \
deepspeed --include localhost:1,2,3,4 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version plain \
    --data_path ${LLAVA_FINETUNE_DATA_DIR}/blip_laion_cc_sbu_558k.json \
    --image_folder ${LLAVA_FINETUNE_DATA_DIR}/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava_geo_7b/using_pretrained_mae/stage_1.5__pretrain_mae_adapter_and_projection \
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
    --llava_geo_config_path ./llava/train/llava_geo_configs/llava_geo_v2_pretrain_stage_1.5.json \
    --tune_mae_adapter True \
    --tune_mm_mlp_adapter True

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
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
