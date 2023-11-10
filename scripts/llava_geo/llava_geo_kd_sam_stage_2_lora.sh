
#!/bin/bash
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"

SAVE_PER_STEPS=2500 # 24000
LR=2e-4 # 2e-4

# MODEL_PATH="lmsys/vicuna-13b-v1.5"
# PRETRAINED_PROJECTOR_PATH="./checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin"

MODEL_PATH="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_7b/llava_geo_kd/checkpoint-2500"
# MODEL_PATH="lmsys/vicuna-7b-v1.5"
# PRETRAINED_PROJECTOR_PATH="${CODE_DIR}/checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

MODEL_TYPE="llava_geo_kd"

# 32 * 1 * 8 = 256
# 16 * 4 * 4 = 256

BITS=16 # 16
BATCH_SIZE=16 # 16
GRAD_ACC_STEP=2 # 1

OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_7b/llava_geo_kd_sam_stage_2_lora

# NOTE: using fp16 instead of bf16 due to SAM model not implemented for bf16
    # --bf16 True \
    # --mm_projector_lr 2e-5 \ => used for all vision adapters including geo encoders like SAM, and original encoder CLIP

# deepspeed --include localhost:1,2,3,4 llava/train/train_mem.py \
# deepspeed --include localhost:2,3,4,5 llava/train/train_mem.py \
    # --deepspeed ${CODE_DIR}/scripts/zero2.json \
# deepspeed --include localhost:1 llava/train/train_mem.py \
deepspeed --include localhost:1,4,5,6 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${CODE_DIR}/scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_DIR}/llava_v1_5_mix665k.json \
    --image_folder ${DATA_DIR} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --bits ${BITS} \
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
    --model_type ${MODEL_TYPE} \
    --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_kd_stage2.json" \
    --tune_mm_mlp_adapter True \
    --tune_sam_adapter True