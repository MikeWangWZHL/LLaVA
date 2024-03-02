
#!/bin/bash
CODE_DIR="/scratch/bcdq/wangz3/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/scratch/bcdq/wangz3/llava_data/LLaVA-Pretrain"

SAVE_PER_STEPS=1000 # 24000
LR=5e-4 # 1e-3

# MODEL_PATH="lmsys/vicuna-13b-v1.5"
# PRETRAINED_PROJECTOR_PATH="./checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin"

MODEL_PATH="lmsys/vicuna-7b-v1.5"
PRETRAINED_PROJECTOR_PATH="${CODE_DIR}/checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

MODEL_TYPE="llava_geo_early_fusion"

# 32 * 1 * 8 = 256
# 16 * 4 * 4 = 256

BITS=16 # 16
BATCH_SIZE=8 # 32
GRAD_ACC_STEP=4 # 1

# NOTE: using fp16 instead of bf16 due to SAM model not implemented for bf16
    # --bf16 True \

# deepspeed --include localhost:1,2,3,4 llava/train/train_mem.py \
# deepspeed --include localhost:0,1,2,3 llava/train/train_mem.py \
deepspeed --include localhost:0 llava/train/train_mem.py \
    --deepspeed ${CODE_DIR}/scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version plain \
    --data_path ${DATA_DIR}/blip_laion_cc_sbu_558k.json \
    --image_folder ${DATA_DIR}/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${PRETRAINED_PROJECTOR_PATH} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --bits ${BITS} \
    --output_dir ${CODE_DIR}/checkpoints/llava_geo_7b/earlyfusion \
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
    --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_earlyfusion.json" \
    --tune_mm_mlp_adapter False \
    --tune_sam_adapter True