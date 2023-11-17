
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"
GEO_DATA_DIR="/data/wangz3/projects/llava_data/geo-mix"

SAVE_PER_STEPS=1000 # 24000
LR=2e-4 # 2e-4

MODEL_PATH="liuhaotian/llava-v1.5-7b"

MODEL_TYPE="llava"

BITS=16 # 16
BATCH_SIZE=16 # 16
GRAD_ACC_STEP=1 # 1

OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture/original_llava_7b_finetune_ori_subset_and_geo_lora

# DATA_PATH=${DATA_DIR}/llava_v1_5_mix665k.json
# DATA_PATH=${GEO_DATA_DIR}/geo_mix_v1_76k.json
DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v1_ori-66k_geo-76k.json

# deepspeed --include localhost:4 llava/train/train_mem.py \
deepspeed --include localhost:4,5,6,7 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${CODE_DIR}/scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
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
    --model_type ${MODEL_TYPE}
    # --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_kd_stage2.json" \
    # --tune_mm_mlp_adapter True \
    # --tune_sam_adapter True