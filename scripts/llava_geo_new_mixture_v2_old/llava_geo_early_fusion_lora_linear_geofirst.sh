
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"
GEO_DATA_DIR="/data/wangz3/projects/llava_data/geo-mix"
DEEPSPEED="zero2"

### job 1: stage 1 with geo-l1 + ori 5%
# ### stage 1 ###
SAVE_PER_STEPS=1000 # 24000
BITS=16 # 16
BATCH_SIZE=16 # 16
GRAD_ACC_STEP=2 # 1

MODEL_TYPE="llava_geo_early_fusion"
LR=1e-4 # 2e-4

MODEL_PATH="liuhaotian/llava-v1.5-7b"
OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/linear_geofirst/llava_geo_early_fusion_7b_ori-55k_geo-l1-69k_stage1_v2_geofirst

DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage1_ori-55k_geo-l1-69k.json

# deepspeed --include localhost:4 llava/train/train_mem.py \
deepspeed --include localhost:0,1,3,4 llava/train/train_mem.py \
    --deepspeed ${CODE_DIR}/scripts/${DEEPSPEED}.json \
    --model_name_or_path ${MODEL_PATH} \
    --version plain \
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
    --model_type ${MODEL_TYPE} \
    --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_earlyfusion_linear_geofirst.json" \
    --tune_mm_mlp_adapter False \
    --tune_sam_adapter True

### job 2: stage 1 with geo-l1 + ori 5%
# #### stage 2 ####

SAVE_PER_STEPS=1000 # 1000

MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/linear_geofirst/llava_geo_early_fusion_7b_ori-55k_geo-l1-69k_stage1_v2_geofirst
OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/linear_geofirst/llava_geo_early_fusion_7b_ori-33k_geo-l1-69k_stage2_v2_geofirst_lora

DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage2_ori-33k_geo-l1-69k.json


MODEL_TYPE="llava_geo_early_fusion"
LR=1e-4
PROJ_LR=2e-5

BITS=16 # 16
BATCH_SIZE=8 # 16
GRAD_ACC_STEP=2 # 1

deepspeed --include localhost:0,1,3,4 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr ${PROJ_LR} \
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
    --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_earlyfusion_linear_geofirst.json" \
    --tune_mm_mlp_adapter True \
    --tune_sam_adapter True






######################
######################


# ### job 3: stage 1 with geo-l1-l2 + ori 5%
# # ### stage 1 ###
# SAVE_PER_STEPS=1000 # 24000
# BITS=16 # 16
# BATCH_SIZE=16 # 16
# GRAD_ACC_STEP=2 # 1

# MODEL_PATH="liuhaotian/llava-v1.5-7b"
# MODEL_TYPE="llava_geo_early_fusion"
# LR=1e-4 # 2e-4
# OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-55k_geo-l1-l2-88k_stage1_v2

# DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage1_ori-55k_geo-l1l2-88k.json

# # deepspeed --include localhost:4 llava/train/train_mem.py \
# deepspeed --include localhost:4,5,6,7 llava/train/train_mem.py \
#     --deepspeed ${CODE_DIR}/scripts/${DEEPSPEED}.json \
#     --model_name_or_path ${MODEL_PATH} \
#     --version plain \
#     --data_path ${DATA_PATH} \
#     --image_folder ${DATA_DIR} \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --fp16 True \
#     --bits ${BITS} \
#     --output_dir ${OUTPUT_DIR} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps ${GRAD_ACC_STEP} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps ${SAVE_PER_STEPS} \
#     --save_total_limit 1 \
#     --learning_rate ${LR} \
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
#     --model_type ${MODEL_TYPE} \
#     --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_earlyfusion.json" \
#     --tune_mm_mlp_adapter False \
#     --tune_sam_adapter True

### job 2: stage 2 with geo-l1l2 + ori 5% lora
# #### stage 2 ####

# SAVE_PER_STEPS=1000 # 1000

# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-55k_geo-l1-l2-88k_stage1_v2
# # OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-33k_geo-l1-l2-88k_stage2_v2_lora
# OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-6k_geo-l1-l2-88k_stage2_v2_lora

# # DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage2_ori-33k_geo-l1l2-88k.json
# DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage2_ori-33k_geo-l1l2-88k.json

# MODEL_TYPE="llava_geo_early_fusion"
# LR=1e-4
# PROJ_LR=2e-5

# BITS=16 # 16
# BATCH_SIZE=8 # 16
# GRAD_ACC_STEP=2 # 1

# deepspeed --include localhost:4,5,6,7 llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr ${PROJ_LR} \
#     --deepspeed ${CODE_DIR}/scripts/${DEEPSPEED}.json \
#     --model_name_or_path ${MODEL_PATH} \
#     --version v1 \
#     --data_path ${DATA_PATH} \
#     --image_folder ${DATA_DIR} \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --fp16 True \
#     --bits ${BITS} \
#     --output_dir ${OUTPUT_DIR} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps ${GRAD_ACC_STEP} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps ${SAVE_PER_STEPS} \
#     --save_total_limit 1 \
#     --learning_rate ${LR} \
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
#     --model_type ${MODEL_TYPE} \
#     --llava_geo_config_path "${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_earlyfusion.json" \
#     --tune_mm_mlp_adapter True \
#     --tune_sam_adapter True