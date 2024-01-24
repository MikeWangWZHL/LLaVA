
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"
GEO_DATA_DIR="/data/wangz3/projects/llava_data/geo-mix"
DEEPSPEED="zero2"

GEO_CONFIG_JSON="${CODE_DIR}/llava/train/llava_geo_configs/llava_geo_earlyfusion_kd.json"

# ### job 1: llava geo early fusion: stage 1 from scratch: all stage1 + geo l1%
# # ### stage 1 ###
# SAVE_PER_STEPS=1000 # 24000
# BITS=16 # 16
# BATCH_SIZE=16 # 16
# GRAD_ACC_STEP=1 # 1

# MODEL_PATH="lmsys/vicuna-7b-v1.5"
# PRETRAINED_PROJECTOR_PATH="./checkpoints/projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

# MODEL_TYPE="llava_geo_early_fusion_kd"
# LR=1e-4 # 2e-4

# OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-558k_geoprojonly_linear_stage1_v2

# DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage1_ori-558k_geo-l1-69k.json

# # deepspeed --include localhost:4 llava/train/train_mem.py \
# deepspeed --include localhost:3,4,5,6 llava/train/train_mem.py \
#     --deepspeed ${CODE_DIR}/scripts/${DEEPSPEED}.json \
#     --model_name_or_path ${MODEL_PATH} \
#     --version plain \
#     --data_path ${DATA_PATH} \
#     --image_folder ${DATA_DIR} \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --pretrain_mm_mlp_adapter ${PRETRAINED_PROJECTOR_PATH} \
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
#     --llava_geo_config_path ${GEO_CONFIG_JSON} \
#     --tune_mm_mlp_adapter False \
#     --tune_sam_adapter True \
#     --use_geo_image_features_only True



# ### job 2: llava geo early fusion: stage 2 from scratch: all stage2 + geo l1%

## NOTE: has Nan loss problem
SAVE_PER_STEPS=1000 # 1000

MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-558k_geoprojonly_linear_stage1_v2
OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_linear_stage2_v2_lora

MODEL_TYPE="llava_geo_early_fusion_kd"
LR=1e-4
PROJ_LR=2e-5

BITS=16 # 16
BATCH_SIZE=16 # 16
GRAD_ACC_STEP=1 # 1

DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage2_ori-665k_geo-l1-69k.json

# deepspeed --include localhost:4 llava/train/train_mem.py \
deepspeed --include localhost:3,4,5,6 --master_port 29501 llava/train/train_mem.py \
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
    --llava_geo_config_path ${GEO_CONFIG_JSON} \
    --tune_mm_mlp_adapter True \
    --tune_sam_adapter True


### merge the lora model above
echo "merge the lora model above..."
MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-558k_geoprojonly_linear_stage1_v2
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_linear_stage2_v2_lora
MODEL_NAME=llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_linear_stage2_v2_lora
OUTPUT_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_linear_stage2_v2_lora_merged

python scripts/merge_lora_weights_geo.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --model-name ${MODEL_NAME} \
    --save-model-path ${OUTPUT_PATH}


### job 3: stage 3 with geo-l1 + ori 5%
echo "second lora training..."
SAVE_PER_STEPS=1000 # 1000

MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_linear_stage2_v2_lora_merged
OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_lora_merged_ori-33k_l1_linear_stage3_v2_lora

DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage2_ori-33k_geo-l1-69k.json

MODEL_TYPE="llava_geo_early_fusion_kd"
LR=1e-4
PROJ_LR=2e-5

BITS=16 # 16
BATCH_SIZE=8 # 16
GRAD_ACC_STEP=2 # 1

deepspeed --include localhost:3,4,5,6 --master_port 29501 llava/train/train_mem.py \
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
    --llava_geo_config_path ${GEO_CONFIG_JSON} \
    --tune_mm_mlp_adapter True \
    --tune_sam_adapter True




CUDA_ID=5
MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_linear_stage2_v2_lora_merged
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_llavageo-sam-kd/llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_lora_merged_ori-33k_l1_linear_stage3_v2_lora
MODEL_NAME="llava_geo_early_fusion_kd_7b_ori-665k_geo-l1_lora_merged_ori-33k_l1_linear_stage3_v2_lora"
cd /data/wangz3/projects/ecole-gvs-method
bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH

