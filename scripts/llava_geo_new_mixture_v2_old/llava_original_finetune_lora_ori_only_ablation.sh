
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"
GEO_DATA_DIR="/data/wangz3/projects/llava_data/geo-mix"
DEEPSPEED="zero2"


SAVE_PER_STEPS=250 # 24000
LR=2e-4 # 2e-4

# job 1 ############################################
### fully tuned llava | geo-mix-v2-l1
echo "running job 0"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
MODEL_TYPE="llava"
BITS=16 # 16
BATCH_SIZE=16 # 16
GRAD_ACC_STEP=1 # 1
OUTPUT_DIR=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/original_llava_7b_finetune_ori_33k_only_v2_lora
DATA_PATH=${DATA_DIR}/llava_geo_mix_merged_v2_stage2_ori-33k_seperated.json

# deepspeed --include localhost:4 llava/train/train_mem.py \
deepspeed --include localhost:4,5,6,7 llava/train/train_mem.py \
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
#############################################


cd /data/wangz3/projects/ecole-gvs-method

CUDA_ID=4
MODEL_BASE="liuhaotian/llava-v1.5-7b"
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/original_llava_7b_finetune_ori_33k_only_v2_lora
MODEL_NAME="original_llava_v1.5-7b_finetune_ori_33k_only_v2_lora"
