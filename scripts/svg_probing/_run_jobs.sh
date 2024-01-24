### train
# bash scripts/svg_probing/svg_text_lora.sh
# # bash scripts/svg_probing/image_text_lora_from_llava_stage1.sh
# bash scripts/svg_probing/image_text_lora_from_llava_stage2.sh

### eval
cd /data/wangz3/projects/ecole-gvs-method
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
CUDA_ID=5

# TASK_NAME="vsr"
TASK_NAME="whats_up"

# svg-text
MODEL_BASE="lmsys/vicuna-7b-v1.5"
MODEL_PATH="${CODE_DIR}/checkpoints/svg_probing/svg_text_lora/${TASK_NAME}"
MODEL_NAME="svg_text_lora_vicuna_7b_v1.5__${TASK_NAME}"
TASK_NAME_FULL="svg_probing_svg-text_${TASK_NAME}"
bash scripts/svg_probing_eval/eval_svg_probing.sh $TASK_NAME_FULL $CUDA_ID $MODEL_BASE $MODEL_PATH $MODEL_NAME

# # svg-image
# MODEL_BASE="liuhaotian/llava-v1.5-7b"
# MODEL_PATH="${CODE_DIR}/checkpoints/svg_probing/image_text_lora/${TASK_NAME}_from_llava_stage2"
# MODEL_NAME="image_text_lora_llava_7b_v1.5_stage2__${TASK_NAME}"
# TASK_NAME_FULL="svg_probing_image-text_${TASK_NAME}"
# bash scripts/svg_probing_eval/eval_svg_probing.sh $TASK_NAME_FULL $CUDA_ID $MODEL_BASE $MODEL_PATH $MODEL_NAME

# # original svg-text
# MODEL_BASE="None"
# MODEL_PATH="lmsys/vicuna-7b-v1.5"
# MODEL_NAME="lmsys_vicuna-7b-v1.5"
# TASK_NAME_FULL="svg_probing_svg-text_${TASK_NAME}"
# bash scripts/svg_probing_eval/eval_svg_probing.sh $TASK_NAME_FULL $CUDA_ID $MODEL_BASE $MODEL_PATH $MODEL_NAME

# # original image-text
# MODEL_BASE="None"
# MODEL_PATH="liuhaotian/llava-v1.5-7b"
# MODEL_NAME="liuhaotian_llava-v1.5-7b"
# TASK_NAME_FULL="svg_probing_image-text_${TASK_NAME}"
# bash scripts/svg_probing_eval/eval_svg_probing.sh $TASK_NAME_FULL $CUDA_ID $MODEL_BASE $MODEL_PATH $MODEL_NAME
