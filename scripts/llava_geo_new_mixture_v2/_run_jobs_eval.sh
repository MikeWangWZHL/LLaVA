

CUDA_ID=4
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
MODEL_BASE="liuhaotian/llava-v1.5-13b"
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/original_llava_13b_finetune_ori-33k_geo_l1-69k_v2_lora

MODEL_NAME="original_llava_13b_finetune_ori-33k_geo_l1-69k_v2_lora"

cd /data/wangz3/projects/ecole-gvs-method
bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH



CUDA_ID=4
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_13b_ori-55k_l1-69k_stage1_v2
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_13b_ori-33k_l1-69k_stage2_v2_lora

MODEL_NAME="llava_geo_early_fusion_13b_ori-33k_l1-69k_stage2_v2_lora"

cd /data/wangz3/projects/ecole-gvs-method
bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH