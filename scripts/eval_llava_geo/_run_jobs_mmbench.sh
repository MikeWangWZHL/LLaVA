CUDA_ID=2

#
MODEL_NAME=llava_geo_early_fusion_7b_ori-33k_geo-l1-69k_stage2_v2_lora
MODEL_BASE="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-55k_geo-l1-69k_stage1_v2"
MODEL_PATH="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-33k_geo-l1-69k_stage2_v2_lora"
bash scripts/eval_llava_geo/mmbench.sh $CUDA_ID $MODEL_PATH $MODEL_BASE $MODEL_NAME

#
MODEL_NAME=original_llava_7b_finetune_geo_l1_only_v2_lora
MODEL_BASE="liuhaotian/llava-v1.5-7b"
MODEL_PATH="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/original_llava_7b_finetune_geo_l1_only_v2_lora"
bash scripts/eval_llava_geo/mmbench.sh $CUDA_ID $MODEL_PATH $MODEL_BASE $MODEL_NAME

#
MODEL_NAME=original_llava_7b_finetune_ori-33k_geo_l1-69k_v2_lora
MODEL_BASE="liuhaotian/llava-v1.5-7b"
MODEL_PATH="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/original_llava_7b_finetune_ori-33k_geo_l1-69k_v2_lora"
bash scripts/eval_llava_geo/mmbench.sh $CUDA_ID $MODEL_PATH $MODEL_BASE $MODEL_NAME
