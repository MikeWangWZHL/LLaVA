MODEL_BASE=/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-55k_only_stage1_v2
MODEL_PATH=/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-55k_only_stage2_v2_lora
MODEL_NAME=llava_geo_early_fusion_7b_ori-55k_only_stage2_v2_lora
OUTPUT_PATH=/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-55k_only_stage2_v2_lora_merged


python scripts/merge_lora_weights_geo.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --model-name ${MODEL_NAME} \
    --save-model-path ${OUTPUT_PATH}