
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/data/wangz3/projects/llava_data"
GEO_DATA_DIR="/data/wangz3/projects/llava_data/geo-mix"
DEEPSPEED="zero2"

for ABLATION_NAME in "wo_2d_shapes_comparison" "wo_2d_shapes_single" "wo_CLEVR_v1.0" "wo_geoclidean_qa" "wo_geometry3k" "wo_sketch" "wo_vsr"
do
    CUDA_ID=3
    MODEL_BASE="liuhaotian/llava-v1.5-7b"
    MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2_ablation/original_llava_7b_finetune_ori-33k_geo_l1-69k_${ABLATION_NAME}
    MODEL_NAME=ablation__original_llava_7b_finetune_ori-33k_geo_l1-69k_${ABLATION_NAME}_lora
    echo "run eval: ${MODEL_NAME} on CUDA_ID: ${CUDA_ID}"
    cd /data/wangz3/projects/ecole-gvs-method
    # bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH
    bash scripts/run_llava_geo_lora_inference_all.sh MMMU $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH
    echo ""
done

