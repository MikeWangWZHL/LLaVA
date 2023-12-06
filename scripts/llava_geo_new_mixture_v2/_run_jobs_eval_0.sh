echo "sleep 32400"
sleep 32400

CUDA_ID=4
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
MODEL_BASE="liuhaotian/llava-v1.5-13b"
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/original_llava_13b_finetune_ori-33k_geo_l1-69k_v2_lora

MODEL_NAME="original_llava_13b_finetune_ori-33k_geo_l1-69k_v2_lora"

cd /data/wangz3/projects/ecole-gvs-method
bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH