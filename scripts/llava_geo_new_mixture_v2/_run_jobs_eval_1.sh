# CUDA_ID=0
# CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
# MODEL_BASE="liuhaotian/llava-v1.5-13b"
# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/original_llava_13b_finetune_ori-33k_geo_l1-69k_v2_lora

# MODEL_NAME="original_llava_13b_finetune_ori-33k_geo_l1-69k_v2_lora"

# cd /data/wangz3/projects/ecole-gvs-method
# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH

SRC_ROOT=/data/wangz3/projects/ecole-gvs-method
CUDA_ID=7
CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_13b_ori-55k_l1-69k_stage1_v2
MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_13b_ori-33k_l1-69k_stage2_v2_lora

MODEL_NAME="llava_geo_early_fusion_13b_ori-33k_l1-69k_stage2_v2_lora"

cd /data/wangz3/projects/ecole-gvs-method

# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH


cd ${SRC_ROOT}/third_party/ecole-gvs-benchmark/ecole_gvs_benchmark/tasks/third_party_benchmarks/MathVista
bash scripts/llava_geo/run_llava_geo_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH


# Other VL benchmarks
cd ${SRC_ROOT}/third_party/LLaVA
# llava bench
bash scripts/eval_llava_geo/llavabench.sh $CUDA_ID $MODEL_PATH $MODEL_BASE $MODEL_NAME

# mme
bash scripts/eval_llava_geo/mme.sh $CUDA_ID $MODEL_PATH $MODEL_BASE $MODEL_NAME

# mmbench
bash scripts/eval_llava_geo/mmbench.sh $CUDA_ID $MODEL_PATH $MODEL_BASE $MODEL_NAME