## training
# bash scripts/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_original_finetune_lora.sh
# bash scripts/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_lora.sh
# bash scripts/llava_geo_new_mixture_v2/llava_geo_early_fusion_full_two_stage_plus_geo_l1_lora.sh
# bash scripts/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_full_two_stage_plus_geo_l1_lora.sh

# added ablation
# echo "ablation v1 mix"
# bash scripts/llava_geo_new_mixture_v1_ablation/llava_original_finetune_lora.sh




# ### eval jobs ###
# CODE_DIR="/data/wangz3/projects/ecole-gvs-method/third_party/LLaVA"
# DATA_DIR="/data/wangz3/projects/llava_data"
# GEO_DATA_DIR="/data/wangz3/projects/llava_data/geo-mix"
# CUDA_ID=4


# # previous eval jobs
# MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-665k_geo-l1_mlp_stage2_v2_lora_merged
# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_v2/llava_geo_early_fusion_7b_ori-665k_geo-l1_lora_merged_ori-33k_l1_mlp_stage3_v2_lora
# MODEL_NAME="llava_geo_early_fusion_7b_ori-665k_geo-l1_lora_merged_ori-33k_l1_mlp_stage3_v2_lora"
# cd /data/wangz3/projects/ecole-gvs-method
# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH


# # --------------- #
# # job 1
# MODEL_BASE="liuhaotian/llava-v1.5-7b"
# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/original_llava_7b_finetune_ori-33k_geo_l1-69k_sharegpt4v-40k_lora
# MODEL_NAME="original_llava_7b_finetune_ori-33k_geo_l1-69k_sharegpt4v-40k_lora"
# cd /data/wangz3/projects/ecole-gvs-method
# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH

# --------------- #
# # job 2
# MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_7b_ori-55k_l1-69k_sharegpt4v-81k_stage1
# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_7b_ori-55k_l1-69k_sharegpt4v-81k-41k_stage2_lora
# MODEL_NAME="llava_geo_early_fusion_7b_ori-55k_l1-69k_sharegpt4v-81k-41k_stage2_lora"
# cd /data/wangz3/projects/ecole-gvs-method
# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH

# # --------------- #
# # job 3.1
# MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_7b_ori-558k_sharegpt4v-81k_geoprojonly_mlp_stage1
# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_7b_ori-665k_sharegpt4v-81k_mlp_stage2_lora_merged
# MODEL_NAME="llava_geo_early_fusion_7b_ori-665k_sharegpt4v-81k_mlp_stage2_lora_merged"
# cd /data/wangz3/projects/ecole-gvs-method
# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH

# # job 3.2
# MODEL_BASE=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_7b_ori-665k_sharegpt4v-81k_mlp_stage2_lora_merged
# MODEL_PATH=${CODE_DIR}/checkpoints/llava_geo_new_mixture_w_sharegpt4v_geofiltered/llava_geo_early_fusion_7b_ori-665k_33k_l1-69k_sharegpt4v-81k_40k_mlp_stage3_lora
# MODEL_NAME=llava_geo_early_fusion_7b_ori-665k_33k_l1-69k_sharegpt4v-81k_40k_mlp_stage3_lora
# cd /data/wangz3/projects/ecole-gvs-method
# bash scripts/_run_eval_everything_lora.sh $CUDA_ID $MODEL_NAME $MODEL_BASE $MODEL_PATH


