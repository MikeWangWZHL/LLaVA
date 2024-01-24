
# ### llava bench
# echo "Llava bench results"
# #
# MODEL_NAME=llava-v1.5-7b
# bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_7b_finetune_geo_l1_only_v2_lora
# bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_7b_finetune_ori-33k_geo_l1-69k_v2_lora
# bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=llava_geo_early_fusion_7b_ori-33k_geo-l1-69k_stage2_v2_lora
# bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
# echo "====================="

# MODEL_NAME=original_llava_v1.5-7b_finetune_ori_33k_only_v2_lora
# bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
# echo "====================="

# MODEL_NAME=llava_geo_early_fusion_7b_ori-55k_only_stage2_v2_lora
# bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
# echo "====================="

MODEL_NAME=llava_geo_early_fusion_7b_fullstage2data_geo-l1_tune_geo_projector_ori-33k_geo-l1_stage2_v2_lora
bash scripts/eval_llava_geo/llavabench_show_results.sh $MODEL_NAME
echo "====================="



# ### MME bench
# echo "MME results"
# #
# MODEL_NAME=llava-v1.5-7b
# bash scripts/eval_llava_geo/mme_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_7b_finetune_geo_l1_only_v2_lora
# bash scripts/eval_llava_geo/mme_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_7b_finetune_ori-33k_geo_l1-69k_v2_lora
# bash scripts/eval_llava_geo/mme_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=llava_geo_early_fusion_7b_ori-33k_geo-l1-69k_stage2_v2_lora
# bash scripts/eval_llava_geo/mme_show_results.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_v1.5-7b_finetune_ori_33k_only_v2_lora
# bash scripts/eval_llava_geo/mme_show_results.sh $MODEL_NAME
# echo "====================="

MODEL_NAME=llava_geo_early_fusion_7b_fullstage2data_geo-l1_tune_geo_projector_ori-33k_geo-l1_stage2_v2_lora
bash scripts/eval_llava_geo/mme_show_results.sh $MODEL_NAME
echo "====================="


# ### MMbench
# echo "MMbench convert results"
# #
# MODEL_NAME=llava-v1.5-7b
# bash scripts/eval_llava_geo/mmbench_convert_answers.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_7b_finetune_geo_l1_only_v2_lora
# bash scripts/eval_llava_geo/mmbench_convert_answers.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=original_llava_7b_finetune_ori-33k_geo_l1-69k_v2_lora
# bash scripts/eval_llava_geo/mmbench_convert_answers.sh $MODEL_NAME
# echo "====================="

# #
# MODEL_NAME=llava_geo_early_fusion_7b_ori-33k_geo-l1-69k_stage2_v2_lora
# bash scripts/eval_llava_geo/mmbench_convert_answers.sh $MODEL_NAME
# echo "====================="


