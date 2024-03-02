CODE_DIR="/scratch/bcdq/wangz3/ecole-gvs-method/third_party/LLaVA"
DATA_DIR="/scratch/bcdq/wangz3/llava_data"

DATA_PATH=${DATA_DIR}/shapeworld/train_image-scene.json
OUTPUT_DIR=${CODE_DIR}/checkpoints/svg_tasks/img_2_scene_json__shapeworld__fullfinetune_ep1_from_llava_stage2
NUM_INSTANCES=15000
bash ncsa_scripts/train_scripts/svg_tasks/image_text_fullfinetune_from_llava_stage2.sh $DATA_PATH $OUTPUT_DIR $NUM_INSTANCES