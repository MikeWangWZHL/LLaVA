
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "/data/wangz3/example_images/image (1).png" \