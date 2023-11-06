import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# write a simple test function doing some tensor manipulation using gpu
def test_gpu():
    x = torch.rand(5, 3).to('cuda')
    y = torch.rand(5, 3).to('cuda')
    z = x + y
    print(z)


def test_llava():
    print(IMAGE_TOKEN_INDEX)

if __name__ == '__main__':
    print('testing...')
    test_gpu()
    test_llava()