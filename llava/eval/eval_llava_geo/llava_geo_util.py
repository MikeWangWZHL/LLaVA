
import os
import sys
CFP = os.path.abspath(__file__)
PWD = os.path.dirname(CFP)
sys.path.append(os.path.join(PWD, '../../../../../')) # ecole-gvs-method

from llava_geo_inference import *


def load_llava_geo_model(args):
    disable_torch_init()

    # set up model
    if args.model_name is None:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = args.model_name

    tokenizer, model, image_processors, context_len = load_pretrained_model_geo(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.load_8bit, 
        args.load_4bit, 
        device=args.device
    )
    
    # set up conversation
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower() or "geo" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "llava_v0" in model_name.lower():
        conv_mode = "llava_v0"
    else:
        conv_mode = "llava_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    print("model base:", args.model_base)
    print("model path:", args.model_path)
    print("model name:", model_name)
    print("conv_mode:", args.conv_mode)
    return tokenizer, model, image_processors, model_name