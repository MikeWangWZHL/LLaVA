import argparse
from llava.model.builder import load_pretrained_model, load_pretrained_model_geo
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):

    if args.model_name is None:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = args.model_name
        model_name = model_name.replace("/", "_")

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')
    
    tokenizer, model, image_processors, context_len = load_pretrained_model_geo(
        args.model_path, 
        args.model_base, 
        model_name,
        device_map='cpu'
    )

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
