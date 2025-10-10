'''
@Description: 
@Author: jiajunlong
@Date: 2024-06-19 19:30:17
@LastEditTime: 2024-06-19 19:32:47
@LastEditors: jiajunlong
'''
import argparse
import json
import requests
from PIL import Image
from io import BytesIO

import torch
from transformers import TextStreamer

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()
    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model(model_name_or_path=args.model_path, load_8bit=args.load_8bit, load_4bit=args.load_4bit, device=args.device)
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.to(args.device)
    if getattr(text_processor.template, 'role', None) is None:
        roles = ['USER', 'ASSISTANT']
    else:
        roles = text_processor.template.role.apply()
    msg = Message()
    image = None
    image_tensor = None
    if args.image_file:
        image = load_image(args.image_file)
        image_tensor = image_processor(image)
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)

    sensor_payload = None
    if args.sensor is not None:
        sensor_payload = json.loads(args.sensor)

    sensor_used = False

    def run_generation(user_input, use_sensor=True):
        nonlocal image, image_tensor, sensor_used

        include_sensor = sensor_payload is not None and use_sensor and not sensor_used
        if include_sensor and DEFAULT_SENSOR_TOKEN not in user_input:
            user_input = DEFAULT_SENSOR_TOKEN + '\n' + user_input

        if image is not None:
            user_input = DEFAULT_IMAGE_TOKEN + '\n' + user_input
            image = None

        msg.add_message(user_input)
        result = text_processor(msg.messages, mode='eval')
        prompt = result['prompt']
        input_ids = result['input_ids'].unsqueeze(0).to(model.device)

        sensors = [sensor_payload] if include_sensor else None

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                sensors=sensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        msg.messages[-1]['value'] = outputs
        sensor_used = sensor_used or include_sensor

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        print(f"{roles[1]}: {outputs}")
        return outputs

    if args.prompt is not None:
        run_generation(args.prompt)
    else:
        while True:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break
            run_generation(inp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='phi')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional single-turn prompt. If provided, the CLI runs once and exits.")
    parser.add_argument("--sensor", type=str, default=None,
                        help="JSON string describing sensor payload (requires <sensor> token in prompt unless auto-prepended).")
    args = parser.parse_args()
    main(args)
