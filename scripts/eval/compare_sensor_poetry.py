#!/usr/bin/env python3
"""Compare finetuned vs. baseline TinyLLaVA models on a shared sensor prompt."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

from transformers import AutoTokenizer

from tinyllava.model.load_model import load_pretrained_model
from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.serve.cli import load_image
from tinyllava.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SENSOR_TOKEN,
    Message,
    disable_torch_init,
)
from tinyllava.data import ImagePreprocess, TextPreprocess


@dataclass
class GenerationResult:
    tag: str
    prompt: str
    attach_sensor: bool
    output_text: str


def _load_model_with_fallback(model_path: str, device: str):
    """Load TinyLLaVA checkpoint, supporting sharded factory outputs."""
    try:
        return load_pretrained_model(model_name_or_path=model_path, device=device)
    except OSError as err:
        # Fall back to factory checkpoint layout (separate submodules).
        if "no file named" not in str(err).lower():
            raise

    config = TinyLlavaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=config.tokenizer_use_fast,
        padding_side=config.tokenizer_padding_side,
        trust_remote_code=True,
    )
    model = TinyLlavaForConditionalGeneration(config)
    model.tokenizer = tokenizer

    text_dtype = getattr(model.config.text_config, "torch_dtype", None)
    dtype = torch.float16 if text_dtype in (None, torch.float16, "float16") else torch.float32
    llm_kwargs = dict(
        model_name_or_path=config.llm_model_name_or_path,
        pretrained_llm_path=os.path.join(model_path, "language_model"),
        torch_dtype=dtype,
    )
    model.load_llm(**llm_kwargs)
    model.load_vision_tower(
        model_name_or_path=config.vision_model_name_or_path,
        pretrained_vision_tower_path=os.path.join(model_path, "vision_tower"),
    )
    model.load_connector(pretrained_connector_path=os.path.join(model_path, "connector"))
    if config.sensor_encoder_type:
        model.load_sensor_encoder(
            sensor_encoder_type=config.sensor_encoder_type,
            pretrained_sensor_encoder_path=os.path.join(model_path, "sensor_encoder"),
        )

    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, "tokenizer_model_max_length", 2048)
    return model, tokenizer, image_processor, context_len


def run_single(
    *,
    model_path: str,
    tag: str,
    prompt: str,
    sensor_json: Optional[str],
    image_path: Optional[str],
    attach_sensor: bool,
    device: str,
    conv_mode: str,
    temperature: float,
    max_new_tokens: int,
) -> GenerationResult:
    """Generate a single response for the given model configuration."""
    disable_torch_init()
    torch.set_grad_enabled(False)

    model, tokenizer, image_processor, _ = _load_model_with_fallback(
        model_path=model_path,
        device=device,
    )
    model.eval()

    text_processor = TextPreprocess(tokenizer, conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    model.to(device)

    vision_dtype = None
    vision_module = getattr(model.vision_tower, "_vision_tower", None)
    if vision_module is not None:
        try:
            vision_param = next(vision_module.parameters())
        except StopIteration:
            vision_param = None
        if vision_param is not None:
            vision_dtype = vision_param.dtype
    if vision_dtype is None:
        try:
            vision_dtype = next(model.parameters()).dtype
        except StopIteration:
            vision_dtype = torch.float32

    msg = Message()
    user_prompt = prompt

    image_tensor = None
    if image_path:
        pil_image = load_image(image_path)
        image_tensor = image_processor(pil_image)
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=vision_dtype)
        if DEFAULT_IMAGE_TOKEN not in user_prompt:
            user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{user_prompt}"

    sensor_payload = None
    if attach_sensor and sensor_json:
        sensor_payload = json.loads(sensor_json)
        if DEFAULT_SENSOR_TOKEN not in user_prompt:
            user_prompt = f"{DEFAULT_SENSOR_TOKEN}\n{user_prompt}"

    msg.add_message(user_prompt)
    processed = text_processor(msg.messages, mode="eval")
    input_ids = processed["input_ids"].unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            sensors=[sensor_payload] if sensor_payload is not None else None,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    return GenerationResult(
        tag=tag,
        prompt=user_prompt,
        attach_sensor=attach_sensor and sensor_payload is not None,
        output_text=generated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned-model", required=True, help="Path to finetuned TinyLLaVA checkpoint")
    parser.add_argument("--baseline-model", required=True, help="Path to baseline (non-finetuned) checkpoint")
    parser.add_argument("--prompt", default="<sensor>\n위 센서 기록을 소재로 짧은 서정시를 지어줘.", help="Evaluation prompt")
    parser.add_argument(
        "--sensor-json",
        default='{"temperature": 14.7, "humidity": 69.5, "wind_direction": 4, "imu": [-0.013, -0.362, 9.757, 0.014, 0.028, 0.058]}',
        help="Sensor payload as JSON string",
    )
    parser.add_argument("--image-path", default="/home/sihsch/tiny-LLaVA/dataset/llava/llava_pretrain/images/00414/004146340.jpg", help="Optional image to include in the prompt")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--conv-mode", default="qwen2_base", help="Conversation template")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--baseline-uses-sensor", action="store_true", help="Also pass sensor payload to the baseline model")
    parser.add_argument("--output", default=None, help="Optional path to dump comparison JSON")
    args = parser.parse_args()

    finetuned_result = run_single(
        model_path=args.finetuned_model,
        tag="finetuned",
        prompt=args.prompt,
        sensor_json=args.sensor_json,
        image_path=args.image_path,
        attach_sensor=True,
        device=args.device,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    baseline_result = run_single(
        model_path=args.baseline_model,
        tag="baseline",
        prompt=args.prompt,
        sensor_json=args.sensor_json,
        image_path=args.image_path,
        attach_sensor=args.baseline_uses_sensor,
        device=args.device,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Comparison ===")
    for result in (finetuned_result, baseline_result):
        header = f"[{result.tag}] sensor_attached={result.attach_sensor}"
        print(header)
        print("-" * len(header))
        print(result.output_text)
        print()

    if args.output:
        payload = {
            "prompt": args.prompt,
            "image_path": args.image_path,
            "sensor_json": json.loads(args.sensor_json),
            "results": [
                {
                    "tag": finetuned_result.tag,
                    "attach_sensor": finetuned_result.attach_sensor,
                    "output_text": finetuned_result.output_text,
                },
                {
                    "tag": baseline_result.tag,
                    "attach_sensor": baseline_result.attach_sensor,
                    "output_text": baseline_result.output_text,
                },
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
