#!/usr/bin/env python3
"""Compare two TinyLLaVA checkpoints on a shared prompt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.serve.cli import load_image
from tinyllava.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SENSOR_TOKEN,
    Message,
    disable_torch_init,
)
from tinyllava.data import ImagePreprocess, TextPreprocess


PRETRAIN_DEFAULT_PROMPT = "Craft a literary paragraph that embodies this moment."


@dataclass
class GenerationBundle:
    tag: str
    prompt: str
    output_text: str


def _format_sensor_text(sensor_dict: Optional[dict[str, Any]]) -> Optional[str]:
    if not sensor_dict:
        return None

    sensor_lines = []
    for key, value in sensor_dict.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value_str = json.dumps(value, ensure_ascii=False)
        else:
            value_str = str(value)
        sensor_lines.append(f"- {label}: {value_str}")

    if not sensor_lines:
        return None
    return "Sensor data:\n" + "\n".join(sensor_lines)


def _load_model(model_path: str, device: str):
    try:
        return load_pretrained_model(model_name_or_path=model_path, device=device)
    except OSError as exc:
        if "no file named" not in str(exc).lower():
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

    text_dtype = getattr(config.text_config, "torch_dtype", None)
    dtype = torch.float16 if text_dtype in (None, torch.float16, "float16") else torch.float32
    model.load_llm(
        model_name_or_path=config.llm_model_name_or_path,
        pretrained_llm_path=os.path.join(model_path, "language_model"),
        torch_dtype=dtype,
    )
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


def _generate(
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
) -> GenerationBundle:
    disable_torch_init()
    torch.set_grad_enabled(False)

    model, tokenizer, image_processor, _ = _load_model(model_path, device)
    model.eval()

    text_proc = TextPreprocess(tokenizer, conv_mode)
    image_proc = ImagePreprocess(image_processor, model.config)
    model.to(device)

    requires_sensor = bool(getattr(model.config, "sensor_encoder_type", None))
    if requires_sensor and not (attach_sensor and sensor_json):
        raise ValueError(
            f"Model '{tag}' expects sensor inputs; provide --sensor with sensor JSON or snapshot."
        )

    msg = Message()
    user_prompt = prompt

    image_tensor = None
    if image_path:
        pil_image = load_image(image_path)
        image_tensor = image_proc(pil_image)
        vision_module = getattr(model.vision_tower, "_vision_tower", None)
        dtype = None
        if vision_module is not None:
            try:
                dtype = next(vision_module.parameters()).dtype
            except StopIteration:
                dtype = None
        if dtype is None:
            try:
                dtype = next(model.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=dtype)
        if DEFAULT_IMAGE_TOKEN not in user_prompt:
            user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{user_prompt}"

    sensor_payload = None
    if attach_sensor and sensor_json:
        sensor_payload = json.loads(sensor_json)
        if DEFAULT_SENSOR_TOKEN not in user_prompt:
            user_prompt = f"{DEFAULT_SENSOR_TOKEN}\n{user_prompt}"

    msg.add_message(user_prompt)
    processed = text_proc(msg.messages, mode="eval")
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

    output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return GenerationBundle(tag=tag, prompt=user_prompt, output_text=output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="List of model paths")
    parser.add_argument("--tags", nargs="+", help="List of tags corresponding to models")
    parser.add_argument("--prompt", default=None, help="Shared prompt applied when specific prompts are not set")
    parser.add_argument("--prompts", nargs="+", default=None, help="Optional list of per-model prompt overrides")
    parser.add_argument("--image", default=None, help="Optional image path")
    parser.add_argument("--sensor-json", default=None, help="Optional sensor payload as JSON string")
    parser.add_argument("--sensor", action="store_true", help="Attach the sensor payload to both models")
    parser.add_argument("--snapshot", default=None, help="JSON snapshot containing image_path and sensor_data")
    parser.add_argument("--snapshot-dir", default=None, help="Directory containing JSON snapshots to process")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to store per-snapshot comparison outputs as JSON",
    )
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--conv-mode", default="qwen2_base", help="Conversation template key")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--output", default=None, help="Optional JSON file to store the comparison")
    args = parser.parse_args()

    if args.models is None or args.tags is None:
        raise SystemExit("--models and --tags must be provided")
    if len(args.models) != len(args.tags):
        raise SystemExit("Number of models must match number of tags")
    prompt_overrides: list[Optional[str]] = []
    if args.prompts is not None:
        if len(args.prompts) != len(args.models):
            raise SystemExit("Number of prompts must match number of models")
        prompt_overrides = list(args.prompts)
    else:
        prompt_overrides = [None] * len(args.models)

    def select_prompt(tag: str, specific: Optional[str], default_sensor_text: Optional[str]) -> str:
        if specific is not None:
            return specific
        if args.prompt is not None:
            return args.prompt
        if tag and "pretrain" in tag.lower():
            if default_sensor_text:
                return default_sensor_text + "\n\nCraft a literary paragraph that embodies this moment."
            return PRETRAIN_DEFAULT_PROMPT
        return default_sensor_text or ""

    snapshot_entries: list[Optional[str]] = []
    if args.snapshot:
        snapshot_entries.append(str(Path(args.snapshot)))

    directory_snapshots: list[str] = []
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        if not snapshot_dir.is_dir():
            raise SystemExit(f"--snapshot-dir must point to a directory: '{snapshot_dir}'")
        directory_snapshots = sorted(str(path) for path in snapshot_dir.glob("*.json"))
        if not directory_snapshots:
            print(f"[warn] No JSON snapshots found in directory '{snapshot_dir}'.", file=sys.stderr)
        snapshot_entries.extend(directory_snapshots)

    seen_snapshots: set[str] = set()
    ordered_snapshots: list[str] = []
    for entry in snapshot_entries:
        if entry is None:
            continue
        if entry not in seen_snapshots:
            ordered_snapshots.append(entry)
            seen_snapshots.add(entry)

    snapshot_queue: list[Optional[str]] = ordered_snapshots or [None]

    log_dir_path: Optional[Path] = None
    if args.log_dir:
        log_dir_path = Path(args.log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

    print("=== TinyLLaVA Comparison ===")

    all_runs: list[dict[str, Any]] = []

    def run_for_snapshot(snapshot_path: Optional[str]) -> dict[str, Any]:
        local_image = args.image
        local_sensor_json = args.sensor_json
        sensor_data_dict: Optional[dict[str, Any]] = None
        snapshot_payload: Optional[dict[str, Any]] = None

        if snapshot_path:
            with open(snapshot_path, "r", encoding="utf-8") as snapshot_file:
                snapshot_payload = json.load(snapshot_file)

            if local_image is None:
                local_image = snapshot_payload.get("image_path")

        if local_sensor_json is not None:
            try:
                sensor_data_dict = json.loads(local_sensor_json)
            except json.JSONDecodeError:
                sensor_data_dict = None
        elif snapshot_payload and snapshot_payload.get("sensor_data") is not None:
            sensor_data_dict = snapshot_payload["sensor_data"]
            local_sensor_json = json.dumps(sensor_data_dict, ensure_ascii=False)

        if sensor_data_dict is None and snapshot_payload:
            sensor_data_dict = snapshot_payload.get("sensor_data")

        attach_sensor_flag = args.sensor or bool(sensor_data_dict)
        sensor_text = _format_sensor_text(sensor_data_dict)

        if snapshot_path:
            print(f"--- Snapshot: {snapshot_path}")

        run_bundles: list[GenerationBundle] = []
        for idx, (model_path, tag) in enumerate(zip(args.models, args.tags)):
            prompt_override = prompt_overrides[idx]
            prompt_value = select_prompt(tag, prompt_override, sensor_text)

            model_attach_sensor = attach_sensor_flag and not (tag and "pretrain" in tag.lower())

            bundle = _generate(
                model_path=model_path,
                tag=tag,
                prompt=prompt_value,
                sensor_json=local_sensor_json,
                image_path=local_image,
                attach_sensor=model_attach_sensor,
                device=args.device,
                conv_mode=args.conv_mode,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            run_bundles.append(bundle)

            header = f"[{bundle.tag}]"
            print(header)
            print("-" * len(header))
            print(bundle.output_text)
            print()

        if sensor_text:
            print(sensor_text)

        payload: dict[str, Any] = {
            "snapshot": snapshot_path,
            "shared_prompt": args.prompt,
            "image": local_image,
            "sensor_json": sensor_data_dict,
            "sensor_data": sensor_data_dict,
            "results": [
                {"tag": item.tag, "prompt": item.prompt, "output": item.output_text}
                for item in run_bundles
            ],
        }

        if log_dir_path is not None:
            base_name = "comparison"
            if snapshot_path:
                base_name = Path(snapshot_path).stem
            log_path = log_dir_path / f"{base_name}_comparison.json"
            with open(log_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            payload["log_file"] = str(log_path)

        return payload

    for snapshot_path in snapshot_queue:
        run_payload = run_for_snapshot(snapshot_path)
        all_runs.append(run_payload)

    if args.output:
        if len(all_runs) == 1:
            output_payload = all_runs[0]
        else:
            output_payload = {"shared_prompt": args.prompt, "runs": all_runs}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
