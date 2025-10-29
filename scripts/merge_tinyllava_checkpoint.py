#!/usr/bin/env python
"""Merge TinyLLaVA Factory checkpoints saved per-module into a single HF-style save."""

import argparse
import os
import shutil
import sys
from typing import Optional

import torch

from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration


def _load_state_dict(path: str, description: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[warn] Skip missing {description}: {path}")
        return None
    print(f"[info] Loading {description} from {path}")
    return torch.load(path, map_location="cpu")


def merge_checkpoint(src_dir: str, dst_dir: str, overwrite: bool = False) -> None:
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        raise ValueError("Destination directory must be different from source directory.")

    if os.path.exists(dst_dir):
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(f"Destination directory already exists: {dst_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    print(f"[info] Loading config from {src_dir}")
    config = TinyLlavaConfig.from_pretrained(src_dir)
    model = TinyLlavaForConditionalGeneration(config)

    lm_state = _load_state_dict(os.path.join(src_dir, "language_model", "pytorch_model.bin"), "language model")
    if lm_state is not None:
        load_info = model.language_model.load_state_dict(lm_state, strict=False)
        missing_keys = getattr(load_info, "missing_keys", [])
        unexpected_keys = getattr(load_info, "unexpected_keys", [])
        if missing_keys:
            print(f"[warn] Missing keys when loading language model: {missing_keys}")
        if unexpected_keys:
            print(f"[warn] Unexpected keys when loading language model: {unexpected_keys}")
        # Some Qwen-style checkpoints omit lm_head in the saved state dict because it is tied
        # to the token embeddings. Ensure the tie happens after partially loading weights.
        tie_fn = getattr(model.language_model, "tie_weights", None)
        if callable(tie_fn):
            tie_fn()

    vision_state = _load_state_dict(os.path.join(src_dir, "vision_tower", "pytorch_model.bin"), "vision tower")
    if vision_state is not None:
        model.vision_tower._vision_tower.load_state_dict(vision_state)

    connector_state = _load_state_dict(os.path.join(src_dir, "connector", "pytorch_model.bin"), "connector")
    if connector_state is not None:
        model.connector.load_state_dict(connector_state, strict=False)

    sensor_state = _load_state_dict(os.path.join(src_dir, "sensor_encoder", "pytorch_model.bin"), "sensor encoder")
    if sensor_state is not None and model.sensor_encoder is not None:
        model.sensor_encoder.load_state_dict(sensor_state, strict=False)
    elif sensor_state is not None:
        print("[warn] Sensor weights found but config has no sensor encoder; skipping")

    print(f"[info] Saving merged checkpoint to {dst_dir}")
    model.save_pretrained(dst_dir)
    model.tokenizer.save_pretrained(dst_dir)
    config.save_pretrained(dst_dir)

    for extra in ["added_tokens.json", "special_tokens_map.json", "generation_config.json"]:
        src_path = os.path.join(src_dir, extra)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst_dir, extra))

    print("[info] Merge complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Path to TinyLLaVA Factory checkpoint with module subfolders")
    parser.add_argument("dst", help="Output directory for merged checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        merge_checkpoint(args.src, args.dst, overwrite=args.overwrite)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
