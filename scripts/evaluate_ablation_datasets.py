#!/usr/bin/env python3
"""
Evaluate ablation study datasets using a single finetuned model.
This script processes JSON files containing dataset samples and evaluates them.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def _load_model(model_path: str, device: str):
    """Load the TinyLLaVA model."""
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


def evaluate_dataset_sample(
    model, tokenizer, image_processor, sample: Dict[str, Any], 
    device: str, conv_mode: str, temperature: float, max_new_tokens: int
) -> Dict[str, Any]:
    """Evaluate a single sample from the dataset."""
    
    text_proc = TextPreprocess(tokenizer, conv_mode)
    image_proc = ImagePreprocess(image_processor, model.config)
    
    # Extract information from sample
    image_path = sample.get('image', '')
    conversations = sample.get('conversations', [])
    sensor_data = sample.get('sensor_data', {})
    
    if not conversations:
        return {"error": "No conversations found in sample"}
    
    # Get the human prompt
    human_prompt = None
    for conv in conversations:
        if conv.get('from') == 'human':
            human_prompt = conv.get('value', '')
            break
    
    if not human_prompt:
        return {"error": "No human prompt found in conversations"}
    
    # Load image if available
    image_tensor = None
    if image_path and os.path.exists(image_path):
        try:
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
            
            # Add image token if not present
            if DEFAULT_IMAGE_TOKEN not in human_prompt:
                human_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{human_prompt}"
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
    
    # Handle sensor data
    sensor_payload = None
    if sensor_data:
        sensor_payload = sensor_data
        if DEFAULT_SENSOR_TOKEN not in human_prompt:
            human_prompt = f"{DEFAULT_SENSOR_TOKEN}\n{human_prompt}"
    
    # Process the prompt
    msg = Message()
    msg.add_message(human_prompt)
    processed = text_proc(msg.messages, mode="eval")
    input_ids = processed["input_ids"].unsqueeze(0).to(model.device)
    
    # Generate response
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
    
    return {
        "prompt": human_prompt,
        "response": output_text,
        "image_path": image_path,
        "sensor_data": sensor_data,
        "expected_response": conversations[1].get('value', '') if len(conversations) > 1 else ""
    }


def evaluate_dataset(
    model_path: str, dataset_path: str, output_dir: str,
    device: str = "cuda", conv_mode: str = "phi", temperature: float = 0.0,
    max_new_tokens: int = 256, max_samples: int = None
):
    """Evaluate an entire dataset."""
    
    disable_torch_init()
    torch.set_grad_enabled(False)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, tokenizer, image_processor, _ = _load_model(model_path, device)
    model.eval()
    model.to(device)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
    
    print(f"Evaluating {len(dataset)} samples...")
    
    results = []
    for i, sample in enumerate(dataset):
        print(f"Processing sample {i+1}/{len(dataset)}...")
        
        try:
            result = evaluate_dataset_sample(
                model, tokenizer, image_processor, sample,
                device, conv_mode, temperature, max_new_tokens
            )
            result["sample_id"] = i
            results.append(result)
            
            # Print progress
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Generated: {result['response'][:100]}...")
                
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            results.append({"sample_id": i, "error": str(e)})
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = Path(dataset_path).stem
    output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": model_path,
            "dataset_path": dataset_path,
            "num_samples": len(dataset),
            "device": device,
            "conv_mode": conv_mode,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation study datasets")
    parser.add_argument("--model-path", required=True, help="Path to the finetuned model")
    parser.add_argument("--dataset", required=True, help="Path to the dataset JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")
    parser.add_argument("--conv-mode", default="phi", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    
    args = parser.parse_args()
    
    evaluate_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()