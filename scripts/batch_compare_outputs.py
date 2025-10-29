#!/usr/bin/env python3
"""Batch process all sensor snapshots and compare model outputs."""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path

from compare_model_outputs import _generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="List of model paths")
    parser.add_argument("--tags", nargs="+", required=True, help="List of tags corresponding to models")
    parser.add_argument("--snapshots-dir", default="/home/yc424k/sensing/captured_data/sensor_snapshots", 
                       help="Directory containing snapshot JSON files")
    parser.add_argument("--output-dir", default="./batch_outputs", 
                       help="Directory to save output JSON files")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--conv-mode", default="qwen2_base", help="Conversation template key")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--prompt", default=None, help="Shared prompt applied when specific prompts are not set")
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.tags):
        raise SystemExit("Number of models must match number of tags")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all snapshot files
    snapshot_files = glob.glob(os.path.join(args.snapshots_dir, "*.json"))
    snapshot_files.sort()
    
    if not snapshot_files:
        raise SystemExit(f"No JSON files found in {args.snapshots_dir}")
    
    print(f"Found {len(snapshot_files)} snapshot files")
    print(f"Processing with models: {args.tags}")
    
    for snapshot_file in snapshot_files:
        print(f"\nProcessing: {os.path.basename(snapshot_file)}")
        
        # Load snapshot
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  Skipping {os.path.basename(snapshot_file)}: {str(e)}")
            continue
        
        # Extract data from snapshot
        image_path = snapshot_data.get("image_path")
        sensor_data = snapshot_data.get("sensor_data")
        sensor_json = json.dumps(sensor_data, ensure_ascii=False) if sensor_data else None
        
        # Create sensor text for prompting
        sensor_text = None
        if sensor_data:
            sensor_lines = []
            for key, value in sensor_data.items():
                label = key.replace('_', ' ').title()
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                sensor_lines.append(f"- {label}: {value_str}")
            sensor_text = "Sensor data:\n" + "\n".join(sensor_lines)
        
        def select_prompt(tag: str, default_sensor_text: str) -> str:
            if args.prompt is not None:
                return args.prompt
            if tag and "pretrain" in tag.lower():
                if default_sensor_text:
                    return default_sensor_text + "\n\nCraft a literary paragraph that embodies this moment."
                return "Craft a literary paragraph that embodies this moment."
            return default_sensor_text or ""
        
        # Generate outputs for each model
        results = []
        for model_path, tag in zip(args.models, args.tags):
            prompt_value = select_prompt(tag, sensor_text)
            
            # Determine if sensor should be attached
            attach_sensor = sensor_data is not None and not ("pretrain" in tag.lower())
            
            try:
                bundle = _generate(
                    model_path=model_path,
                    tag=tag,
                    prompt=prompt_value,
                    sensor_json=sensor_json,
                    image_path=image_path,
                    attach_sensor=attach_sensor,
                    device=args.device,
                    conv_mode=args.conv_mode,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                
                results.append({
                    "tag": bundle.tag,
                    "prompt": bundle.prompt,
                    "output": bundle.output_text
                })
                
                print(f"  {tag}: Generated {len(bundle.output_text)} characters")
                
            except Exception as e:
                print(f"  {tag}: Error - {str(e)}")
                results.append({
                    "tag": tag,
                    "prompt": prompt_value,
                    "output": f"ERROR: {str(e)}"
                })
        
        # Create output filename based on snapshot filename
        snapshot_basename = os.path.splitext(os.path.basename(snapshot_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{snapshot_basename}_comparison_{timestamp}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save results
        output_data = {
            "timestamp": timestamp,
            "snapshot_file": snapshot_file,
            "image_path": image_path,
            "sensor_data": sensor_data,
            "models": args.models,
            "tags": args.tags,
            "parameters": {
                "device": args.device,
                "conv_mode": args.conv_mode,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "custom_prompt": args.prompt
            },
            "results": results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved results to: {output_filename}")
    
    print(f"\nBatch processing complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()