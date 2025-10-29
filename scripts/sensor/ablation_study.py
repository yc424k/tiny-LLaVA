#!/usr/bin/env python3
"""
Sensor Ablation Study Script

This script performs ablation studies by systematically removing individual sensors
and comparing model performance. It creates modified datasets where specific sensor
data fields are excluded, then runs evaluation on each variant.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_sensor_dataset(input_path: str) -> List[Dict[str, Any]]:
    """Load the original sensor dataset."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_ablated_dataset(data: List[Dict[str, Any]], removed_sensor: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Create a dataset with a specific sensor removed."""
    # Limit samples if specified
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    
    ablated_data = []
    
    for sample in data:
        ablated_sample = sample.copy()
        
        # Remove the sensor from sensor_data
        if 'sensor_data' in ablated_sample and removed_sensor in ablated_sample['sensor_data']:
            ablated_sample['sensor_data'] = ablated_sample['sensor_data'].copy()
            del ablated_sample['sensor_data'][removed_sensor]
        
        # Update the prompt to exclude the removed sensor
        if 'conversations' in ablated_sample:
            conversations = []
            for conv in ablated_sample['conversations']:
                if conv['from'] == 'human' and 'value' in conv:
                    # Remove sensor from the prompt text
                    prompt = conv['value']
                    sensor_data = ablated_sample.get('sensor_data', {})
                    
                    # Reconstruct the sensor data part of the prompt
                    if sensor_data:
                        sensor_lines = []
                        for key, value in sensor_data.items():
                            label = key.replace("_", " ").title()
                            sensor_lines.append(f"- {label}: {value}")
                        
                        # Replace sensor data in prompt
                        if "Sensor data:" in prompt:
                            prompt_parts = prompt.split("Sensor data:")
                            if len(prompt_parts) >= 2:
                                before_sensor = prompt_parts[0]
                                after_sensor_parts = prompt_parts[1].split("\n\n")
                                after_sensor = "\n\n".join(after_sensor_parts[1:]) if len(after_sensor_parts) > 1 else ""
                                
                                new_sensor_data = "\n".join(sensor_lines)
                                prompt = f"{before_sensor}Sensor data:\n{new_sensor_data}\n\n{after_sensor}".strip()
                    
                    conv = conv.copy()
                    conv['value'] = prompt
                
                conversations.append(conv)
            ablated_sample['conversations'] = conversations
        
        ablated_data.append(ablated_sample)
    
    return ablated_data


def save_ablated_dataset(data: List[Dict[str, Any]], output_path: str):
    """Save the ablated dataset to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_evaluation(model_path: str, dataset_path: str, output_dir: str, removed_sensor: str = None):
    """Run model evaluation on the ablated dataset."""
    eval_script = "scripts/compare_model_outputs.py"
    
    # Create output directory for this ablation
    if removed_sensor:
        ablation_output_dir = os.path.join(output_dir, f"ablation_no_{removed_sensor}")
    else:
        ablation_output_dir = os.path.join(output_dir, "ablation_baseline")
    
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    # Run evaluation (you may need to adjust these parameters based on your setup)
    cmd = [
        "python", eval_script,
        "--model-path", model_path,
        "--dataset", dataset_path,
        "--output-dir", ablation_output_dir,
        "--device", "cuda",
        "--temperature", "0.0"
    ]
    
    print(f"Running evaluation for {'baseline' if not removed_sensor else f'no_{removed_sensor}'}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Note: This is a placeholder - you'll need to adjust based on your actual evaluation script
    # result = subprocess.run(cmd, capture_output=True, text=True)
    # return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Perform sensor ablation study")
    parser.add_argument("--input-dataset", required=True, 
                       help="Path to original sensor dataset JSON file")
    parser.add_argument("--model-path", required=True,
                       help="Path to the trained model for evaluation")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save ablated datasets and results")
    parser.add_argument("--sensors", nargs='+', 
                       default=["temperature", "humidity", "wind_direction"],
                       help="List of sensors to ablate")
    parser.add_argument("--run-evaluation", action="store_true",
                       help="Also run model evaluation on ablated datasets")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to use from the dataset")
    
    args = parser.parse_args()
    
    # Load original dataset
    print(f"Loading dataset from {args.input_dataset}...")
    original_data = load_sensor_dataset(args.input_dataset)
    print(f"Loaded {len(original_data)} samples")
    
    # Limit samples if specified
    if args.max_samples and len(original_data) > args.max_samples:
        original_data = original_data[:args.max_samples]
        print(f"Limited to {len(original_data)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save baseline (full sensor) dataset
    baseline_path = os.path.join(args.output_dir, "ablation_baseline.json")
    save_ablated_dataset(original_data, baseline_path)
    print(f"Saved baseline dataset to {baseline_path}")
    
    # Create ablated datasets
    ablated_datasets = {}
    
    for sensor in args.sensors:
        print(f"\nCreating ablated dataset without '{sensor}'...")
        ablated_data = create_ablated_dataset(original_data, sensor, args.max_samples)
        
        output_path = os.path.join(args.output_dir, f"ablation_no_{sensor}.json")
        save_ablated_dataset(ablated_data, output_path)
        ablated_datasets[sensor] = output_path
        
        print(f"Saved ablated dataset to {output_path}")
        
        # Verify the ablation worked
        sample_sensor_data = ablated_data[0].get('sensor_data', {})
        if sensor in sample_sensor_data:
            print(f"WARNING: {sensor} was not properly removed from sensor_data!")
        else:
            print(f"✓ Successfully removed {sensor} from sensor_data")
    
    # Run evaluations if requested
    if args.run_evaluation:
        print("\n" + "="*50)
        print("Running evaluations...")
        
        # Baseline evaluation
        run_evaluation(args.model_path, baseline_path, args.output_dir)
        
        # Ablated evaluations
        for sensor, dataset_path in ablated_datasets.items():
            run_evaluation(args.model_path, dataset_path, args.output_dir, sensor)
    
    print("\n" + "="*50)
    print("Ablation study setup complete!")
    print(f"Results saved in: {args.output_dir}")
    print("\nGenerated datasets:")
    print(f"- Baseline: {baseline_path}")
    for sensor, path in ablated_datasets.items():
        print(f"- No {sensor}: {path}")
    
    if not args.run_evaluation:
        print("\nTo run evaluations, use --run-evaluation flag or run them manually:")
        print("python scripts/compare_model_outputs.py --help")


if __name__ == "__main__":
    main()