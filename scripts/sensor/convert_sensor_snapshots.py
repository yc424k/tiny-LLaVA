#!/usr/bin/env python3
"""
Convert sensor snapshot files to LLaVA conversation format for ablation study.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_sensor_snapshots(input_dir: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load sensor snapshot files from directory."""
    snapshot_files = list(Path(input_dir).glob("snapshot_*.json"))
    
    if max_samples and len(snapshot_files) > max_samples:
        snapshot_files = snapshot_files[:max_samples]
    
    snapshots = []
    skipped = 0
    for file_path in snapshot_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Skip empty files
                    skipped += 1
                    continue
                snapshot = json.loads(content)
                snapshot['id'] = file_path.stem  # Use filename as ID
                snapshots.append(snapshot)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Skipping invalid file {file_path}: {e}")
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"Skipped {skipped} invalid/empty files")
    
    return snapshots


def create_conversation_format(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sensor snapshot to LLaVA conversation format."""
    sensor_data = snapshot.get('sensor_data', {})
    
    # Create sensor data text
    sensor_lines = []
    for key, value in sensor_data.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            # For IMU data, format as comma-separated values
            value_str = ", ".join(f"{v:.2f}" for v in value)
            sensor_lines.append(f"- {label}: [{value_str}]")
        else:
            sensor_lines.append(f"- {label}: {value}")
    
    sensor_text = "\n".join(sensor_lines)
    
    # Create prompt with <sensor> token
    prompt = f"<sensor>\nSensor data:\n{sensor_text}\n\nCraft a literary paragraph that embodies this moment."
    
    # Create target text (placeholder - you might want to generate or provide real targets)
    target = "A moment captured in time, where environmental conditions converge to create a unique atmospheric experience."
    
    # Build conversation format
    conversation = {
        "id": snapshot.get('id'),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "assistant", "value": target}
        ],
        "sensor_data": sensor_data
    }
    
    # Add image path if available
    if 'image_path' in snapshot:
        conversation['image_path'] = snapshot['image_path']
    
    return conversation


def main():
    parser = argparse.ArgumentParser(description="Convert sensor snapshots to LLaVA format")
    parser.add_argument("--input-dir", required=True, help="Directory containing sensor snapshot JSON files")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to convert")
    
    args = parser.parse_args()
    
    print(f"Loading sensor snapshots from {args.input_dir}...")
    snapshots = load_sensor_snapshots(args.input_dir, args.max_samples)
    print(f"Loaded {len(snapshots)} snapshots")
    
    print("Converting to LLaVA conversation format...")
    conversations = []
    for snapshot in snapshots:
        conv = create_conversation_format(snapshot)
        conversations.append(conv)
    
    # Save converted dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(conversations)} conversations to {args.output}")


if __name__ == "__main__":
    main()