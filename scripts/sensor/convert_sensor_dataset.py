#!/usr/bin/env python
"""Convert raw sensor dataset into TinyLLaVA conversation format.

Example:
    python scripts/sensor/convert_sensor_dataset.py \
        --input dataset/sensor/sensor_datasets.json \
        --output dataset/sensor/sensor_llava.json

The script wraps each sample with a two-turn dialogue. The human turn always
includes the <sensor> token so that TinyLLaVA can insert sensor embeddings
during training. An optional prompt template can be supplied to customise the
instruction text.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_PROMPT = (
    "<sensor>\n"
    "Temperature {temperature_celsius}°C, humidity {humidity_percent}%, scenario {context_scenario}. "
    "Craft a literary paragraph that embodies this moment."
)


class DefaultDict(dict):
    def __missing__(self, key):
        return ""


def extract_prompt_variables(sample: Dict[str, Any]) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    sensor = sample.get("sensor_data", {}) or {}

    def _store(prefix: str, obj: Dict[str, Any]):
        for key, value in obj.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                _store(new_key, value)
            elif isinstance(value, (list, tuple)):
                variables[new_key] = ", ".join(str(v) for v in value)
            else:
                variables[new_key] = str(value)

    _store("", sensor)

    metadata = sample.get("metadata", {}) or {}
    environmental = metadata.get("environmental_analysis", {}) or {}
    _store("env", environmental)
    novel_meta = metadata.get("novel_metadata", {}) or {}
    _store("novel", novel_meta)
    context = sensor.get("context", {}) or {}
    if context:
        _store("context", context)

    # convenience aliases
    if "temperature" in sensor:
        variables.setdefault("temperature_celsius", str(sensor["temperature"]))
    if "humidity" in sensor:
        variables.setdefault("humidity_percent", str(sensor["humidity"]))
    if "wind_direction" in sensor:
        variables.setdefault("wind_direction_degrees", str(sensor["wind_direction"]))

    return variables


def render_prompt(sample: Dict[str, Any], template: str) -> str:
    variables = extract_prompt_variables(sample)
    populated = template.format_map(DefaultDict(variables))
    return populated


def build_conversations(prompt: str, target_text: str):
    return [
        {"from": "human", "value": prompt},
        {"from": "assistant", "value": target_text.strip()},
    ]


def transform_sample(sample: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    target = sample.get("target_paragraph", "").strip()
    if not target:
        return None

    prompt = render_prompt(sample, prompt_template)
    if "<sensor>" not in prompt:
        raise ValueError("Rendered prompt must include <sensor> token. Adjust template.")

    conversations = build_conversations(prompt, target)

    converted = {
        "id": sample.get("id"),
        "conversations": conversations,
        "sensor_data": sample.get("sensor_data", {}),
    }

    if "metadata" in sample:
        converted["metadata"] = sample["metadata"]

    # Preserve other top-level keys (e.g., splits) if they exist
    for key, value in sample.items():
        if key in converted or key in {"target_paragraph"}:
            continue
        converted.setdefault(key, value)

    return converted


def main():
    parser = argparse.ArgumentParser(description="Sensor dataset converter")
    parser.add_argument("--input", required=True, help="Path to raw sensor JSON file")
    parser.add_argument("--output", required=True, help="Path to save TinyLLaVA-ready JSON")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
        help="Instruction text for the human turn (must contain <sensor> token)",
    )
    args = parser.parse_args()

    prompt_template = args.prompt_template
    if "<sensor>" not in prompt_template:
        raise ValueError("Prompt template must contain the <sensor> token.")

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    skipped = 0
    for sample in data:
        try:
            converted_sample = transform_sample(sample, prompt_template)
        except ValueError as err:
            raise ValueError(f"Error processing sample {sample.get('id')}: {err}") from err
        if converted_sample is None:
            skipped += 1
            continue
        converted.append(converted_sample)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} samples (skipped {skipped}) -> {output_path}")


if __name__ == "__main__":
    main()
