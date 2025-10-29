#!/usr/bin/env python3
"""Helper to launch the TinyLLaVA CLI from a sensor snapshot JSON."""
import argparse
import json
import subprocess
from pathlib import Path

from tinyllava.utils.constants import DEFAULT_SENSOR_TOKEN


def load_snapshot(snapshot_path: Path) -> dict:
    with snapshot_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prompt(template: str, sensor_payload: dict | None, inject_sensor: bool) -> str:
    prompt = template
    if sensor_payload is None:
        if "{sensor}" in template:
            raise ValueError("Prompt template contains {sensor} but snapshot has no sensor data.")
        return prompt

    sensor_text = json.dumps(sensor_payload, ensure_ascii=True)

    if "{sensor}" in template:
        prompt = template.format(sensor=sensor_text)
        if inject_sensor and DEFAULT_SENSOR_TOKEN not in prompt:
            prompt = f"{DEFAULT_SENSOR_TOKEN}\n{prompt}"
        return prompt

    if inject_sensor and DEFAULT_SENSOR_TOKEN not in prompt:
        prompt = f"{DEFAULT_SENSOR_TOKEN}\n{sensor_text}\n{prompt}"
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TinyLLaVA CLI with inputs taken from a snapshot JSON file.")
    parser.add_argument("snapshot", type=Path, help="Path to the sensor snapshot JSON file.")
    default_model = Path(__file__).resolve().parents[2] / "checkpoints" / "llava_factory" / "qwen3.0-finetune"
    parser.add_argument(
        "--model-path",
        default=str(default_model),
        help="Checkpoint to load.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device argument passed to the CLI.")
    parser.add_argument("--prompt-template", default="explain photo", help="Prompt text; {sensor} placeholder is replaced with sensor JSON.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional override for generation temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional override for max new tokens.")
    parser.add_argument("--load-8bit", action="store_true", help="Forward the load-8bit flag.")
    parser.add_argument("--load-4bit", action="store_true", help="Forward the load-4bit flag.")
    parser.add_argument("--no-inject-sensor", dest="inject_sensor", action="store_false", help="Disable prompt sensor injection.")
    parser.set_defaults(inject_sensor=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the command instead of executing it.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Additional arguments forwarded verbatim.")

    args = parser.parse_args()

    snapshot_path = args.snapshot.expanduser().resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    snapshot_payload = load_snapshot(snapshot_path)

    repo_root = Path(__file__).resolve().parents[2]

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()
    else:
        model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    image_path = snapshot_payload.get("image_path")
    if not image_path:
        raise KeyError("Snapshot JSON must provide an image_path field.")

    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    sensor_payload = snapshot_payload.get("sensor_data")
    prompt = build_prompt(args.prompt_template, sensor_payload, args.inject_sensor)

    command = [
        "python",
        "-m",
        "tinyllava.serve.cli",
        "--model-path",
        str(model_path),
        "--image-file",
        str(image_path),
        "--prompt",
        prompt,
        "--device",
        args.device,
    ]

    if sensor_payload is not None:
        command.extend(["--sensor", json.dumps(sensor_payload, ensure_ascii=True)])

    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])

    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])

    if args.load_8bit:
        command.append("--load-8bit")

    if args.load_4bit:
        command.append("--load-4bit")

    if args.extra:
        command.extend(args.extra)

    if args.dry_run:
        printable = " ".join(command)
        print(printable)
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
