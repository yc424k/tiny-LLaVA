# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `tinyllava/`: `model/` defines multimodal backbones, `train/` orchestrates HuggingFace-style training, `data/` provides dataset builders, `eval/` wraps evaluation pipelines, and `serve/` exposes CLI/web demos. Shell entry points reside in `scripts/` (`scripts/train/train_phi.sh`, `scripts/eval/vqav2.sh`) and expect you to override paths before running. Reference materials are in `assets/`, and sample configs plus instructions sit in `dataset/`; `tinyllava_visualizer/` contains the visualization tool shipped with this repo.

## Build, Test, and Development Commands
Create an environment aligned with the project baseline: `conda create -n tinyllava_factory python=3.10` followed by `conda activate tinyllava_factory`, then `pip install -e .` and `pip install flash-attn==2.5.7 --no-build-isolation`. Launch end-to-end training with `bash scripts/train/train_phi.sh` (update data, `output_dir`, and deepspeed config as needed) or swap in the other model-specific scripts. Use `bash scripts/eval/vqav2.sh` or its siblings to score checkpoints; set `CUDA_VISIBLE_DEVICES` before invoking to control GPU splits. For local inference tests, run `python -m tinyllava.serve.cli --model-path <checkpoint> --image <image_path>` to sanity-check outputs.

## Coding Style & Naming Conventions
Python sources target 3.10, follow four-space indentation, and prefer PEP 8 naming: snake_case for functions/files, PascalCase for classes (see `tinyllava/train/train.py`). Keep imports explicit when adding code; utilities already in `tinyllava.utils` should be pulled with targeted imports rather than new wildcards. Document public entry points with concise docstrings describing required arguments (`ModelArguments`, `TrainingArguments`, etc.) and keep configuration defaults close to where they are consumed.

## Testing Guidelines
Repository testing centers on dataset-level evaluation. Duplicate the provided shell scripts under `scripts/eval/`, customize `MODEL_PATH`, `MODEL_NAME`, and dataset roots, and execute `bash scripts/eval/<task>.sh` to produce merged predictions. For quick iteration, reduce `CHUNKS` or point `--answers-file` to a temporary directory to avoid polluting shared storage. Capture metric summaries in the README-friendly tables or wandb runs when introducing new recipes, and note any dataset filtering so others can reproduce.

## Commit & Pull Request Guidelines
Recent history favors short imperative commits (`Update README.md`, `Update train_phi.sh`); continue that style while referencing the modified module (`Refine train_phi data paths`). Each PR should explain the motivation, list key changes, and include run artifacts: command snippets, dataset pointers, evaluation numbers, and (for UI changes) screenshots. Link related GitHub issues or discussion threads, and flag any dependency changes so maintainers can refresh the documented setup.
