#!/bin/bash

# Batch comparison script for sensor snapshots
cd /home/yc424k/tiny-LLaVA

python scripts/batch_compare_outputs.py \
    --models checkpoints/llava_factory/tinyllama1.1-finetune checkpoints/llava_factory/tinyllama1.1-pretrained \
    --tags tinyllama_finetune tinyllama_pretrain \
    --snapshots-dir /home/yc424k/sensing/captured_data/sensor_snapshots \
    --output-dir ./batch_comparison_results \
    --max-new-tokens 256 \
    --device cuda