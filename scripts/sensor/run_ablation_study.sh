#!/bin/bash
# Automated Sensor Ablation Study Runner
#
# This script runs a complete ablation study by:
# 1. Creating datasets with individual sensors removed
# 2. Running evaluations on each ablated dataset
# 3. Comparing results across all variants

set -e

# Configuration
SENSOR_SNAPSHOTS_DIR="/home/yc424k/sensing/captured_data/sensor_snapshots"
ORIGINAL_DATASET="dataset/sensor/sensor_llava_converted.json"
MODEL_PATH="checkpoints/llava_factory/tinyllama1.1-finetune"
OUTPUT_DIR="ablation_results/$(date +%Y%m%d_%H%M%S)"
SENSORS=("temperature" "humidity" "wind_direction" "imu")

# Convert sensor snapshots to LLaVA format if needed
if [ ! -f "$ORIGINAL_DATASET" ]; then
    echo "Converting sensor snapshots to LLaVA format..."
    mkdir -p dataset/sensor
    
    python scripts/sensor/convert_sensor_snapshots.py \
        --input-dir "$SENSOR_SNAPSHOTS_DIR" \
        --output "$ORIGINAL_DATASET" \
        --max-samples 1909
    
    echo "Conversion completed!"
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

echo "Starting Sensor Ablation Study"
echo "=============================="
echo "Dataset: $ORIGINAL_DATASET"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Sensors: ${SENSORS[*]}"
echo ""

# Create ablated datasets
echo "Step 1: Creating ablated datasets..."
python scripts/sensor/ablation_study.py \
    --input-dataset "$ORIGINAL_DATASET" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --sensors "${SENSORS[@]}" \
    --max-samples 1000

echo ""
echo "Step 2: Running evaluations..."
echo "Note: Evaluation step requires manual implementation based on your specific evaluation pipeline"
echo "The ablated datasets are ready in: $OUTPUT_DIR"

# Optional: Run a simple comparison script
if [ -f "scripts/sensor/compare_ablation_results.py" ]; then
    echo "Step 3: Comparing results..."
    python scripts/sensor/compare_ablation_results.py \
        --results-dir "$OUTPUT_DIR"
fi

echo ""
echo "Ablation study setup complete!"
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "Manual next steps:"
echo "1. Review the generated datasets in $OUTPUT_DIR"
echo "2. Run your evaluation script on each dataset:"
echo "   - Baseline: $OUTPUT_DIR/ablation_baseline.json"
for sensor in "${SENSORS[@]}"; do
    echo "   - No $sensor: $OUTPUT_DIR/ablation_no_${sensor}.json"
done
echo "3. Compare the results to determine sensor importance"