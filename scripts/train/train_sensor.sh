#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <SENSOR_DATA_PATH> <LLM_VERSION> <VT_VERSION> <CONNECTOR_TYPE> <PRETRAINED_MODEL_PATH>"
    exit 1
fi

SENSOR_DATA_PATH="$1"
LLM_VERSION="$2"
VT_VERSION="$3"
CN_VERSION="$4"
PRETRAINED_MODEL_PATH="$5"

VT_VERSION2=""
SENSOR_ENCODER_TYPE="fusion"
SENSOR_TOKEN_LENGTH=1
SENSOR_FIELD="sensor_data"
SENSOR_FEATURE_DIM=256
SENSOR_ATTENTION_HEADS=8
CONV_VERSION=phi
VERSION=sensor
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
OUTPUT_ROOT=/mnt/data/sata/yinghu/checkpoints/llava_factory

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

OUTPUT_DIR="$OUTPUT_ROOT/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-sensor"

deepspeed --include localhost:4,5,6,7 --master_port 29511 tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path $SENSOR_DATA_PATH \
    --image_folder "" \
    --is_multimodal True \
    --use_dummy_image True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --sensor_encoder_type $SENSOR_ENCODER_TYPE \
    --sensor_token_length $SENSOR_TOKEN_LENGTH \
    --sensor_feature_dim $SENSOR_FEATURE_DIM \
    --sensor_attention_heads $SENSOR_ATTENTION_HEADS \
    --sensor_field $SENSOR_FIELD \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_type_sensor full \
    --tune_type_connector frozen \
    --group_by_modality_length False \
    --is_multimodal False \
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-sensor
