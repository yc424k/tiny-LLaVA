#!/bin/bash

# Sensor-only finetune entry point for OpenELM-based TinyLLaVA checkpoints.
# Defaults mirror finetune_openelm.sh but swap the dataset for sensor payloads
# and enable the fusion sensor encoder.

DEFAULT_SENSOR_DATA_PATH="/home/sihsch/tiny-LLaVA/dataset/sensor/sensor_llava_en.json"
DEFAULT_LLM_VERSION="apple/OpenELM-270M-Instruct"
DEFAULT_VT_VERSION="google/siglip-so400m-patch14-384"
DEFAULT_CN_VERSION="mlp2x_gelu"
DEFAULT_PRETRAINED_MODEL_PATH="/home/sihsch/tiny-LLaVA/checkpoints/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-pretrain"

if [ $# -eq 0 ]; then
    SENSOR_DATA_PATH="$DEFAULT_SENSOR_DATA_PATH"
    LLM_VERSION="$DEFAULT_LLM_VERSION"
    VT_VERSION="$DEFAULT_VT_VERSION"
    CN_VERSION="$DEFAULT_CN_VERSION"
    PRETRAINED_MODEL_PATH="$DEFAULT_PRETRAINED_MODEL_PATH"
elif [ $# -eq 5 ]; then
    SENSOR_DATA_PATH="$1"
    LLM_VERSION="$2"
    VT_VERSION="$3"
    CN_VERSION="$4"
    PRETRAINED_MODEL_PATH="$5"
else
    echo "Usage: $0 [<SENSOR_DATA_PATH> <LLM_VERSION> <VT_VERSION> <CONNECTOR_TYPE> <PRETRAINED_MODEL_PATH>]"
    exit 1
fi

VT_VERSION2=""
CONV_VERSION=llama
TRAIN_RECIPE=common
VERSION=elm_base
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}

SENSOR_ENCODER_TYPE=${SENSOR_ENCODER_TYPE:-fusion}
SENSOR_TOKEN_LENGTH=${SENSOR_TOKEN_LENGTH:-1}
SENSOR_FEATURE_DIM=${SENSOR_FEATURE_DIM:-256}
SENSOR_ATTENTION_HEADS=${SENSOR_ATTENTION_HEADS:-8}
SENSOR_FIELD=${SENSOR_FIELD:-sensor_data}

NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29507}
TRAIN_BATCH=${TRAIN_BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EVAL_BATCH=${EVAL_BATCH:-4}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-8}
REPORT_BACKEND=${REPORT_BACKEND:-tensorboard}

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

OUTPUT_DIR="/home/sihsch/tiny-LLaVA/checkpoints/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-sensor-finetune"

deepspeed --num_gpus $NUM_GPUS --master_port $MASTER_PORT tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path $SENSOR_DATA_PATH \
    --image_folder "" \
    --is_multimodal False \
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
    --fp16 True \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_type_sensor full \
    --tune_type_connector frozen \
    --group_by_modality_length False \
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $TRAIN_BATCH \
    --per_device_eval_batch_size $EVAL_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
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
    --dataloader_num_workers $DATALOADER_WORKERS \
    --lazy_preprocess True \
    --report_to $REPORT_BACKEND \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-sensor-finetune
