#!/bin/bash

# ======================================================================
# To setup improve env vars, run this script first:
# source ./setup_improve.sh
# ======================================================================

DATA_DIR="/home/onarykov/repurposing-scripts"

# Params
MODEL_NAME=deepttc

SPLIT=0

EPOCHS=2
# EPOCHS=10
# EPOCHS=25
#EPOCHS=50
# EPOCHS=500

CUDA=1

# This script abs path
# script_dir="$(dirname "$0")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
echo "Script full path directory: $script_dir"

# Separate dirs
echo $script_dir
gout=${script_dir}/out.end_to_end_repurposing
ML_DATA_DIR=$gout/preprocess/split_${SPLIT}
MODEL_DIR=$gout/train/split_${SPLIT}
INFER_DIR=$gout/infer/split_${SPLIT}

# Preprocess
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${MODEL_NAME}_preprocess_improve.py \
    --train_split_file ${SPLIT}_train.txt \
    --val_split_file ${SPLIT}_val.txt \
    --test_split_file ${SPLIT}_test.txt \
    --input_dir ${DATA_DIR}/raw_data \
    --output_dir $ML_DATA_DIR

# Train
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${MODEL_NAME}_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR \
    --epochs $EPOCHS

# Infer
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${MODEL_NAME}_infer_improve.py \
    --input_data_dir $ML_DATA_DIR\
    --input_model_dir $MODEL_DIR\
    --output_dir $INFER_DIR \
    --calc_infer_score false
    # --calc_infer_score true
