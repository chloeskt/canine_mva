#!/bin/bash
set -eux

MODEL_NAME="google/canine-c"
OUTPUT_DIR="/mnt/hdd/canine/models"

cd ../../
source env/bin/activate

python source/qa/main.py \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --doc_stride 512 \
  --max_length 2048 \
  --max_answer_length 256 \
  --n_best_size 20 \
  --batch_size 6 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --nb_epochs 5 \
  --warmup_proportion 0.1 \
  --best_f1 68.\
  --squad_v2 True \
  --freeze False \
  --lr_scheduler True \
  --drive False \
  --clipping False \
