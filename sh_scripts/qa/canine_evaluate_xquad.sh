#!/bin/bash
set -eux

MODEL_PATH="/mnt/hdd/canine/models/canine_model.pt"
HG_MODEL_CHECKPOINT="google/canine-c"

cd ../../
source env/bin/activate

LANGUAGES=('xquad.en' 'xquad.ar' 'xquad.de' 'xquad.zh' 'xquad.vi' 'xquad.es' 'xquad.hi' 'xquad.el' 'xquad.th' 'xquad.tr' 'xquad.ru' 'xquad.ro')

for LANG in "${LANGUAGES[@]}"; do \
    python source/qa/canine_evaluation.py \
      --model_path "$MODEL_PATH" \
      --dataset_name xquad \
      --data_dir None \
      --huggingface_model_checkpoint "$HG_MODEL_CHECKPOINT" \
      --language "$LANG" \
      --squad_v2 False \
      --max_answer_length 256 \
      --max_length 2048 \
      --doc_stride 512 \
      --n_best_size 20 \
      --batch_size 8 \
      --device cuda ;
done
