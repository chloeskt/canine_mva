#!/bin/bash
set -eux

OUTPUT_DIR="/mnt/hdd/canine/models/mbert/"

source ../../env/bin/activate

pip uninstall transformers -y
git clone https://github.com/huggingface/transformers
cd transformers
pip install .

cd examples/pytorch/question-answering
pip install -r requirements.txt

python run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --per_device_eval_batch_size=6 \
  --per_device_train_batch_size=6 \
  --save_steps 5000

rm -rf transformers
