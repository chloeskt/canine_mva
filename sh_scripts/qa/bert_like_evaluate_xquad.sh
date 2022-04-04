#!/bin/bash
set -eux

MODEL_PATH="/mnt/hdd/canine/models/mbert/mbert_model.pt"
OUTPUT_DIR="/mnt/hdd/canine/models/mbert/"

source ../env/bin/activate

pip uninstall transformers -y
git clone https://github.com/huggingface/transformers
cd transformers
pip install .

cd examples/pytorch/question-answering
pip install -r requirements.txt

LANGUAGES=('xquad.en' 'xquad.ar' 'xquad.de' 'xquad.zh' 'xquad.vi' 'xquad.es' 'xquad.hi' 'xquad.el' 'xquad.th' 'xquad.tr' 'xquad.ru' 'xquad.ro')

for lang in "${LANGUAGES[@]}"; do \
  python run_qa.py \
      --model_name_or_path "$MODEL_PATH" \
      --dataset_name xquad \
      --dataset_config_name xquad."$lang" \
      --do_eval \
      --n_best_size 20 \
      --max_answer_length 30 \
      --max_seq_length 384 \
      --doc_stride 128 \
      --output_dir "$OUTPUT_DIR" \
      --per_device_eval_batch_size=6 ;
done

rm -rf transformers
