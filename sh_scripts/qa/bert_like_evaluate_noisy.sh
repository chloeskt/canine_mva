#!/bin/bash
set -eux

INPUT_JSON="/mnt/hdd/canine/noisy_40.json"
MODEL_PATH="/mnt/hdd/canine/models/mbert/"

source ../../env/bin/activate

pip uninstall transformers -y
rm -rf transformers
git clone https://github.com/huggingface/transformers
cd transformers
pip install .

cd examples/pytorch/question-answering
pip install -r requirements.txt
cp -r "$INPUT_JSON" /home/kaliayev/Documents/MVA/canine_mva/sh_scripts/qa/transformers/examples/pytorch/question-answering

python run_qa.py \
        --model_name_or_path "$MODEL_PATH"\
        --validation_file "./noisy_40.json" \
        --do_eval \
        --version_2_with_negative \
        --n_best_size 20 \
        --max_answer_length 30 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir "$MODEL_PATH" \
        --per_device_eval_batch_size=8

rm -rf transformers
