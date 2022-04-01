#!/bin/bash

source ../env/bin/activate

pip uninstall transformers -y
git clone https://github.com/huggingface/transformers
cd transformers
pip install .

cd examples/pytorch/question-answering
pip install -r requirements.txt

python run_qa.py \
    --model_name_or_path /mnt/hdd/canine/models/mbert \
    --dataset_name xquad \
    --dataset_config_name xquad.th \
    --do_eval \
    --n_best_size 20 \
    --max_answer_length 30 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /mnt/hdd/canine/models/mbert/ \
    --per_device_eval_batch_size=6  \
