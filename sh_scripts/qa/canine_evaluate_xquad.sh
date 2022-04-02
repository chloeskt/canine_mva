#!/bin/bash

source ../env/bin/activate

rm -rf canine_mva
git clone https://"${GITHUB_TOKEN}"@github.com/chloeskt/canine_mva.git
cd canine_mva

LANGUAGES=('xquad.en' 'xquad.ar' 'xquad.de' 'xquad.zh' 'xquad.vi' 'xquad.es' 'xquad.hi' 'xquad.el' 'xquad.th' 'xquad.tr' 'xquad.ru' 'xquad.ro')

for lang in "${LANGUAGES[@]}"; do \
    python source/qa/canine_evaluate_xquad.py \
      --model_path /mnt/hdd/canine/models/canine_model.pt \
      --language "$lang" \
      --max_answer_length 256 \
      --max_length 2048 \
      --doc_stride 512 \
      --n_best_size 20 \
      --batch_size 6 \
      --device cuda ;
done

rm -rf canine_mva
