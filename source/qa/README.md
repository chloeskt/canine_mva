# Question Answering 

## Organization

This subfolder contains the whole code associated with the Question Answering experiments. It can developed by Chloé Sekkat
and it can be viewed as a Python package whose main functions/classes can be found in the ``__init__.py``.

## Description

### Experiments

In this section, we are interested by the capacities of CANINE versus BERT-like models such as BERT, mBERT and XLM-RoBERTa on
Question Answering tasks. CANINE is a pre-trained tokenization-free and vocabulary-free encoder, that operates directly on character sequences without explicit
tokenization. It seeks to generalize beyond the orthographic forms encountered during pre-training.

We evaluate its capacities on extractive question answering (select minimal span answer within a context) on SQuAD dataset. The latter is
a unilingual (English) dataset available in Hugging Face (simple as ``load_dataset("squad_v2")``). Obtained F1-scores are being
compared to BERT-like models (BERT, mBERT and XLM-RoBERTa). 

A second step is to assess its capacities of generalization in the context of zero-shot transfer. Finetuned on an English
dataset and then directly evaluated on a multi-lingual dataset with 11 languages of various morphologies (XQuAD). 

### Structure of the folder

```
qa
├── __init__.py
├── canine_evaluate_xquad.py        # Script to evaluate CANINE on XQUaD dataset  
├── processing   
    ├── preprocessor.py             # Basic dataset preprocessor; note that the dataset you choose must have SQuAD format                     
    ├── qa_dataset.py               # Pytorch wrapper on our dataset                    
    └── tokenized_dataset.py        # Main class to prepare the dataset for the QA task
├── utils   
    ├── training_utils.py           # Main functions to help in training/evaluation (see notebooks/qa/squad_finetuning.ipynb)
    └── utils.py                    # Bunch of utils functions
└── README.md
```

## How to run experiments:

Either take a look at the folder ```sh_scripts/qa```  or manually run the following commands (note that you must go back 
to the **root** directory in that case):

```bash
# general setup
git clone 
cd canine_mva
python -m venv env
source env/bin/activate
pip install -r requirements

# To finetune CANINE on SQuADv2
python source/qa/main.py \
      --model_name google/canine-c \
      --output_dir path_to_output_dir \
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
      --clipping False 

# To evaluate our finetuned CANINE model on xquad.de
python source/qa/canine_evaluate_xquad.py \
      --model_path path_towards_your_pretrained_model \
      --language xquad.de \
      --max_answer_length 256 \
      --max_length 2048 \
      --doc_stride 512 \
      --n_best_size 20 \
      --batch_size 8 \
      --device cuda 
```

## Complementary folders

- ```notebooks/qa```: notebooks to finetune CANINE model on SQuAD like dataset and notebook to reproduce CANINE paper 
results based on the original tensorflow code.
- ```sh_scripts/qa```: bash scripts to train BERT, mBERT and XLM-ROBERTA on SQuAD like datasets. Bash script to
automatically evaluate CANINE on XQuAD rather than using the notebook (``canine_evaluate_xquad.sh``). Finally, a script 
to evaluate BERT-like models (**must already have been finetuned**) on XQuAD dataset (``bert_like_evaluate_xquad.sh``). 
For further information, please refer to the corresponding ```README.md```.

## Finetuned models

All finetuned models used in these experiments can be found [here](https://drive.google.com/drive/folders/1JVR6J8OjSTQ66fBseqHsSzoryTCQXVO_?usp=sharing).

## Results \& Observations

### Finetuning on SQuADv2

In this settings, CANINE performs decently well  (especially CANINE-c i.e. CANINE trained with Autoregressive Character Loss).

### Zero-shot transfer

In this setting, CANINE does not perform very well. On average it is -20 F1 lower than XLM-RoBERTa and -10 F1 lower than mBERT 
even if we expected CANINE to perfom better since it operates on characters and hence is free of the constraints of manually 
engineered tokenizers (which often do not work well for some languages e.g. for languages that do not use whitespaces 
such as Thai or Chinese) and fixed vocabulary. The gap between XLM-RoBERTa and CANINE-C increases when evaluated on 
languages such as Vietnamese, Thai or Chinese. These languages are mostly isolating ones i.e. language with a morpheme 
per word ratio close to one and almost no inflectional morphology.

### Discussion

In our zero-shot transfer QA experiments, CANINE does not appear to perform as well as token-based transformers such as 
mBERT. It might be because it was finetuned on English (analytical language) and hence cannot adapt well in zero-shot 
transfer especially to isolating languages (Thai, Chinese) and synthetic ones with agglutinative morphology (Turkish) 
or non-concatenative (Arabic). CANINE works decently well for languages close enough to English, e.g. Spanish or German. 
While mBERT and CANINE have both been pretrained on the top 104 languages with the largest Wikipedia using a MLM objective, 
XLM-RoBERTa was pretrained on 2.5TB of filtered CommonCrawl data containing 100 languages. This might be a confounding 
variable. Finally, we would have liked to evaluate CANINE on noisy QA dataset to see how it would perform in non-ideal 
settings; our intuition is that it might be more robust to noise in the inputs. 
