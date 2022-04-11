# Question Answering 

## Organization

This subfolder contains the whole code associated with the Question Answering experiments. It has been developed by Chloé 
Sekkat and can be viewed as a Python package whose main functions/classes can be found in the ``__init__.py``.

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

A third experiment is to test the abilities of CANINE to handle noisy inputs, especially noisy questions as in real life
settings the questions are often noisy (misspellings, wrong grammar, etc - think of ASR systems or keyboard error while
typing).

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
python source/qa/canine_evaluation.py \
      --model_path path_towards_your_pretrained_model \
      --dataset_name xquad \
      --data_dir None \
      --huggingface_model_checkpoint name_of_hugging_face_checkpoint \
      --language xquad.de \
      --squad_v2 False \
      --max_answer_length 256 \
      --max_length 2048 \
      --doc_stride 512 \
      --n_best_size 20 \
      --batch_size 8 \
      --device cuda 
      
# To create a noisy version of SQuADv2
python3 source/qa/processing/noisifier.py \
      --dataset_name squad_v2 \
      --output_dir path_to_output_dir \
      --noise_level 0.1 \
      --augmenter_type KeyboardAug
      
      
# To evaluate our finetuned CANINE model on noisy validation/test set of SQuADv2 (created above)
python source/qa/canine_evaluation.py \
      --model_path path_towards_your_pretrained_model \
      --dataset_name noisy \
      --data_dir path_to_directory_containing_noisy_data \
      --huggingface_model_checkpoint name_of_hugging_face_checkpoint \
      --language xquad.en \
      --squad_v2 True \
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

They were trained with the following parameters:

|                               | **BERT** | **mBERT** | **XLM-RoBERTa** | **CANINE-C** | **CANINE-S** |
|:-----------------------------:|:--------:|:---------:|:---------------:|:------------:|:------------:|
| Training time for 1 epoch     | 1h42     | 1h39      |      2h03       | 4h28         | 4h13         |
| Batch size                    | 6        | 6         |        6        | 6            | 6            |
| Learning Rate                 | 3e-5     | 3e-5      |      3e-5       | 5e-5         | 5e-5         |
| Weigh decay                   | 0        | 0         |        0        | 0.01         | 0.1          |
| Nb of epochs                  | 2        | 4         |        4        | 2            | 6            |
| Number of training examples   | 132335   | 132335    |     131823      | 130303       | 130303       |
| Number of validation examples | 12245    | 12245     |      12165      | 11861        | 11861        |
| Max length                    | 348      | 348       |       348       | 2048         | 2048         |
| Doc stride                    | 128      | 128       |       128       | 512          | 512          |
| Max answer length             | 30       | 30        |       30        | 256          | 256          |


## Results \& Observations

### Finetuning on SQuADv2

|          | **CANINE-C** | **CANINE-S** | **mBERT** | **BERT** | **XLM-RoBERTa** |
|:--------:|:------------:|:------------:|:---------:|:--------:|:---------------:|
| F1-score | 74,1         |     72,5     | 77,51     | 76,02    | 78,3            |
| EM score | 69,2         |     69,6     | 74,1      | 73,08    | 75,12           |

In this settings, CANINE performs decently well  (especially CANINE-c i.e. CANINE trained with Autoregressive Character Loss).

### Zero-shot transfer

In this setting, CANINE does not perform very well. On average it is -20 F1 lower than XLM-RoBERTa and -10 F1 lower than mBERT 
even if we expected CANINE to perfom better since it operates on characters and hence is free of the constraints of manually 
engineered tokenizers (which often do not work well for some languages e.g. for languages that do not use whitespaces 
such as Thai or Chinese) and fixed vocabulary. The gap between XLM-RoBERTa and CANINE-C increases when evaluated on 
languages such as Vietnamese, Thai or Chinese. These languages are mostly isolating ones i.e. language with a morpheme 
per word ratio close to one and almost no inflectional morphology.

#### F1 scores:
|            | **CANINE-C** | **CANINE-S** | **mBERT-base** | **BERT-base** | **XLM-RoBERTa** |
|:----------:|:------------:|:------------:|:--------------:|:-------------:|:---------------:|
| English    | 78,77        | 79,03        | 83,59          | 82,3          | 82,8            |
| Arabic     | 43,78        | 29,74        | 54,09          | 11,76         | 62,48           |
| German     | 59,57        | 55,35        | 68,4           | 19,41         | 72,47           |
| Greek      | 46,93        | 30,82        | 56,47          | 10,21         | 70,93           |
| Spanish    | 60,47        | 59,48        | 72,84          | 19,72         | 75,18           |
| Hindi      | 35,21        | 30,93        | 51,06          | 11,07         | 62,1            |
| Russian    | 60,49        | 55,09        | 68,33          | 9,47          | 73,12           |
| Thai       | 37,28        | 31,2         | 27,63          | 10,04         | 65,21           |
| Turkish    | 31,09        | 23,83        | 44,62          | 16,76         | 65,34           |
| Vietmanese | 43,14        | 35,52        | 64,49          | 24,63         | 73,44           |
| Chinese    | 34,86        | 28,68        | 52,71          | 8,15          | 65,68           |
| Romanian   | 56,62        | 43,69        | 69,31          | 20,03         | 74,78           |
| Average    | 49,02        | 41,95        | 59,46          | 20,30         | 69,16           |

#### Exact Match:
|            | **CANINE-C** | **CANINE-S** | **mBERT-base** | **BERT-base** | **XLM-RoBERTa** |
|:----------:|:------------:|:------------:|:--------------:|:-------------:|:---------------:|
| English    | 67,38        | 66,34        | 79,51          | 69,57         | 72,18           |
| Arabic     | 26,25        | 13,75        | 37,22          | 4             | 45,79           |
| German     | 43,16        | 38,27        | 50,84          | 4,9           | 55,21           |
| Greek      | 29,14        | 13,42        | 40,16          | 5,37          | 53,19           |
| Spanish    | 42,74        | 39,57        | 54,45          | 4,7           | 56,3            |
| Hindi      | 18,93        | 16,54        | 36,97          | 4,8           | 45,042          |
| Russian    | 43,48        | 35,65        | 52,1           | 4,62          | 55,54           |
| Thai       | 20,5         | 17,91        | 21,26          | 2,6           | 54,28           |
| Turkish    | 14,8         | 10,11        | 29,41          | 4,87          | 48,85           |
| Vietmanese | 25,17        | 19,65        | 45,21          | 7,64          | 54,02           |
| Chinese    | 21,36        | 20,2         | 42,26          | 3,1           | 55,63           |
| Romanian   | 39,98        | 26,5         | 54,62          | 6,21          | 61,26           |
| Average    | 32,74        | 26,49        | 45,33          | 10,20         | 53,19           |


### Noisy questions on SQuADv2

In this experience, the goal is to evaluate the models' robustness of noise. To do so, we created 3 noisy versions of
the SQuADv2 dataset where the questions have been artificially enhanced with noisy (in our case we chose ``RandomCharAug``
from ``nlpaug`` library with action `substitute` but in our package 4 other types of noise have been developed - refer to `processing/noisifier.py`).

Three levels of noise were chosen: 10\%, 20\% and 40\% (similar to NLI and Sentiment Analysis experiments). Each word
gets transformed with probability $p$ into a misspelled version of it (see [nlpaug documentation](https://github.com/makcedward/nlpaug/blob/master/nlpaug/augmenter/char/random.py)
for more information).

The noise is **only** applied to the test set (on SQuADv2) made of 1187 examples. We compared the 5 models we finetuned 
on the clean version of SQuADv2 (first experiment) on these 3 noisy datasets (on for each level of $p$). The following
table gathers the results (averaged over 3 runs):

| **Type of noise: RandomCharAug - substitute** 	| **Noise level 10%** 	|        	| **Noise level 20%** 	|        	| **Noise level 40%** 	 | 	        |
|-----------------------------------------------	|---------------------	|--------	|---------------------	|--------	|-----------------------|----------|
|                                               	| **F1 score**        	| **EM** 	| **F1 score**        	| **EM** 	| **F1 score**        	 | **EM** 	 |
| **CANINE-C**                                  	| 69,64               	| 66,89  	| 67,88               	| 65,43  	| 66,03               	 | 63,9   	 |
| **CANINE-S**                                  	| 72,25               	| 69,65  	| 70,3                	| 68,03  	| **67,18**           	 | 64,6   	 |
| **BERT**                                      	| 73,68               	| 70,79  	| 71,22               	| 68,55  	| 66,42               	 | 63,74  	 |
| **mBERT**                                     	| 74                  	| 70,75  	| 71,66               	| 68,46  	| 67,08               	 | 64,74  	 |
| **XLM-RoBERTa**                               	| **74,54**           	| 71,61  	| **72,68**           	| 69,81  	| **67,12**           	 | 64,43  	 |

Overall XLM-RoBERTa is a very powerful model, it is the best in all experiences we attempted. However it is worth 
highlighting that once the noise level is high (i.e. > 40\%), both CANINE-C and CANINE-S perform similarly to BERT-like 
models. CANINE-S is even better than mBERT and BERT. CANINE-S does seem to fairly robust to high level of 
artificial noise. 

Further experiments should be run with other types of noise to confirm these results.

### Discussion

In our zero-shot transfer QA experiments, CANINE does not appear to perform as well as token-based transformers such as 
mBERT. It might be because it was finetuned on English (analytical language) and hence cannot adapt well in zero-shot 
transfer especially to isolating languages (Thai, Chinese) and synthetic ones with agglutinative morphology (Turkish) 
or non-concatenative (Arabic). CANINE works decently well for languages close enough to English, e.g. Spanish or German. 
While mBERT and CANINE have both been pretrained on the top 104 languages with the largest Wikipedia using a MLM objective, 
XLM-RoBERTa was pretrained on 2.5TB of filtered CommonCrawl data containing 100 languages. This might be a confounding 
variable. Also, CANINE-S seems to be robust to high level of artificial noise and even slightly better than BERT and 
mBERT. Finally, one might also note that multilingual model do, overall, have better capacities of generalization and
better scores on these Question Answering tasks.
