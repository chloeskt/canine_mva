# Question Answering 

## Organization

This subfolder contains the whole code associated with the Question Answering task, it can be viewed as a Python package
whose main functions/classes can be found in the ``__init__.py``.

## Description

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

## Complementary folders

- ```notebooks/qa```: notebooks to finetune CANINE model on SQuAD like dataset and notebook to reproduce CANINE paper 
results based on the original tensorflow code.
- ```sh_scripts/qa```: bash scripts to train BERT, mBERT and XLM-ROBERTA on SQuAD like datasets. Bash script to
automatically evaluate CANINE on XQuAD rather than using the notebook (``canine_evaluate_xquad.sh``). Finally, a script 
to evaluate BERT-like models (**must already have been finetuned**) on XQuAD dataset (``bert_like_evaluate_xquad.sh``).
