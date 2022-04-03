# Final project for the course Algorithms for speech and natural language processing - Master MVA, ENS Paris-Saclay (2021-2022)

This repository contains the work done by Yujin Cho, Loubna Ben Allal, Gabriel Baker and Chloé Sekkat for their final 
project in the course ``Algorithms for speech and natural language processing`` at the master MVA, ENS Paris-Saclay. 

For this project, we worked on a tokenization-free encoder for language representation called [CANINE](https://arxiv.org/abs/2103.06874). 
It is the first character-level pre-trained deep encoder that is completely tokenization-free and vocabulary-free. We studied the 
performance of CANINE on several NLP tasks and compare it to other models (similar to BERT).

# Installation 

The code was tested with Python3.8.6

```python
pip install -r requirements
```

Please note that you will need to update manually the versions of ``torch``, ``torchaudio`` and ``torchvision`` depending
on your hardware.

# Structure of the repository

This repository has been divided into 3 main categories, each containing sub-categories depending on the task we evaluated
the models on. 

```
canine_mva
├── requirements.txt
├── sh_scripts                  # Bash scripts
    └── qa/                     # QA related bash scripts (CLI)
├── source                      # Source code main package   
    ├── qa/                     # QA related source code                               
    └── nli/                    # NLI related source code  
    └── NER/                    # NER related source code  
├── notebooks                   # Notebooks (showcase of experiments)
    └── qa/                     # QA related notebooks
└── README.md
```

# Question Answering:
- ``notebooks/qa``: the notebooks we used for our QA experiments. 
- ``source/qa``: source code needed to prepare and train CANINE for Question Answering task (see application in the notebook
``squad_finetuning``)
- ``sh_scripts/qa``: collection of bash scripts to finetune and evaluate models on QA tasks.

# NLI
- ``source/nli``: source code needed to prepare and train CANINE and BERT for NLI task (for more information, refer to 
corresponding ``README.md``)

# NER
- ``source/NER``: source code needed to prepare and train CANINE for NER task (for more information, refer to corresponding ``README.md``)
