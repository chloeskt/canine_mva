# Question Answering

## Bash scripts to help launching CLI easily

There are several bash scripts meant to help you either finetune or evaluate models. In order to be able to use them, 
just as any bash script, run ``chmod +x my_script.sh`` then ``./my_script.sh`` in the current directory.

We created these scripts in order to easily change the parameters. It is similar to a CLI. 

- ``bert_finetuning.sh``: finetune BERT model (from Hugging Face) on SQuaDv2 dataset (all SQuaAD-like datasets are actually available)
- ```mbert_finetuning.sh```: finetune mBERT model (from Hugging Face) on SQuaDv2 dataset (all SQuaAD-like datasets are actually available)
- ```xlm_roberta_finetuning.sh```: finetune XLM-RoBERTa model (from Hugging Face) on SQuaDv2 dataset (all SQuaAD-like datasets are actually available)
- ```canine_evaluate_xquad.sh```: evaluate finetuned CANINE model on XQuAD dataset. In our experiments, CANINE was finetuned on SQuADv2 and we
did zero-shot transfer.
- ```bert_like_evaluate_xquad.sh```: evaluate BERT-like models (BERT, mBERT and XLM-RoBERTa) on XQuAD in a zero-shot learning setting.

## Variables

Please note that before running the scripts, you **must** check the variables that are set inside each script (usually ``MODEL_PATH`` 
and ``OUTPUT_DIR``) and modify it to our convenience.

- ```OUTPUT_DIR```: directory in each the trained model will be stored after finetuning
- ```MODEL_PATH```: path toward our finetuned model.
