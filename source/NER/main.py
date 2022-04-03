import argparse

from data import CustomConll
from model import NERClassifier,BERT_NER()
from train import evaluate_loss,train_canine_NER
from utils import F1_score_word

from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


import os
import matplotlib.pyplot as plt
import copy
import random
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description='NER Task CANINE vs BERT')

# Weight saved
parser.add_argument('--base_dir', type=str, default='/content/drive/MyDrive/ENS/NLP/models',
                    help='Path to the pre computed weights')
parser.add_argument('--weight_name', type=str, default='CANINE_lr_2e-05_val_loss_0.01892_ep_4.pt',
                    help='Path to the pre computed weights')

#Model selection
parser.add_argument('--model_type', type=str, default='CANINE',
                    help='CANINE or BERT')
parser.add_argument('--canine_mode', type=str, default='c',
                    help='c or s')
parser.add_argument('--mode', type=str, default='FULL',
                    help='FULL for training + evaluation, TRAIN or EVAL')
parser.add_argument('--freeze_canine', type=bool, default=False,
                    help='if True, freeze the encoder weights and only update the classification layer weights')
parser.add_argument('--maxlen', type=int, default=2048,
                    help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded ')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=4,
                    help='# number of training epochs')
parser.add_argument('--train_lang', type=str, default='ES',
                    help='# ES for espagnol, NL for Dutch and EN for English')
parser.add_argument('--eval_lang', type=str, default='ES',
                    help='# ES for espagnol, NL for Dutch and EN for English')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("¤"*50)
print('¤¤¤¤¤ Creating model ',args.model_type, '¤¤¤¤¤')
if args.model_type == 'CANINE':
    model = NERClassifier().to(device)
elif args.model_type == 'BERT':
    model = BERT_NER().to(device)
    print('¤¤¤¤¤ only evalutation in EN mode is available ¤¤¤¤¤')
    args.eval_lang = 'EN'

if args.train_lang or args.eval_lang == 'ES':
    print("¤"*50)
    print("¤¤¤¤¤ loading CONLL2002 dataset : Espagnol version ¤¤¤¤¤")
    dataset_es = load_dataset("conll2002",'es')
    if args.train_lang == 'ES':
        train_set = CustomConll(dataset_es,model_type=args.canine_mode)
        val_set = CustomConll(dataset_es,model_type=args.canine_mode,mode="validation")
        train_loader = DataLoader(train_set, batch_size=1, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=2)
    if args.eval_lang == 'ES':
        test_set = CustomConll(dataset_es,model_type=args.canine_mode,mode="test",word_F1_score=True)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=2)

if args.train_lang or args.eval_lang == 'NL':
    print("¤"*50)
    print("¤¤¤¤¤ loading CONLL2002 dataset : Dutch version ¤¤¤¤¤")
    dataset_nl = load_dataset("conll2002",'nl')
    if args.train_lang == 'NL':
        train_set = CustomConll(dataset_nl,model_type=args.canine_mode)
        val_set = CustomConll(dataset_nl,model_type=args.canine_mode,mode="validation")
        train_loader = DataLoader(train_set, batch_size=1, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=2)
    if args.eval_lang == 'NL':
        test_set = CustomConll(dataset_nl,model_type=args.canine_mode,mode="test",word_F1_score=True)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=2)

if args.train_lang or args.eval_lang == 'EN':
    print("¤"*50)
    print("¤¤¤¤¤ loading CONLL2002 dataset : English version ¤¤¤¤¤")
    dataset_en = load_dataset("conll2003")
    if args.train_lang == 'EN':
        train_set = CustomConll(dataset_en,model_type=args.canine_mode)
        val_set = CustomConll(dataset_en,model_type=args.canine_mode,mode="validation")
        train_loader = DataLoader(train_set, batch_size=1, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=2)
    if args.eval_lang == 'EN':
        if args.model_type == "CANINE":
            test_set = CustomConll(dataset_en,model_type=args.canine_mode,mode="test",word_F1_score=True)
            test_loader = DataLoader(test_set, batch_size=1, num_workers=2)
        if args.model_type == "BERT":
            test_set = CustomConll(dataset_en,model_type=args.canine_mode,mode="test",word_F1_score=True,model="BERT")

if args.mode == 'EVAL':
    if args.model_type == 'CANINE':
      print("¤"*50)
      print("¤¤¤¤¤ loading CANINE pretrained weights on NER task ¤¤¤¤¤")
      path = os.path.join(args.base_dir,args.weight_name)
      model.load_state_dict(torch.load(path))
elif args.mode == 'TRAIN' or args.mode == 'FULL':
    print("¤"*50)
    print("¤¤¤¤¤ Start training with",args.eval_lang," dataset ¤¤¤¤¤")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    num_warmup_steps = 0                                      # The number of steps for the warmup phase.
    num_training_steps = args.epochs * len(train_loader)           # The total number of training steps
    t_total = len(train_loader) * args.epochs                      # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    model = train_canine_NER(model,device, criterion, optimizer, args.lr, lr_scheduler, train_loader, val_loader, args.epochs)

if args.mode == 'EVAL' or args.mode == 'FULL' :
    print("¤"*50)
    print("¤¤¤¤¤ Start evaluation with ",args.eval_lang," dataset ¤¤¤¤¤")
    if args.model_type == 'CANINE':
      score = F1_score_word(model,test_loader,device)
    else:
      score = F1_score_word(model,test_set,device,model_type="BERT")
    print("¤"*50)
