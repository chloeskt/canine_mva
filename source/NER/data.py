from transformers import CanineTokenizer,AutoTokenizer
import torch
from torch.utils.data import Dataset


class CustomConll(Dataset):

    def __init__(self, dataset, maxlen=2048, with_labels=True,model_type="c",mode="train",word_F1_score=False,model ='CANINE'):
        '''
        * model_type : should be c or s depending on model training
        * mode : should be train, validation or test
        '''
        self.word_F1_score = word_F1_score
        self.dataset = dataset[mode]  # Preprosessed Dataset from HuggingFace Conll2002
        self.maxlen = maxlen #TBC
        self.model = model
        #Initialize the tokenizer
        if self.model == 'CANINE':
            if model_type=="c":
              self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c") 
            else :
              self.tokenizer = CanineTokenizer.from_pretrained("google/canine-s") 
        elif self.model == 'BERT':
              self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
              if self.maxlen >512:
                print("Maximum token length for BERT is set to 512")
                self.maxlen=512

    def __len__(self):
        return len(self.dataset)-1

    def __getitem__(self, index):
        
        tokens_origin = self.dataset['tokens'][index]
        label_origin =  self.dataset['ner_tags'][index]

        token_identification = [] ## this is used to compute score on word (unlike on caracter as previously)
        label_token = []

        sentence = tokens_origin[0]

        label_token+=([label_origin[0]] * len(tokens_origin[0]))
        token_identification += ([0] * len(tokens_origin[0]))

        for i in range(1,len(tokens_origin)):
          if tokens_origin[i] != "," and tokens_origin[i] != "." and tokens_origin[i] != ")" and tokens_origin[i-1] != "(" and tokens_origin[i] != "?" and tokens_origin[i] != "!": #Detokenize the sentence
            sentence += " "+tokens_origin[i]
            label_token+= [0]
            label_token+= ([label_origin[i]] * len(tokens_origin[i]))

            token_identification +=[-1] #for space token
            token_identification += ([i] * len(tokens_origin[i]))

          else : 
            sentence+=tokens_origin[i]
            label_token+= ([label_origin[i]] * len(tokens_origin[i]))
            token_identification += ([i] * len(tokens_origin[i]))

        inputs = self.tokenizer(sentence,
                                padding='max_length',  # Pad to max_length
                                truncation=True,
                                max_length=self.maxlen,
                                return_tensors="pt")
        
        if self.model =='BERT':
            return inputs['input_ids'],label_origin

        ### part for CANINE training ###
        token_ids = inputs['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = inputs['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values

        label_token+=([0] * (self.maxlen-len(label_token))) #padding label tensor
        label_final = torch.tensor(label_token)
        label_final = label_final[0:2048] #in case the sentence was padded
        token_identification = torch.tensor(token_identification)
        token_identification = token_identification[0:2048]
        
        if self.word_F1_score:
          label_final=torch.tensor(label_origin)
          return token_ids,attn_masks,label_final,token_identification
        
        else :
          return token_ids,attn_masks,label_final