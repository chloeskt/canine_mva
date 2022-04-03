from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import f1_score

def F1_score_word(model,dataloader,device,model_type="CANINE"):
  model.eval()
  F1_score_avg = 0
  len_dataset = len(dataloader)
  with tqdm(total=len_dataset, unit_scale=True, postfix={'F1_score':0.0}) as pbar:
      
      if model_type=="CANINE":
          for it, (token_ids,attn_masks,labels,tokens_identification) in enumerate(dataloader):
              token_ids = token_ids.to(device)
              size_sentence = token_ids[token_ids!=0].shape[0] #we remove all the padding element and take the original size of sentence
              attn_masks = attn_masks.to(device)
              labels = labels[0]
              labels = list(labels.numpy())

              # Obtaining the logits from the model

              logits = Canine_NER(input_ids=token_ids, attn_masks=attn_masks)
              output_lab = torch.argmax(logits,dim=2)
              
              final_label = []
              for i in range(torch.max(tokens_identification)+1):
                indexes = list(torch.nonzero(tokens_identification==i, as_tuple=True)[1].numpy()) # get the index of the caracters of the word
                if len(indexes)==1:
                  final_label.append(output_lab[0][indexes].item())
                else:
                  label_word,_ = torch.mode(output_lab[0][indexes[0]:indexes[-1]],0) # get the label with the maximum number of occurence
                  final_label.append(label_word.item())

              F1_score_avg += f1_score(labels[0:len(final_label)], final_label,average='micro')
              pbar.set_postfix({'F1_score':F1_score_avg/(it+1)})
              pbar.update(1)

      if model_type=="BERT":
          for it, (token_ids,labels) in enumerate(dataloader):
                token_ids = token_ids.to(device)

                outputs_canine = model(token_ids)
                output_lab = torch.argmax(outputs_canine,dim=2)
                ###Original BERT classes value for NER are not same with preprocessed dataset of HuggingFace 
                output_lab = output_lab-2
                output_lab=torch.where(output_lab == -1, 7, output_lab)
                output_lab=torch.where(output_lab == 0, 8,output_lab)
                output_lab=torch.where(output_lab == -2, 0, output_lab)
                ###
                F1_score_avg += f1_score(output_lab[0][1:len(labels)+1].detach().cpu().numpy(), labels,average='micro') #We skip first token of BERT which is [CLS] token ==> output_lab[1:len(labels)+1]
                pbar.set_postfix({'F1_score':F1_score_avg/(it+1)})
                pbar.update(1)
  
  print("final F1 score is : ",F1_score_avg/(it+1))

  return F1_score_avg