from transformers import CanineForTokenClassification,AutoModelForTokenClassification
import torch
import torch.nn as nn
from transformers import CanineModel,CanineForTokenClassification


class NERClassifier(nn.Module):

    def __init__(self, freeze_canine=False, nclasses=9,model_type="c"):
        super(NERClassifier, self).__init__()
        #  Instantiating CANINE model object
        #if model_type=="c":
        #  self.canine_layer = CanineModel.from_pretrained("google/canine-c")
        #else : 
        #  self.canine_layer = CanineModel.from_pretrained("google/canine-s")
        
        if model_type=="c":
          self.canine_layer = CanineForTokenClassification.from_pretrained("google/canine-c")
        else : 
          self.canine_layer = CanineForTokenClassification.from_pretrained("google/canine-s")

        conf = self.canine_layer.config
        conf.num_labels = nclasses
        self.canine_layer = CanineForTokenClassification(conf)

        # Freeze canine layers and only train the classification layer weights
        if freeze_canine:
            for p in self.canine_layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attn_masks):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
        '''

        outputs_canine = self.canine_layer(input_ids =input_ids,attention_mask =attn_masks)
        logits = outputs_canine.logits
        #output_lab = torch.argmax(logits,dim=2)

        return logits
        
        
class BERT_NER(nn.Module):

  def __init__(self):
        super(BERT_NER, self).__init__()

        self.BERT_layer = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

  def forward(self,input_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
        '''

        outputs_BERT = self.BERT_layer(input_ids)
        logits = outputs_BERT.logits
        #output_lab = torch.argmax(logits,dim=2)

        return logits