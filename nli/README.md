# CANINE for Medical Natural Language Inference on MedNLI data

We are interested in Natural Language Inference (NLI) on medical data using CANINE, a pre-trained tokenization-free encoder, that operates directly on character sequences without explicit tokenization and a fixed vocabulary, it is available in this [repo](https://github.com/google-research/language/tree/master/language/canine). We want to predict the relation between a hypothesis and a premise as:  Entailement, Contraction or Neutral using [MedNLI](https://jgc128.github.io/mednli/), a medical dataset annotated by doctors for NLI. We will also use BERT.

## Data 
Access for the data can be requested [here](https://jgc128.github.io/mednli/). It contains a training, validation and test set with pairs of sentences along with the label of their relation. The data must be placed in the folder `data/` . 

## Fine-tuned models
To use our fine-tuned BERT and CANINE models on MedNLI, you can download the weights in this [link](), and you should place them in the folder `trained-models/`.

To do: 
 - [ ] add link to model weights
 - [ ] add table of results nstead of text
 - [ ] change .pth to .pt in path models
 - [ ] check code