This task was fully developped by Yujin CHO for Algorithms for Speech and Natural Language Processing course.
It allow the user to train and test a CANINE model for NER task on ConLL2002/2003 dataset
It also include a test of BERT model for NER task on ConLL2002/2003 (unfortunatelly the F1 score calculation need to be debuged with BERT model)

You need to launch main.py with the following options :
    --base_dir : Path to the pre computed weights
    --weight_name : name of the pre computed weights
    --model_type : CANINE or BERT
    --canine_mode : c or s
    --mode : FULL for training + evaluation, TRAIN or EVAL
    --freeze_canine : if True, freeze the encoder weights and only update the classification layer weights (only for CANINE)
    --maxlen : maximum length of the tokenized input sentence
    --lr : learning rate (only for CANINE)
    --epochs : number of training epochs (only for CANINE)
    --train_lang : ES for espagnol, NL for Dutch and EN for English (only for CANINE)
    --eval_lang : ES for espagnol, NL for Dutch and EN for English
    
to train a CANINE-C on English NER task, you can launch the following command :
!python3 main.py --model_type CANINE --canine_mode c --mode FULL --epochs 4 --train_lang EN --eval_lang EN
