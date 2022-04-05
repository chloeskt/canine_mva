import sys
from dataclasses import field, dataclass
from pathlib import Path
from typing import Tuple, List

sys.path.append(Path(__file__).parent.parent.parent.as_posix())

from datasets import load_dataset, load_metric, Dataset
from transformers import (
    CanineForQuestionAnswering,
    CanineTokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader

from source.qa import (
    TokenizedDataset,
    Preprocessor,
    set_seed,
    to_pandas,
    remove_examples_longer_than_threshold,
    train,
)

seed = 0
set_seed(seed)


@dataclass
class TrainingArguments:
    """
    Arguments needed to train CANINE model on SQuAD-like dataset (inspired from HuggingFace run_qa.py script)
    """

    model_name: str = field(
        default=None,
        metadata={
            "help": "Name of the pretrained checkpoint: google/canine-s or google/canine-c"
        },
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Output directory, will be used to store finetuned model."},
    )
    learning_rate: float = field(
        default=None,
        metadata={"help": "Value of the learning rate for AdamW optimizer"},
    )
    weight_decay: float = field(
        default=None,
        metadata={"help": "Value of the weight decay for AdamW optimizer"},
    )
    nb_epochs: int = field(default=None, metadata={"help": "Number of training epochs"})
    best_f1: float = field(
        default=None,
        metadata={
            "help": "Threshold to save finetuned model based on F1 score. Model will be saved iff its F1-score "
                    "> best_f1"
        },
    )
    warmup_proportion: float = field(
        default=None,
        metadata={
            "help": "Warmup proportion for `get_linear_schedule_with_warmup`. Note that this argument will be "
                    "used only if `lr_scheduler` == True"
        },
    )
    lr_scheduler: bool = field(
        default=None,
        metadata={
            "help": "If set to True, `get_linear_schedule_with_warmup` will be used."
        },
    )
    freeze: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to freeze CANINE layers and only train last layer. If True, only "
                    "(qa_outputs) layer weights will be updated."
        },
    )
    clipping: bool = field(
        default=False,
        metadata={"help": "If set to True, model's weights will be clipped."},
    )
    squad_v2: bool = field(
        default=False,
        metadata={
            "help": "If true, dataset is similar as SQUADv2 some of the examples do not have an "
                    "answer."
        },
    )
    drive: bool = field(
        default=False, metadata={"help": "Whether or not you are using Google Colab"}
    )
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=512,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    max_answer_length: int = field(
        default=256,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    batch_size: int = field(
        default=12,
        metadata={"help": "Batch size"},
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device, either 'cuda' or 'cpu'"},
    )


def train_canine(
        model_name: str,
        output_dir: str,
        doc_stride: int,
        max_length: int,
        max_answer_length: int,
        n_best_size: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        nb_epochs: int,
        warmup_proportion: float,
        best_f1: float,
        squad_v2: bool,
        freeze: bool,
        lr_scheduler: bool,
        drive: bool,
        clipping: bool,
) -> Tuple[List[float], List[float]]:
    datasets = load_dataset("squad_v2" if squad_v2 else "squad")

    preprocessor = Preprocessor(datasets)
    datasets = preprocessor.preprocess()

    df_train = to_pandas(datasets["train"])
    df_validation = to_pandas(datasets["validation"])

    df_train = remove_examples_longer_than_threshold(
        df_train, max_length=max_length * 2, doc_stride=doc_stride
    )
    df_validation = remove_examples_longer_than_threshold(
        df_validation, max_length=max_length * 2, doc_stride=doc_stride
    )

    datasets["train"] = Dataset.from_pandas(df_train)
    datasets["validation"] = Dataset.from_pandas(df_validation)

    tokenizer = CanineTokenizer.from_pretrained(model_name)

    tokenizer_dataset = TokenizedDataset(
        tokenizer, max_length, doc_stride, squad_v2, language="en"
    )
    tokenized_datasets = datasets.map(
        tokenizer_dataset.tokenize,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    tokenized_datasets["train"] = tokenized_datasets["train"].remove_columns(
        ["example_id"]
    )

    validation_examples = tokenized_datasets["validation"]
    tokenized_datasets["validation"] = tokenized_datasets["validation"].remove_columns(
        ["example_id"]
    )

    metric = load_metric("squad_v2" if squad_v2 else "squad")

    tokenized_datasets["train"].set_format(
        "torch"
    )  # set into pytorch format for dataloader
    tokenized_datasets["validation"].set_format(
        "torch"
    )  # set into pytorch format for dataloader

    # initialize data loader for training data
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    # initialize validation set data loader
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    model = CanineForQuestionAnswering.from_pretrained(model_name)

    if freeze:
        optimizer = torch.optim.AdamW(
            model.qa_outputs.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )

    if lr_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_proportion,
            num_training_steps=len(train_loader) * nb_epochs,
        )

    print("Start training")
    train_losses, val_losses = train(
        model=model,
        num_epochs=nb_epochs,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=datasets["validation"],
        features_val_dataset=validation_examples,
        tokenizer=tokenizer,
        metric=metric,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_answer_length=max_answer_length,
        output_dir=output_dir,
        best_f1=best_f1,
        lr_scheduler=lr_scheduler,
        drive=drive,
        squad_v2=squad_v2,
        n_best_size=n_best_size,
        clipping=clipping,
    )
    return train_losses, val_losses


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print(f"Launching training script for {args.model_name}")
    train_losses, val_losses = train_canine(
        model_name=args.model_name,
        output_dir=args.output_dir,
        doc_stride=args.doc_stride,
        max_length=args.max_length,
        max_answer_length=args.max_answer_length,
        n_best_size=args.n_best_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nb_epochs=args.nb_epochs,
        warmup_proportion=args.warmup_proportion,
        best_f1=args.best_f1,
        squad_v2=args.squad_v2,
        freeze=args.freeze,
        lr_scheduler=args.lr_scheduler,
        drive=args.drive,
        clipping=args.clipping,
    )
    print(
        "List of training losses: ",
        train_losses,
        "\n",
        "List of validation losses: ",
        val_losses
    )
    print(
        "---------------------------------------------------------------------------------------------------------"
    )
    print()
