# coding=utf-8
""" Fine-tuning GPT2ForSequenceClassification on a target tasks.
Author: Karen Garcia"""
import argparse
import torch
import pandas as pd
import utils
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2TokenizerFast, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize(df: pd.DataFrame, tokenizer: GPT2TokenizerFast) :
    tokenized_df = pd.DataFrame(
        df.text.apply(tokenizer).tolist()
    )
    return (
        pd.merge(
            df,
            tokenized_df,
            left_index=True,
            right_index=True,
        )
        .drop(columns="text")
        .to_dict("records")
    )

def compute_metrics(pred):
    labels = pred.label_ids
    pred = pred.predictions
    mse = mean_squared_error(labels, pred)

    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)


    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mse":mse}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", default='gpt2', type=str,
                        help="Path to the pretrained_model.")
    parser.add_argument("--max_steps", default=100, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--eval_steps", default=10, type=int,
                        help="Number of batches.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()

    train_df = utils.download_dataset('glue', True)
    test_df = utils.download_dataset('glue', False)


    config = GPT2Config.from_pretrained(
        args.pretrained_model,
        pad_token_id=50256,  # eos_token_id
        num_labels=1,
    )
    tokenizer = GPT2TokenizerFast.from_pretrained(
        config.model_type,
        padding=True,
        truncation=True,
        pad_token_id=config.pad_token_id,
        pad_token="<|endoftext|>",  # eos_token
    )

    tokenizer.pad_token
    model = GPT2ForSequenceClassification(config)

    train_ds = tokenize(train_df, tokenizer)
    test_ds = tokenize(test_df, tokenizer)

    training_args = TrainingArguments(
        report_to="none",
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        output_dir="./output_target_task",
        greater_is_better=False,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()