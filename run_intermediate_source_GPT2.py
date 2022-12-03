# coding=utf-8
""" Fine-tuning GPT2 on a intermediate tasks.
Author: Karen Garcia"""

import argparse
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import utils


def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator


def train(dataset_name,
          model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          max_steps,
          save_total_limit):


    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print('Download dataset')
    train_dataset = utils.download_dataset(dataset_name, True)
    data_collator = load_data_collator(tokenizer)

    print('Prepare for training')
    model = GPT2LMHeadModel.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_total_limit=save_total_limit
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    print('Training')
    trainer.train()
    trainer.save_model(output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default='squad', type=str,
                        help="Dataset name of the intermediate source task.")
    parser.add_argument("--output_dir", default='./pretrained_model', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name", default='gpt2', type=str,
                        help="Pretrained model name or directory")
    parser.add_argument('--overwrite_output_dir', default=False,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--per_device_train_batch_size', default=8, type=int,
                        help="Batch size per GPU/CPU for training")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=1000, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--save_total_limit", default=1, type=int,
                        help="Steps to save.")

    args = parser.parse_args()
    train(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_total_limit=args.save_total_limit
    )


if __name__ == "__main__":
    main()
