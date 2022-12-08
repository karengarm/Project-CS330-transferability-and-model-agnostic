# coding=utf-8
""" Fine-tuning GPT2ForSequenceClassification on a target tasks.
Authors: Karen Garcia, Phillip Yao-Lakaschus"""
import argparse
import wget
import tarfile
from datasets import load_dataset
import io
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import GPT2Config
from transformers import (
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    GPT2ForSequenceClassification)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_STEPS = 500
EPOCHS = 20


def download_data_set():
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = wget.download(url)
    file = tarfile.open(filename)
    file.extractall()
    file.close()
    labels_ids = {'neg': 0, 'pos': 1}
    n_labels = len(labels_ids)
    return filename, labels_ids, n_labels


class PytorchDataset(Dataset):
    r"""PyTorch Dataset class for loading data.
    This is where the data parsing happens.
    Arguments:
      path (:obj:`str`):
          Path to the data partition.
    """

    def __init__(self, path, use_tokenizer):

        # Check if path exists.
        if not os.path.isdir(path):
            # Raise error if path is invalid.
            raise ValueError('Invalid `path` variable! Needs to be a directory')
        self.texts = []
        self.labels = []
        # Since the labels are defined by folders with data we loop
        # through each label.
        for label in ['pos', 'neg']:
            sentiment_path = os.path.join(path, label)

            # Get all files from path.
            files_names = os.listdir(sentiment_path)  # [:10] # Sample for debugging.
            # Go through each file and read its content.
            for file_name in tqdm(files_names, desc=f'{label} files'):
                file_path = os.path.join(sentiment_path, file_name)

                # Read content.
                content = io.open(file_path, mode='r', encoding='utf-8').read()
                # Fix any unicode issues.
                # content = fix_text(content)
                # Save content.
                self.texts.append(content)
                # Save encode labels.
                self.labels.append(label)
        # Number of exmaples.
        self.n_examples = len(self.labels)
        return

    def __len__(self):
        r"""When used `len` return the number of examples.
        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.
        Arguments:
          item (:obj:`int`):
              Index position to pick an example to return.
        Returns:
          :obj:`Dict[str, str]`: Dictionary of inputs that contain text and
          asociated labels.
        """

        return {'text': self.texts[item],
                'label': self.labels[item]}


class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classification task.
    Arguments:
      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.
      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.
      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.
    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.
        Arguments:
          item (:obj:`list`):
              List of texts and labels.
        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['sentence'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        # labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})

        return inputs


def train(model, dataloader, optimizer_, scheduler_, device_):
    r"""
    Train pytorch model on a single pass through the data loader.
    Arguments:
        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.
        optimizer_ (:obj:`transformers.optimization.AdamW`):
            Optimizer used for training.
        scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            PyTorch scheduler.
        device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.
    Returns:
        :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss].
    """

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        if idx % 10 == 0:
            print("\nbatch id: ", idx)
            print("true labels: ", batch['labels'])
            print("pred labels: ", logits.argmax(axis=-1).flatten().tolist())
            print("accuracy: ", accuracy_score(true_labels, predictions_labels))
        if idx >= MAX_STEPS: break

    avg_epoch_loss = total_loss / len(predictions_labels)
    return true_labels, predictions_labels, avg_epoch_loss


def validation(model, dataloader, device_):
    r"""Validation function to evaluate model performance on a
    separate set of data.
    Arguments:
      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.
      device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.
    """
    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    model.eval()

    # Evaluate data for one epoch
    for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()

            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content
        if idx >= MAX_STEPS: break

    avg_epoch_loss = total_loss / len(predictions_labels)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--pretrained_model", default='src/logs/gpt2-sm_tasks_sst2_k_1/model', type=str, help="Path to the pretrained_model.")
    # parser.add_argument("--pretrained_model", default='gpt2', type=str, help="Path to the pretrained_model.")
    parser.add_argument("--pretrained_model", default='src/logs/qa_tasks_qa_set_high_div/model', type=str,
                        help="Path to the pretrained_model.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Number of batches.")
    parser.add_argument("--max_length", default=512, type=int,
                        help="Pad or truncate text sequences to a specific length.")

    args = parser.parse_args()

    # filename, labels_ids , n_labels = download_data_set()

    labels_ids = {'neg': 0, 'pos': 1}
    n_labels = len(labels_ids)

    # Get model configuration.
    print('Loading configuration...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Get the actual model.
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.pretrained_model,
                                                          config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    print('Model loaded to `%s`' % device)
    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=labels_ids,
                                                              #   max_sequence_len=args.max_length,
                                                              )

    print('Dealing with Train...')
    # Create pytorch dataset.
    # train_dataset = PytorchDataset(path='src/aclImdb/train',use_tokenizer=tokenizer)
    train_dataset = load_dataset('glue', 'sst2', split='train')
    print('Created `train_dataset` with %d examples!' % len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!' % len(train_dataloader))

    print()

    print('Dealing with Validation...')
    # Create pytorch dataset.
    # valid_dataset = PytorchDataset(path='src/aclImdb/test', use_tokenizer=tokenizer)
    valid_dataset = load_dataset('glue', 'sst2', split='validation')
    print('Created `valid_dataset` with %d examples!' % len(valid_dataset))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!' % len(valid_dataloader))

    optimizer = AdamW(model.parameters(),
                      lr=5e-5,
                      eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(MAX_STEPS / 5),
                                                num_training_steps=MAX_STEPS)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}
    all_valid_preds = []
    all_valid_labels = []

    # Loop through each epoch.
    print('Epoch')
    for epoch in tqdm(range(EPOCHS)):
        print()
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(model, valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
            train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        print(classification_report(valid_labels, valid_predict))
        # all_valid_preds.extend(valid_predict)
        # all_valid_labels.extend(valid_labels)
        # all_loss['train_loss'].append(train_loss)
        # all_loss['val_loss'].append(val_loss)
        # all_acc['train_acc'].append(train_acc)
        # all_acc['val_acc'].append(val_acc)

        model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.pretrained_model,
                                                              config=model_config)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id
        model.to(device)
        print('Model loaded to `%s`' % device)
        # Create data collator to encode text and labels into numbers.
        gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, labels_encoder=labels_ids)
        optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(MAX_STEPS / 5),
                                                    num_training_steps=MAX_STEPS)
        all_valid_preds.extend(valid_predict)
        all_valid_labels.extend(valid_labels)
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)
    print('train_loss avg: ', sum(all_loss['train_loss']) / EPOCHS)
    print('train_loss st: ', np.std(np.array(all_loss['train_loss'])))
    print('val_loss', sum(all_loss['val_loss']) / EPOCHS)
    print('val_loss st: ', np.std(np.array(all_loss['val_loss'])))
    print('train_acc', sum(all_acc['train_acc']) / EPOCHS)
    print('train_acc st: ', np.std(np.array(all_acc['train_acc'])))
    print('val_acc', sum(all_acc['val_acc']) / EPOCHS)
    print('val_acc st: ', np.std(np.array(all_acc['val_acc'])))
    print(classification_report(all_valid_labels, all_valid_preds))
    print("done")


if __name__ == "__main__":
    main()