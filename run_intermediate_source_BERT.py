# coding=utf-8
""" Fine-tuning GPT2 on a intermediate tasks.
Author: Karen Garcia"""
import argparse
from transformers import BertForQuestionAnswering, BertTokenizerFast, get_linear_schedule_with_warmup
import utils
from tqdm.auto import tqdm
import torch
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def add_token_positions(encodings, answers, tokenizer):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class QaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default='squad', type=str, #required=True,
                        help="Dataset name of the intermediate source task.")
    parser.add_argument("--output_dir", default='/pretrained_model', type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--overwrite_output_dir', default=False,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--per_device_train_batch_size', default=8, type=int,
                        help="Batch size per GPU/CPU for training")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=1, type=int,
                        help="Total number of training steps to perform.")


    args = parser.parse_args()
    data_train = utils.download_dataset(args.dataset_name)

    print('import BERT-base pretrained model')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenize
    train = tokenizer(data_train['context'].tolist(), data_train['question'].tolist(),
                      truncation=True, padding='max_length',
                      max_length=512, return_tensors='pt')
    add_token_positions(train, data_train['answer'], tokenizer)

    print('build datasets for both our training data')
    train_dataset = QaDataset(train)

    print('feed our Dataset to our Q&A training loop using a Dataloader')

    loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=args.per_device_train_batch_size,
                                         shuffle=True)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=0,
                                                num_training_steps=args.max_steps)
    print('Run loop')
    for epoch in range(args.num_train_epochs):
        loop = tqdm(loader)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)

            loss = outputs[0]
            loss.backward()
            optim.step()
            scheduler.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
