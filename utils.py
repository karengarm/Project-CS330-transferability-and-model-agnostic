# coding=utf-8
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import TensorDataset


def dataset_information(dataset: str) -> str:
    dict_dataset = {
        'squad_v2': [{'type': 'QA',
                      'url': 'HuggingFace'
                      }],
        'squad': [{'type': 'QA',
                   'url': 'HuggingFace'}],
        # 'mlqa': [{'type': 'QA',
        #           'url': 'HuggingFace'
        #           }],
        'quoref': [{'type': 'QA',
                    'url': 'HuggingFace'
                    }],
        'glue': [{'subset': 'sst2',
                'type': 'TC',
                'url': 'HuggingFace'
                }]

    }
    if dataset in dict_dataset:
        result = dict_dataset[dataset]
    else:
        result = None
        print("Dataset not found")

    return result


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True


def prepare_data_for_training_QA(train_dataset):
    context_list = []
    question_list = []
    doc_tokens_list = []
    answer_text_list = []
    start_position_list = []
    end_position_list = []
    is_impossible_list = []
    n_paragraph = len(train_dataset)
    for p in range(n_paragraph):
        context = train_dataset[p]['context']
        context_list.append(context)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    # doc_tokens: Tokens in the document separated by comma in a list
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        doc_tokens_list.append(doc_tokens)
        question_list.append(train_dataset[p]['question'])
        answer = train_dataset[p]['answers']
        if len(answer['text']) > 0:
            is_impossible_list.append(False)
            text = answer['text'][0]
            answer_text_list.append(text)
            startidx = char_to_word_offset[answer['answer_start'][0]]
            endidx = char_to_word_offset[answer['answer_start'][0] + len(text) - 1] + 1
            start_position_list.append(startidx)
            end_position_list.append(endidx)
        else:
            is_impossible_list.append(True)
            answer_text_list.append('')
            start_position_list.append(-1)
            end_position_list.append(-1)

    data_train = pd.DataFrame({
        'question_text': question_list,
        'context_tokens': doc_tokens_list,
        'answer_text': answer_text_list,
        'start_position': start_position_list,
        'end_position': end_position_list,
        'is_impossible': is_impossible_list})
    return data_train

def prepare_data_for_training_TC(train_dataset):
    sentence_list = []
    label_list = []
    n_sentence = len(train_dataset)
    for s in range(n_sentence):
        sentence_list.append(train_dataset[s]['sentence'])
        label = [float(train_dataset[s]['label'])]
        label_list.append(label)

    data_train = pd.DataFrame({
        'text': sentence_list,
        'label': label_list})
    return data_train

def convert_to_inputs_QA(data_train):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    max_seq_length = 1000
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    start_position_list = []
    end_position_list = []

    for example_index, example in data_train.iterrows():
        query_tokens = tokenizer.tokenize(example.question_text)
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.context_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                # Save all new toke after tokenizer in a list
                all_doc_tokens.append(sub_token)

        # Look for Star and end position after tonizer
        tok_end_position = None
        if example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        else:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.context_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for token in all_doc_tokens:
            tokens.append(token)
            segment_ids.append(0)
        # tokens.extend(all_doc_tokens)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # LOOK : tokens of the example with [CLS] at the begining and [SEP] at the end
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        while len(segment_ids) < max_seq_length:
            segment_ids.append(0)

        if not example.is_impossible:
            doc_offset = len(query_tokens) + 2
            start_position = tok_start_position + doc_offset
            end_position = tok_end_position + doc_offset
        else:
            start_position = 0
            end_position = 0

        start_position_idx = [0] * len(input_ids)
        start_position_idx[start_position] = 1
        end_position_idx = [0] * len(input_ids)
        end_position_idx[end_position] = 1

        input_ids_list.extend(input_ids)
        input_mask_list.extend(input_mask)
        segment_ids_list.extend(segment_ids)
        start_position_list.extend(start_position_idx)
        end_position_list.extend(end_position_idx)

    all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask_list, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
    all_start_positions = torch.tensor(start_position_list, dtype=torch.long)
    all_end_positions = torch.tensor(end_position_list, dtype=torch.long)
    data_feature = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                 all_start_positions, all_end_positions)
    return data_feature


def add_end_index(answers, contexts):
    new_answers = []
    question_text = []
    answer_text = []
    start_position = []
    end_position = []
    is_imposible = []
    for answer, context in tqdm(zip(answers, contexts)):
        if len(answer['text']) > 0:
            is_imposible.append(False)
            answer_text.append(answer['text'][0])
            start_position.append(answer['answer_start'][0])
            text = answer['text']
            question_text.append(text)
            startidx = answer['answer_start']
            endidx = startidx + len(text)
            if context[startidx:endidx] == text:
                start_position.append(startidx)
                end_position.append(endidx)
            else:
                for n in [1, 2]:
                    if context[startidx - n:endidx - n] == text:
                        start_position.append(startidx - n)
                        end_position.append(endidx - n)
        else:
            is_imposible.append(True)
            answer_text.append('')
            start_position.append(-1)
            end_position.append(-1)

        new_answers.append(answer)
    return new_answers


def download_dataset(dataset_name):
    data_feature = None
    data_info = dataset_information(dataset_name)
    if data_info:
        type_dataset = data_info[0]['type']
        if type_dataset == 'QA':
            data_feature = load_dataset(dataset_name)['train']
            data_feature = prepare_data_for_training_QA(data_feature)
            data_feature = convert_to_inputs_QA(data_feature)
            return data_feature
        if type_dataset =='TC':
            subset =data_info[0]['subset']
            data_feature_train = load_dataset(dataset_name,subset )['train']
            data_feature_train = prepare_data_for_training_TC(data_feature_train)
            data_feature_test = load_dataset(dataset_name, subset)['validation']
            data_feature_test  = prepare_data_for_training_TC(data_feature_test )
            return data_feature_train, data_feature_test
    else:
        print('Exception: Dataset not found')



#data_train = download_dataset('glue')
