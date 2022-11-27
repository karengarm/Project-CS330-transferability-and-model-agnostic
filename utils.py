# coding=utf-8
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm

def dataset_information(dataset: str) -> str:
    dict_dataset = {
        'squad_v2': [{'type': 'QA',
                      'url': 'HuggingFace'
                      }],
        'squad': [{'type': 'QA',
                   'url': 'HuggingFace'}],
        'drop': [{'type': 'QA',
                  'url': 'HuggingFace'
                  }],
        'mlqa': [{'type': 'QA',
                  'url': 'HuggingFace'
                  }],
        'quoref': [{'type': 'QA',
                    'url': 'HuggingFace'
                    }]
    }
    if dataset in dict_dataset:
        result = dict_dataset[dataset]
    else:
        result = None
        print("Dataset not found")

    return result



def prepare_data_for_training_QA(type_dataset, train_dataset):
    train_context_list = []
    train_question_list = []
    train_answer_list = []
    n_paragraph = len(train_dataset)
    for p in range(n_paragraph):
        train_context_list.append(train_dataset[p]['context'])
        train_question_list.append(train_dataset[p]['question'])
        train_answer_list.append(train_dataset[p]['answers'])
    data_train = pd.DataFrame({
        'context': train_context_list,
        'question': train_question_list,
        'answer': train_answer_list})
    return data_train


def add_end_index(answers, contexts):
    new_answers = []
    for answer, context in tqdm(zip(answers, contexts)):
        if len(answer['text'])>0:
            answer['text'] = answer['text'][0]
            answer['answer_start'] = answer['answer_start'][0]
            text = answer['text']
            startidx = answer['answer_start']
            endidx = startidx + len(text)
            if context[startidx:endidx] == text:
                answer['answer_end'] = endidx
            else:
                for n in [1, 2]:
                    if context[startidx-n:endidx-n] == text:
                        answer['answer_start'] = startidx - n
                        answer['answer_end'] = endidx - n
        else:
            answer['text'] = ''
            answer['answer_start'] = 0
            answer['answer_end'] = 0

        new_answers.append(answer)
    return new_answers

def download_dataset(dataset_name):
    data_train = None
    data_info = dataset_information(dataset_name)
    if data_info:
        type_dataset = data_info[0]['type']
        if type_dataset == 'QA':
            dataset_json = load_dataset(dataset_name)['train']
            data_train = prepare_data_for_training_QA(type_dataset, dataset_json)
            data_train['answer'] = add_end_index(data_train['answer'], data_train['context'])
    else:
        print('Exception: Dataset not found')
    return data_train



