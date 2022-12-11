import argparse
import copy
import os
import qa_utils
import json
import numpy as np
import torch
import qa_finetuning as ft
from transformers import GPT2Tokenizer, GPT2LMHeadModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = 5
NUM_TEST_TASKS = 600


class MAML:
    """Class for training a Reptilian MAML model on NLP tasks."""

    def __init__(
            self,
            model_name,
            datasets,
            num_inner_steps,
            inner_lr,
            outer_lr,
            log_dir,
            num_tasks,
            debug
    ):
        self._num_inner_steps = num_inner_steps
        self._model_name = model_name
        self._datasets = datasets
        self._inner_lrs = inner_lr
        self._outer_lr = outer_lr
        self._outer_lr_0 = outer_lr
        self._num_tasks = num_tasks
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._seed = 1
        self._start_train_step = 0
        self._tracked_state_key = 'transformer.wte.weight'
        self._tracked_state = None
        self._debug = debug

    def _create_dataset(self, dataset, n_examples):
        examples = []
        np.random.seed(self._seed)
        ids = np.random.choice(len(dataset), n_examples, replace=False)
        for i in range(len(ids)):
            examples.append(dataset[int(ids[i])])
        self._seed += 1
        return examples

    def _meta_update(self, model, inner_params):
        st = model.state_dict()
        for key in st:
            if st[key].dtype != torch.uint8 and st[key].dtype != torch.int64:
                # Update of the meta parameters
                st[key] += self._outer_lr * (
                        inner_params[key] / self._num_tasks - st[key])
        model.load_state_dict(st)

    def _update_params(self, model, params_dict={}):
        if params_dict == {}:
            for key in model.state_dict():
                params_dict[key] = torch.zeros_like(model.state_dict()[key])
            return params_dict
        else:
            for key in model.state_dict():
                params_dict[key] += model.state_dict()[key]

    def _outer_step(self, tasks, model, tokenizer):
        outer_model = copy.deepcopy(model)
        inner_params = self._update_params(outer_model, params_dict={})
        for task in tasks:
            if len(task) == 1:
                train_dataset = self._create_dataset(self._datasets[task[0]], self._num_inner_steps)
            elif len(task) == 2:
                train_dataset = self._create_dataset(self._datasets[task[1]], self._num_inner_steps)
            data_feature = qa_utils.prepare_data_for_training_QA(train_dataset)
            data_feature = qa_utils.convert_to_inputs_QA(data_feature)
            data_collator = ft.load_data_collator(tokenizer)
            inner_model = ft.train(outer_model, tokenizer, data_collator, data_feature, max_steps=self._num_inner_steps)
            self._update_params(inner_model, inner_params)
        return inner_params

    def train(self, task_list, max_steps=1000):
        tokenizer = GPT2Tokenizer.from_pretrained(self._model_name)
        model = GPT2LMHeadModel.from_pretrained(self._model_name)
        model.to(DEVICE)
        for i_step in range(max_steps):
            inner_params = self._outer_step(task_list, model, tokenizer)
            self._meta_update(model, inner_params)
            self._outer_lr = self._outer_lr_0 * (1 - i_step / max_steps)
        return model, tokenizer

    def _save(self, model):
        model.save_pretrained(f'{self._log_dir}/model')
        print('Saved model.')


def test_infer(model, tok):
    with torch.inference_mode():
        while True:
            input_ = input("Please insert a prompt: ")
            x_ = tok(input_, return_tensors='pt', padding=True).to(DEVICE)
            out = model(**x_)
            prediction = tok.decode(torch.argmax(out.logits, dim=-1)[0][-1])
            print(prediction)


def get_qa_datasets(task_names):
    from datasets import load_dataset
    datasets = {}
    for key in task_names:
        if len(key) == 1:
            print(key)
            datasets[key[0]] = load_dataset(key[0], split='train')
        if len(key) == 2:
            datasets[key[1]] = load_dataset(key[0], key[1], split='validation')
    return datasets


def create_temp_dataset(dataset, n_examples):
    # Create a dictionary of all the QA examples
    examples = []
    # Get shuffled ids from the dataset with length n_examples
    ids = np.random.choice(len(dataset), n_examples, replace=False)
    for i in range(len(ids)):
        examples.append(dataset[int(ids[i])])
    # Export datasets to json
    with open('src/temp_dataset.json', 'w') as fp:
        json.dump({'data': examples}, fp)


def main(args):
    tasks = args.tasks.split(',')
    tasks_list = [[task] for task in tasks if ';' not in task]
    tasks_list.extend([task.split(';') for task in tasks if ';' in task])
    print("tasks_list: ", tasks_list)
    log_dir = f'.{args.log_dir}/qa_tasks_{args.out}'
    print(f'log_dir: {log_dir}')
    datasets = get_qa_datasets(tasks_list)

    maml = MAML(
        args.model_name,
        datasets,
        args.num_inner_steps,
        args.inner_lr,
        args.outer_lr,
        log_dir,
        len(tasks_list),
        args.debug
    )

    model, _ = maml.train(
        tasks_list,
        args.num_train_iterations,
    )
    maml._save(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--tasks', type=str, default='squad,squad_v2,duorc;SelfRC')
    parser.add_argument('--out', type=str, default='temp',
                        help='Name of the output directory containing the trained MAML model parameters')
    parser.add_argument('--log_dir', type=str, default='src/logs',
                        help='directory to save to or load from')
    parser.add_argument('--num_examples', type=int, default=15,
                        help='number of examples per task in a task')
    parser.add_argument('--num_inner_steps', type=int, default=10,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=2e-5,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--outer_lr', type=float, default=0.5,
                        help='outer-loop learning rate')
    parser.add_argument('--num_train_iterations', type=int, default=500,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument("--model_name", default='gpt2', type=str,
                        help="Pretrained model name or directory")

    main_args = parser.parse_args()
    main(main_args)
    print("Done!")