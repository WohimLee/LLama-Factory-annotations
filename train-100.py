import os
import sys
from typing import List, Union, Dict, Any, Literal, Optional, Sequence
from datasets import DatasetDict, Dataset, IterableDataset, Features
from datasets import load_dataset
from functools import partial

from dataclasses import dataclass

from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoTokenizer
from transformers import training_args


@dataclass
class Template:
    stop_words: List[str]
    replace_eos: bool
    pass

TEMPLATES: Dict[str, Template] = {}


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('/root/datav/nlp/model/qwen/Qwen2-0.5B', use_fast=True,
            split_special_tokens=False, padding_side="right", trust_remote_code=True, cache_dir=None, revision='main', token=None)
    return {"tokenizer": tokenizer, "processor": None}


def split_dataset(dataset: Union["Dataset", "IterableDataset"], val_size=0.1, seed=42):
    dataset = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})


def convert_alpaca(examples: Dict[str, List[Any]]):
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
    for i in range(len(examples['instruction'])):
        prompt = []
        content = []
        content.append(examples['instruction'][i])
        prompt.append({"role": 'user', "content": "\n".join(content)})  # "prompt\nquery"
        response = [{"role": 'assistant', "content": examples['instruction'][i]}]
        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append("")
        outputs["tools"].append("")
        outputs["images"].append([])
    return outputs

def align_dataset(dataset: Union["Dataset", "IterableDataset"]):
    convert_func = convert_alpaca
    column_names = ['instruction', 'input', 'output']
    features = Features.from_dict(
        {
            "prompt": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "response": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "system": {"dtype": "string", "_type": "Value"},
            "tools": {"dtype": "string", "_type": "Value"},
            "images": [{"_type": "Image"}],
        }
    )
    kwargs = dict(num_proc=16, load_from_cache_file=False, desc="Converting format of dataset")
    return dataset.map(convert_func, batched=True, remove_columns=column_names, features=features, **kwargs)

def _load_single_dataset(dataset_attr='identity.json'):
    dataset = load_dataset(path='json', name=None, data_dir=None, data_files=['data/identity.json'], split='train',
            cache_dir=None, token=None, streaming=False, trust_remote_code=True)
    max_samples = min(1000, len(dataset))
    dataset = dataset.select(range(max_samples))
    return align_dataset(dataset)

def merge_dataset(all_datasets: List[Union["Dataset", "IterableDataset"]]):
    if len(all_datasets) == 1:
        return all_datasets[0]
    return

def _get_merged_dataset():
    datasets = []
    datasets.append(_load_single_dataset('identity.json'))
    
    return merge_dataset(datasets)


def _encode_supervised_example(prompt: Sequence[Dict[str, str]], response: Sequence[Dict[str, str]],):
    messages = prompt + response
    input_ids, labels = [], []
    return input_ids, labels

def preprocess_supervised_dataset(examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer",):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            print("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue
        input_ids, labels = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            # template=template,
            tokenizer=tokenizer,
            processor=None,
            cutoff_len=1024,
            train_on_prompt=False,
            mask_history=False,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
    return model_inputs

def print_supervised_dataset_example():
    
    return


def get_preprocess_and_print_func(
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
):
    
    preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer
            )
    print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    return preprocess_func, print_function

def _get_preprocessed_dataset(dataset: Optional[Union["Dataset", "IterableDataset"]],
    stage: Literal["pt", "sft", "rm", "ppo", "kto"], template: "Template",
    tokenizer: "PreTrainedTokenizer", is_eval: bool = False,
    ):
    
    preprocess_func, print_function = get_preprocess_and_print_func(
        stage, template, tokenizer)

    
    column_names = list(next(iter(dataset)).keys())
    
    kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )
    
    dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
    return


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str):
    
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})
    

    
def _get_jinja_template(template, tokenizer):
    jinja_template = ""
    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    )
    system_message = "'<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>'"
    jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"
    
    jinja_template += "{% for message in messages %}"
    jinja_template += "{% set content = message['content'] %}"
    
    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = "'<|start_header_id|>user<|end_header_id|>\n\n' + content + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'"
    jinja_template += "{{ " + user_message + " }}"
    assistant_message = "content + '<|eot_id|>'"
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    
    return jinja_template

def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", name: Optional[str] = None):
    template = TEMPLATES.get(name, None)

    tokenizer.chat_template = _get_jinja_template(template, tokenizer)
    return template

def get_dataset(stage: Literal["pt", "sft", "rm", "ppo", "kto"], tokenizer: "PreTrainedTokenizer"):
    
    template = get_template_and_fix_tokenizer(tokenizer)
    
    dataset = _get_merged_dataset()
    eval_dataset = None

    dataset = _get_preprocessed_dataset(dataset, stage, template, tokenizer, is_eval=False)
    eval_dataset = None
    
    dataset_dict = split_dataset(dataset, seed=42)
    dataset_module = {}
    dataset_module["train_dataset"], dataset_module["eval_dataset"] = dataset_dict["train"], dataset_dict["validation"]
    return dataset_module

def run_sft():
    tokenizer_module = load_tokenizer() # 加载分词器模块，根据模型参数进行初始化
    tokenizer = tokenizer_module["tokenizer"]  # 提取分词器对象
    # 获取数据集模块，包括训练、验证、测试数据集
    dataset_module = get_dataset(stage="sft", **tokenizer_module)
    pass


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"


    run_sft()