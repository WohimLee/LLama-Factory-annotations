
import re
import os
import sys
import json
from typing import List, Union, Dict, Any, Literal, Optional, Sequence, Tuple, Set
from datasets import DatasetDict, Dataset, IterableDataset, Features
from datasets import load_dataset
from functools import partial

from dataclasses import dataclass, field

from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoTokenizer
from transformers import training_args
from transformers import AutoConfig, AutoModelForCausalLM

from abc import ABC, abstractmethod


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]

IGNORE_INDEX = -100

DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具{tool_text}"
)


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[Literal["default", "glm4"]] = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS: ...

    def extract(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        raise NotImplementedError


@dataclass
class ToolUtils(ABC):
    @staticmethod
    @abstractmethod
    def get_function_slots() -> SLOTS: ...

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str: ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]: ...



@dataclass
class Template:
    stop_words: List[str]
    replace_eos: bool
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_prefix: "Formatter"
    format_separator: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    
    
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
    ) -> List[List[int]]:

        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == 'user':
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == 'assistant':
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == 'observation':
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == 'function':
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages
    
    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:

        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids
    
TEMPLATES: Dict[str, Template] = {}


class DefaultToolUtils(ToolUtils):
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["Action: {{name}}\nAction Input: {{arguments}}\n"]

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: List[Tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append((tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class GLM4ToolUtils(ToolUtils):
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["{{name}}\n{{arguments}}"]

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input)
        except json.JSONDecodeError:
            return content

        return [(tool_name, json.dumps(arguments, ensure_ascii=False))]


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError("Expected a string, got {}".format(value))

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements
    



@dataclass
class FunctionFormatter(Formatter):
    def __post_init__(self):
        if self.tool_format == "default":
            self.slots = DefaultToolUtils.get_function_slots() + self.slots
        elif self.tool_format == "glm4":
            self.slots = GLM4ToolUtils.get_function_slots() + self.slots
        else:
            raise NotImplementedError("Tool format {} was not found.".format(self.tool_format))

    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        functions: List[Tuple[str, str]] = []
        try:
            tool_calls = json.loads(content)
            if not isinstance(tool_calls, list):  # parallel function call
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                functions.append((tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False)))

        except json.JSONDecodeError:
            functions = []

        elements = []
        for name, arguments in functions:
            for slot in self.slots:
                if isinstance(slot, str):
                    slot = slot.replace("{{name}}", name).replace("{{arguments}}", arguments)
                    elements.append(slot)
                elif isinstance(slot, (dict, set)):
                    elements.append(slot)
                else:
                    raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements
    

@dataclass
class ToolFormatter(Formatter):
    def __post_init__(self):
        if self.tool_format == "default":
            self._tool_formatter = DefaultToolUtils.tool_formatter
            self._tool_extractor = DefaultToolUtils.tool_extractor
        elif self.tool_format == "glm4":
            self._tool_formatter = GLM4ToolUtils.tool_formatter
            self._tool_extractor = GLM4ToolUtils.tool_extractor
        else:
            raise NotImplementedError("Tool format {} was not found.".format(self.tool_format))

    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            return [self._tool_formatter(tools) if len(tools) != 0 else ""]
        except json.JSONDecodeError:
            return [""]

    def extract(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        return self._tool_extractor(content)


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class Llama2Template(Template):
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: str,
    ) -> List[List[int]]:

        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == 'user':
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == 'assistant':
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == 'observation':
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == 'function':
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Sequence[str] = [],
    image_token: str = "<image>",
    efficient_eos: bool = False,
    replace_eos: bool = False,
) -> None:

    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Llama2Template if name.startswith("llama2") else Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=eos_slots, tool_format="default")
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        stop_words=stop_words,
        replace_eos=replace_eos,
    )


_register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('/root/datav/nlp/model/qwen/Qwen2-0.5B', use_fast=True,
            split_special_tokens=False, padding_side="right", trust_remote_code=True, cache_dir=None, revision='main', token=None)
    return {"tokenizer": tokenizer}


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

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:

    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len

def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    cutoff_len: int,
    ):
    
    messages = prompt + response
    input_ids, labels = [], []
    
    encoded_pairs = template.encode_multiturn(tokenizer, messages)
    total_length = 0
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len
        
        source_label = [IGNORE_INDEX] * source_len
        
        target_label = target_ids
        
        input_ids += source_ids + target_ids
        labels += source_label + target_label
        
    return input_ids, labels

def preprocess_supervised_dataset(examples: Dict[str, List[Any]], template: "Template", tokenizer: "PreTrainedTokenizer",):
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
            template=template,
            tokenizer=tokenizer,
            cutoff_len=1024,
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
            num_proc=16,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    
    dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
    return dataset


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



def load_model(
    tokenizer: "PreTrainedTokenizer"
) -> "PreTrainedModel":

    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)


    return model

def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", name: Optional[str] = None):
    template = TEMPLATES.get('llama3', None)

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
    
    # 根据分词器和其他参数加载预训练模型，准备进行微调
    model = load_model(tokenizer)
    
    
    pass


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"


    run_sft()