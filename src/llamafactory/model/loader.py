# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer. 加载预训练的分词器

    Note: including inplace operation of model_args. 此函数会对 model_args 进行原地操作
    """
    # 初始化分词器初始化的关键字参数
    init_kwargs = _get_init_kwargs(model_args)
    try:
        # 尝试使用 model_args 中指定的参数加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,          # 预训练模型或分词器的路径
            use_fast=model_args.use_fast_tokenizer, # 是否使用快速分词器
            split_special_tokens=model_args.split_special_tokens, # 是否拆分特殊标记
            padding_side="right",                   # 将填充位置设置为右侧
            **init_kwargs,                          # 其他初始化参数
        )
    except ValueError:  # 如果发生错误，重试并强制使用快速分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,          # 重试时使用相同的模型路径
            use_fast=True,                          # 强制使用快速分词器
            padding_side="right",                   # 将填充位置设置为右侧
            **init_kwargs,                          # 其他初始化参数
        )

    # 如果 model_args 中提供了新的特殊标记，添加这些标记
    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens), # 添加新的特殊标记
            replace_additional_special_tokens=False,  # 不替换现有的特殊标记
        )
        # 记录添加的新特殊标记
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        # 如果添加了新的标记且未设置 resize_vocab，则启用词汇表调整
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    # 如有必要，对分词器进行额外的修补
    patch_tokenizer(tokenizer)

    # 如果需要视觉输入，加载相应的处理器
    if model_args.visual_inputs:
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
            setattr(processor, "tokenizer", tokenizer)  # 将分词器附加到处理器上
        except Exception:   # 如果处理器无法加载，则抛出错误
            raise ValueError(
                "This multimodal LLM is not supported.\n"
                "Download LLaVA-1.5 models from: https://huggingface.co/llava-hf\n"
                "Download Yi-VL models from: https://huggingface.co/BUAADreamer"
            )
    else:
        processor = None    # 如果不需要视觉输入，则将处理器设置为 None

    # 返回分词器和处理器（如果有）的字典
    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",         # 分词器，用于处理模型输入的文本
    model_args: "ModelArguments",             # 模型参数，包括模型的名称、路径等
    finetuning_args: "FinetuningArguments",   # 微调参数，用于控制模型微调的细节
    is_trainable: bool = False,               # 是否使模型可训练，默认为不可训练
    add_valuehead: bool = False,              # 是否为模型添加 value head，用于特定任务
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    
    # 初始化模型加载的关键字参数
    init_kwargs = _get_init_kwargs(model_args)
    
    # 加载模型配置
    config = load_config(model_args)
    
    # 修补配置，调整配置以适应分词器、模型参数和是否可训练的选项
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None         # 初始化模型变量为 None
    lazy_load = False    # 初始化延迟加载标志为 False
    
    # 如果使用 Unsloth 加载模型
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True  # 如果指定了适配器路径，启用延迟加载
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)  # 加载 Unsloth 预训练模型


    # 如果模型尚未加载且不启用延迟加载
    if model is None and not lazy_load:
        # 添加配置到初始化参数中
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        # 根据不同的模型类型加载对应的模型
        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)  # 加载混合深度的预训练模型
        elif model_args.visual_inputs:
            model = AutoModelForVision2Seq.from_pretrained(**init_kwargs)  # 加载视觉到序列的预训练模型
        elif model_args.train_from_scratch:
            model = AutoModelForCausalLM.from_config(config)  # 从配置开始训练模型
        else:
            model = AutoModelForCausalLM.from_pretrained(**init_kwargs)  # 从预训练模型加载

        # 如果需要，将模型转换为混合深度模型
        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)


    # 如果不启用延迟加载
    if not lazy_load:
        # 修补模型以适应分词器、模型参数、是否可训练和是否添加 value head
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        
        # 注册自动类以使模型、配置和分词器能够自动序列化和反序列化
        register_autoclass(config, model, tokenizer)

    # 初始化适配器，适应微调参数
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # 如果需要添加 value head
    if add_valuehead:
        # 使用 value head 加载模型
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        
        # 修补 value head 模型
        patch_valuehead_model(model)

        # 设置 value head 路径
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        # 加载 value head 参数
        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)  # 将 value head 参数加载到模型中
            logger.info("从检查点加载了 valuehead: {}".format(vhead_path))

    # 如果模型不可训练
    if not is_trainable:
        model.requires_grad_(False)  # 禁用所有参数的梯度计算
        for param in model.parameters():
            # 如果参数的数据类型为 float32 且计算数据类型不同，则转换数据类型
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()  # 将模型设置为评估模式
    else:
        model.train()  # 将模型设置为训练模式

    # 计算可训练参数和所有参数的数量
    trainable_params, all_param = count_parameters(model)
    
    # 记录模型参数统计信息
    if is_trainable:
        param_stats = "可训练参数: {:,} || 所有参数: {:,} || 可训练%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "所有参数: {:,}".format(all_param)

    logger.info(param_stats)  # 打印参数统计信息

    # 如果需要打印参数状态，输出每个参数的信息
    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "名称: {}, 数据类型: {}, 设备: {}, 可训练: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model  # 返回加载的模型
