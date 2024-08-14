import logging
import os
import sys
import json

import math
import numpy as np
import matplotlib.pyplot as plt
from types import MethodType
from typing import Any, Dict, Optional, Tuple
from typing import TYPE_CHECKING, List, Optional
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass

import torch
import transformers
from torch.utils.data import Dataset
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers import ProcessorMixin
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import PredictionOutput
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import get_parameter_names
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available
from transformers.modeling_utils import is_fsdp_enabled
from transformers.utils.versions import require_version
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers import Trainer
from transformers import Seq2SeqTrainer
from transformers.dynamic_module_utils import get_relative_imports

from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model

from transformers.optimization import get_scheduler

from transformers import PreTrainedModel, Seq2SeqTrainingArguments
from trl import AutoModelForCausalLMWithValueHead

from transformers.trainer import TRAINER_STATE_NAME
from transformers import TrainerControl, TrainerState, TrainingArguments

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import EvalPrediction
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers import PretrainedConfig, PreTrainedModel

from src.llamafactory.extras.constants import CHECKPOINT_NAMES
from src.llamafactory.extras.logging import get_logger
from src.llamafactory.extras.misc import check_dependencies, get_current_device
from src.llamafactory.hparams.data_args import DataArguments
from src.llamafactory.hparams.evaluation_args import EvaluationArguments
from src.llamafactory.hparams.finetuning_args import FinetuningArguments
from src.llamafactory.hparams.generating_args import GeneratingArguments
from src.llamafactory.hparams.model_args import ModelArguments

from transformers.utils import is_jieba_available, is_nltk_available
import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version

def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def is_rouge_available():
    return _is_package_available("rouge_chinese")

def is_galore_available():
    return _is_package_available("galore_torch")



if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge

if is_galore_available():
    from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit


from transformers import Seq2SeqTrainingArguments, TrainerCallback

from src.llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


IGNORE_INDEX = -100


logger = get_logger(__name__)


def _parse_train_args(args: Optional[Dict[str, Any]] = None): #  -> _TRAIN_CLS
    parser = HfArgumentParser(_TRAIN_ARGS)
    parser.add_argument('--local-rank', type=int, default=0, help="Local rank for distributed training")
    return _parse_args(parser, args)



def get_train_args(args: Optional[Dict[str, Any]] = None): #  -> _TRAIN_CLS
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage != "sft" and data_args.neat_packing:
        raise ValueError("`neat_packing` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo" and not training_args.do_train:
        raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

    if finetuning_args.stage == "ppo" and model_args.shift_attn:
        raise ValueError("PPO training is incompatible with S^2-Attn.")

    if finetuning_args.stage == "ppo" and finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
        raise ValueError("Unsloth does not support lora reward model.")

    if (
        finetuning_args.stage == "ppo"
        and training_args.report_to
        and training_args.report_to[0] not in ["wandb", "tensorboard"]
    ):
        raise ValueError("PPO only accepts wandb or tensorboard logger.")

    if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    if (training_args.do_eval or training_args.do_predict) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please specify dataset for evaluation.")

    if training_args.predict_with_generate and data_args.eval_dataset is None:
        raise ValueError("Cannot use `predict_with_generate` if `eval_dataset` is None.")

    if training_args.predict_with_generate and finetuning_args.compute_accuracy:
        raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA in DeepSpeed ZeRO-3.")

    if finetuning_args.pure_bf16:
        if not is_torch_bf16_gpu_available():
            raise ValueError("This device does not support `pure_bf16`.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")

    if (
        finetuning_args.use_galore
        and finetuning_args.galore_layerwise
        and training_args.parallel_mode == ParallelMode.DISTRIBUTED
    ):
        raise ValueError("Distributed training does not support layer-wise GaLore.")

    if finetuning_args.use_badam and training_args.parallel_mode == ParallelMode.DISTRIBUTED:
        if finetuning_args.badam_mode == "ratio":
            raise ValueError("Radio-based BAdam does not yet support distributed training, use layer-wise BAdam.")
        elif not is_deepspeed_zero3_enabled():
            raise ValueError("Layer-wise BAdam only supports DeepSpeed ZeRO-3 training.")

    if finetuning_args.use_galore and training_args.deepspeed is not None:
        raise ValueError("GaLore is incompatible with DeepSpeed yet.")

    if model_args.infer_backend == "vllm":
        raise ValueError("vLLM backend is only available for API, CLI and Web.")

    if model_args.visual_inputs and data_args.packing:
        raise ValueError("Cannot use packing in MLLM fine-tuning.")

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if data_args.neat_packing and not data_args.packing:
        logger.warning("`neat_packing` requires `packing` is True. Change `packing` to True.")
        data_args.packing = True

    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.quantization_bit is None
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning("Remember to add embedding layers to `additional_target` to make the added tokens trainable.")

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if training_args.do_train and finetuning_args.use_galore and not finetuning_args.pure_bf16:
        logger.warning("Using GaLore with mixed precision training may significantly increases GPU memory usage.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info("Resuming training from {}.".format(training_args.resume_from_checkpoint))
            logger.info("Change `output_dir` or use `overwrite_output_dir` to avoid.")

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning(
            "Add {} to `adapter_name_or_path` to resume training from checkpoint.".format(
                training_args.resume_from_checkpoint
            )
        )

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # Log on each process the small summary
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode == ParallelMode.DISTRIBUTED,
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args




# 从传入的参数字典中提取各类训练相关的参数
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)




def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)

    if model_args.visual_inputs:
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
            setattr(processor, "tokenizer", tokenizer)
        except Exception:
            raise ValueError(
                "This multimodal LLM is not supported.\n"
                "Download LLaVA-1.5 models from: https://huggingface.co/llava-hf\n"
                "Download Yi-VL models from: https://huggingface.co/BUAADreamer"
            )
    else:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}



def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template, data_args.tool_format)
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(data_args.tokenized_path))

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]
            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))

            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]
        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module


def count_parameters(model: "torch.nn.Module") -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param



def load_valuehead_params(path_or_repo_id: str, model_args: "ModelArguments") -> Dict[str, torch.Tensor]:
    r"""
    Loads value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = cached_file(filename=V_HEAD_SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = cached_file(filename=V_HEAD_WEIGHTS_NAME, **kwargs)
        return torch.load(vhead_file, map_location="cpu")
    except Exception as err:
        err_text = str(err)

    logger.info("Provided path ({}) does not contain value head weights: {}.".format(path_or_repo_id, err_text))
    logger.info("Ignore the above message if you are not resuming the training of a value head model.")
    return None



def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantized models can only be used for the LoRA tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 and not fsdp (zero3 or fsdp already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and (is_deepspeed_zero3_enabled() or is_fsdp_enabled()):
        logger.info("ZeRO3 / FSDP detected, remaining trainable params in float32.")
    else:
        logger.info("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError("Unknown finetuning type: {}.".format(finetuning_args.finetuning_type))

    return model



def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()



def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    if model_args.visual_inputs:
        autocast_projector_dtype(model, model_args)

    if is_trainable:
        prepare_model_for_training(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning("Cannot properly tag the model.")



def convert_pretrained_model_to_mod(
    model: "PreTrainedModel", config: "PretrainedConfig", model_args: "ModelArguments"
) -> "PreTrainedModel":
    from MoD import apply_mod_to_hf

    if getattr(config, "model_type", None) not in MOD_SUPPORTED_MODELS:
        raise ValueError("Current model is not supported by mixture-of-depth.")

    model = apply_mod_to_hf(model)
    model = model.to(model_args.compute_dtype)
    return model



def load_mod_pretrained_model(**init_kwargs) -> "PreTrainedModel":
    from MoD import AutoMoDModelForCausalLM

    return AutoMoDModelForCausalLM.from_pretrained(**init_kwargs)


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments"
) -> Optional["PreTrainedModel"]:
    r"""
    Optionally loads pretrained model with unsloth. Used in training.
    """
    from unsloth import FastLanguageModel

    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args)
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model = None
        model_args.use_unsloth = False

    return model


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    if is_torch_npu_available():
        use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
        torch.npu.set_compile_mode(jit_compile=use_jit_compile)

    configure_attn_implementation(config, model_args, is_trainable)
    configure_rope(config, model_args, is_trainable)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(config, model_args, is_trainable)

    if model_args.use_cache and not is_trainable:
        setattr(config, "use_cache", True)
        logger.info("Using KV cache for faster generation.")

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "qwen2" and is_trainable and model_args.flash_attn == "fa2":
        setattr(config, "use_cache", False)  # qwen2 does not support use_cache when using flash attn

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # cast data type of the model if:
    # 1. not deepspeed zero3 and not fsdp (keep zero3 or fsdp in float32)
    # 2. quantization_bit is not None (qlora)
    if (not is_deepspeed_zero3_enabled() and not is_fsdp_enabled()) or model_args.quantization_bit is not None:
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"]:  # device map requires low_cpu_mem_usage=True
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def use_modelscope() -> bool:
    return os.environ.get("USE_MODELSCOPE_HUB", "0").lower() in ["true", "1"]


def try_download_model_from_ms(model_args: "ModelArguments") -> str:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    try:
        from modelscope import snapshot_download

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        return snapshot_download(model_args.model_name_or_path, revision=revision, cache_dir=model_args.cache_dir)
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")


def skip_check_imports() -> None:
    if os.environ.get("FORCE_CHECK_IMPORTS", "0").lower() not in ["true", "1"]:
        transformers.dynamic_module_utils.check_imports = get_relative_imports


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


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        elif model_args.visual_inputs:
            model = AutoModelForVision2Seq.from_pretrained(**init_kwargs)
        elif model_args.train_from_scratch:
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


@dataclass
class SFTDataCollatorWith4DAttentionMask(DataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        return features


@dataclass
class ComputeSimilarity:
    r"""
    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


def numpify(inputs: Union["NDArray", "torch.Tensor"]) -> "NDArray":
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu()
        if inputs.dtype == torch.bfloat16:  # numpy does not support bfloat16 until 1.21.4
            inputs = inputs.to(torch.float32)

        inputs = inputs.numpy()

    return inputs


@dataclass
class ComputeAccuracy:
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor: "ProcessorMixin") -> None:
        r"""
        Initializes a callback for saving the processor.
        """
        self.processor = processor

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if args.should_save:
            getattr(self.processor, "image_processor").save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""
    Initializes a callback for converting the PiSSA adapter to a normal one.
    """

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info("Initial PiSSA adapter will be saved at: {}.".format(pissa_init_dir))
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info("Converted PiSSA adapter will be saved at: {}.".format(pissa_convert_dir))
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir, safe_serialization=args.save_safetensors, convert_pissa_to_lora=pissa_init_dir
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                model.delete_adapter("pissa_init")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


def create_custom_scheduler(
    training_args: "Seq2SeqTrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict: Dict["torch.nn.Parameter", "torch.optim.lr_scheduler.LRScheduler"] = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer_dict[param],
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
            )

        def scheduler_hook(param: "torch.nn.Parameter"):
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)


class DummyOptimizer(torch.optim.Optimizer):
    r"""
    A dummy optimizer used for the GaLore algorithm.
    """

    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[Dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass


def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    forbidden_modules = {"lm_head"}

    if model.config.model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model.config.model_type == "internlm2":
        forbidden_modules.add("output")
    elif model.config.model_type in ["llava", "paligemma"]:
        forbidden_modules.add("multi_modal_projector")

    if freeze_vision_tower:
        forbidden_modules.add("vision_tower")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def _create_galore_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    if len(finetuning_args.galore_target) == 1 and finetuning_args.galore_target[0] == "all":
        galore_targets = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
    else:
        galore_targets = finetuning_args.galore_target

    galore_params: List["torch.nn.Parameter"] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in galore_targets):
            for param in module.parameters():
                if param.requires_grad and len(param.shape) > 1:
                    galore_params.append(param)

    galore_kwargs = {
        "rank": finetuning_args.galore_rank,
        "update_proj_gap": finetuning_args.galore_update_interval,
        "scale": finetuning_args.galore_scale,
        "proj_type": finetuning_args.galore_proj_type,
    }

    id_galore_params = {id(param) for param in galore_params}
    decay_params, nodecay_params = [], []  # they are non-galore parameters
    trainable_params: List["torch.nn.Parameter"] = []  # galore_params + decay_params + nodecay_params
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            if id(param) not in id_galore_params:
                if name in decay_param_names:
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)

    _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

    if training_args.optim == "adamw_torch":
        optim_class = GaLoreAdamW
    elif training_args.optim in ["adamw_bnb_8bit", "adamw_8bit", "paged_adamw_8bit"]:
        optim_class = GaLoreAdamW8bit
    elif training_args.optim == "adafactor":
        optim_class = GaLoreAdafactor
    else:
        raise NotImplementedError("Unknow optim: {}".format(training_args.optim))

    if finetuning_args.galore_layerwise:
        if training_args.gradient_accumulation_steps != 1:
            raise ValueError("Per-layer GaLore does not support gradient accumulation.")

        optimizer_dict: Dict["torch.Tensor", "torch.optim.Optimizer"] = {}
        for param in nodecay_params:
            param_groups = [dict(params=[param], weight_decay=0.0)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in decay_params:
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in galore_params:  # galore params have weight decay
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay, **galore_kwargs)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        def optimizer_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                optimizer_dict[param].step()
                optimizer_dict[param].zero_grad()

        for param in trainable_params:
            param.register_post_accumulate_grad_hook(optimizer_hook)

        optimizer = DummyOptimizer(lr=training_args.learning_rate, optimizer_dict=optimizer_dict)
    else:
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0),
            dict(params=decay_params, weight_decay=training_args.weight_decay),
            dict(params=galore_params, weight_decay=training_args.weight_decay, **galore_kwargs),
        ]
        optimizer = optim_class(param_groups, **optim_kwargs)

    logger.info("Using GaLore optimizer, may cause hanging at the start of training, wait patiently.")
    return optimizer


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    embedding_lr = finetuning_args.loraplus_lr_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info("Using LoRA+ optimizer with loraplus lr ratio {:.2f}.".format(finetuning_args.loraplus_lr_ratio))
    return optimizer


def _get_decay_parameter_names(model: "PreTrainedModel") -> List[str]:
    r"""
    Returns a list of names of parameters with weight decay. (weights in non-layernorm layers)
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def _create_badam_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    decay_params, nodecay_params = [], []
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in decay_param_names:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=nodecay_params, weight_decay=0.0),
        dict(params=decay_params, weight_decay=training_args.weight_decay),
    ]

    if finetuning_args.badam_mode == "layer":
        from badam import BlockOptimizer

        base_optimizer = optim_class(param_groups, **optim_kwargs)
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=list(model.named_parameters()),
            block_prefix_list=None,
            switch_block_every=finetuning_args.badam_switch_interval,
            start_block=finetuning_args.badam_start_block,
            switch_mode=finetuning_args.badam_switch_mode,
            verbose=finetuning_args.badam_verbose,
            ds_zero3_enabled=is_deepspeed_zero3_enabled(),
        )
        logger.info(
            f"Using BAdam optimizer with layer-wise update, switch mode is {finetuning_args.badam_switch_mode}, "
            f"switch block every {finetuning_args.badam_switch_interval} steps, "
            f"default start block is {finetuning_args.badam_start_block}"
        )

    elif finetuning_args.badam_mode == "ratio":
        from badam import BlockOptimizerRatio

        assert finetuning_args.badam_update_ratio > 1e-6
        optimizer = BlockOptimizerRatio(
            param_groups=param_groups,
            named_parameters_list=list(model.named_parameters()),
            update_ratio=finetuning_args.badam_update_ratio,
            mask_mode=finetuning_args.badam_mask_mode,
            verbose=finetuning_args.badam_verbose,
            include_embedding=False,
            **optim_kwargs,
        )
        logger.info(
            f"Using BAdam optimizer with ratio-based update, update ratio is {finetuning_args.badam_update_ratio}, "
            f"mask mode is {finetuning_args.badam_mask_mode}"
        )

    return optimizer


def create_custom_optimzer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:
    if finetuning_args.use_galore:
        return _create_galore_optimizer(model, training_args, finetuning_args)

    if finetuning_args.loraplus_lr_ratio is not None:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_badam:
        return _create_badam_optimizer(model, training_args, finetuning_args)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: List[str] = ["loss"]) -> None:
    r"""
    Plots loss curves and saves the image.
    """
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = data_args.dataset

    if model_args.use_unsloth:
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]

    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub


def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def run_sft(
    model_args: "ModelArguments",               # 模型相关的参数，包含模型类型、路径等信息
    data_args: "DataArguments",                 # 数据处理相关的参数，例如数据集路径、预处理方法等
    training_args: "Seq2SeqTrainingArguments",  # 序列到序列训练的参数，如批次大小、学习率等
    finetuning_args: "FinetuningArguments",     # 微调相关的参数，决定如何对模型进行微调
    generating_args: "GeneratingArguments",     # 生成相关的参数，用于控制模型生成文本的行为
    callbacks: Optional[List["TrainerCallback"]] = None,  # 可选的回调函数列表，用于在训练过程中执行特定操作
):
    # 加载分词器模块，根据模型参数进行初始化
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]  # 提取分词器对象

    # 获取数据集模块，包括训练、验证、测试数据集
    dataset_module = get_dataset(
        model_args, data_args, training_args, stage="sft", **tokenizer_module
    )

    # 根据分词器和其他参数加载预训练模型，准备进行微调
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 如果模型是量化后的，并且不执行训练操作，则设置特定属性以兼容预测过程
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # 这里是一个临时的兼容性调整

    # 初始化数据整理器，处理输入数据的格式和注意力掩码
    data_collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,                                        # 使用已经加载的分词器
        pad_to_multiple_of=8 if training_args.do_train else None,   # 训练时的填充长度要求
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,  # 标签填充
        block_diag_attn=model_args.block_diag_attn,                 # 是否使用块对角注意力机制
        attn_implementation=getattr(model.config, "_attn_implementation", None),  # 注意力机制的具体实现
        compute_dtype=model_args.compute_dtype,                     # 计算数据类型，如float32、float16等
    )

    # 覆盖 Seq2SeqTrainer 的解码参数
    # 如果 generation_max_length 没有被设置，则使用 data_args.cutoff_len 作为生成文本的最大长度
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len

    # 设置生成时使用的束搜索数量（num_beams），如果 data_args.eval_num_beams 没有被设置，则保持原值
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # 如果模型使用视觉输入，则不移除未使用的列，否则保持 remove_unused_columns 的原值
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # 评估指标模块的初始化
    metric_module = {}

    # 如果在预测时使用生成模式，则添加计算相似度的指标
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    # 如果微调参数要求计算准确率，则添加计算准确率的指标，并设置 logits 的预处理函数
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # 初始化自定义的 Seq2SeqTrainer 训练器
    trainer = CustomSeq2SeqTrainer(
        model=model,                        # 要训练的模型
        args=training_args,                 # 训练参数，包括上面设置的解码参数
        finetuning_args=finetuning_args,    # 微调参数
        data_collator=data_collator,        # 数据整理器，用于处理输入数据
        callbacks=callbacks,                # 回调函数列表，用于在训练过程中执行特定操作
        **dataset_module,                   # 解包数据集模块，传递给训练器
        **tokenizer_module,                 # 解包分词器模块，传递给训练器
        **metric_module,                    # 解包评估指标模块，传递给训练器
    )

    # 为 `model.generate` 函数准备关键字参数
    gen_kwargs = generating_args.to_dict()                      # 将生成参数转换为字典形式，以便后续传递给生成函数
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids  # 设置结束标记（eos token）的ID，包含标准结束标记及额外的特殊标记
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id         # 设置填充标记（pad token）的ID，确保生成的序列长度一致
    gen_kwargs["logits_processor"] = get_logits_processor()     # 获取并设置 logits 处理器，用于在生成过程中对 logits 进行处理


    # Training 训练过程
    if training_args.do_train:                                  # 如果设置了进行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)  # 开始训练，可能从检查点恢复
        trainer.save_model()                                    # 保存训练后的模型
        trainer.log_metrics("train", train_result.metrics)      # 记录训练过程中的指标（如损失、准确率等）
        trainer.save_metrics("train", train_result.metrics)     # 保存这些训练指标到磁盘
        trainer.save_state()                                    # 保存训练状态（如优化器状态、随机种子等），便于后续继续训练
        # 如果当前进程是主要进程（在分布式训练中），并且要求绘制损失曲线
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])  # 绘制并保存损失曲线图

    # 如果设置了使用生成模式进行预测，则调整分词器的填充方式
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # 使用左填充，以适应生成的需求

    # Evaluation 评估过程
    if training_args.do_eval:  # 如果设置了进行评估
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)  # 执行评估，并根据生成的关键字参数进行解码
        # 如果使用生成模式进行预测，eval_loss 可能会错误，因此将其移除
        if training_args.predict_with_generate:
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)    # 记录评估过程中的指标
        trainer.save_metrics("eval", metrics)   # 保存这些评估指标到磁盘
        
    # Predict 预测过程
    if training_args.do_predict:  # 如果设置了进行预测
        # 执行预测，并根据生成的关键字参数进行解码
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        # 如果使用生成模式进行预测，predict_loss 可能会错误，因此将其移除
        if training_args.predict_with_generate:
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)     # 记录预测过程中的指标
        trainer.save_metrics("predict", predict_results.metrics)    # 保存这些预测指标到磁盘
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)  # 保存预测结果

    # 创建模型卡片并推送到模型存储库
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)  # 创建模型卡片，记录模型的关键信息，并推送到指定存储库


if __name__ == "__main__":
    run_sft()