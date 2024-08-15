# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    继承自 Seq2SeqTrainer，用于计算生成式指标，如 BLEU 和 ROUGE
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        """
        初始化函数，设置微调参数和处理器，并根据配置添加相应的回调函数。
        
        参数：
        - finetuning_args: 微调参数的实例。
        - processor: 可选的处理器实例。
        - **kwargs: 其他关键字参数，传递给父类的初始化函数。
        """
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            # 如果提供了处理器，则添加保存处理器的回调
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            # 如果配置了 PISSA 转换，则添加相应的回调
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            # 如果使用 BAdam 优化器，则导入相关回调并替换梯度裁剪函数
            from badam import BAdamCallback, clip_grad_norm_old_version

            # 将旧版本的梯度裁剪方法绑定到加速器实例
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            # 添加 BAdam 优化器的回调
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        """
        创建自定义优化器。如果尚未创建优化器，则根据模型、训练参数和微调参数创建。
        
        返回：
        - optimizer: 创建的优化器实例。
        """
        if self.optimizer is None:
            # 创建自定义优化器
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()
    

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        """
        创建自定义学习率调度器。
        
        参数：
        - num_training_steps: 训练的总步数。
        - optimizer: 可选的优化器实例。
        
        返回：
        - scheduler: 创建的学习率调度器实例。
        """
        # 创建自定义调度器
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
        在生成的 tokens 中移除提示部分。

        子类可以重写此方法以注入自定义行为。
        
        参数：
        - model: 待评估的模型。
        - inputs: 输入数据的字典。
        - prediction_loss_only: 是否只返回损失。
        - ignore_keys: 在模型输出中要忽略的键列表。
        
        返回：
        - loss: 计算的损失值。
        - generated_tokens: 生成的 tokens。
        - labels: 原始的标签数据。
        """
        # 如果输入中包含 "labels"，则创建其副本以备后用
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # 备份标签
        if self.args.predict_with_generate:
            # 确保 tokenizer 使用左侧填充
            assert self.tokenizer.padding_side == "left", "此方法仅接受左填充的张量。"
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                # 如果输入长度大于标签长度，则对标签进行填充
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                # 如果标签长度大于输入长度，则截断标签（为了解决 llama2 fp16 兼容性问题）
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        # 调用父类的 prediction_step 方法，忽略返回的标签（可能已被截断）
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            # 将生成的 tokens 中的提示部分替换为填充 token
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels
    

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        将源张量填充到与目标张量相同的长度。

        参数：
        - src_tensor: 源张量，需要被填充。
        - tgt_tensor: 目标张量，提供目标长度。
        
        返回：
        - padded_tensor: 填充后的张量。
        """
        # 确保 tokenizer 已定义填充 token
        assert self.tokenizer.pad_token_id is not None, "需要定义填充 token。"
        # 创建与目标张量形状相同的填充张量，初始化为填充 token 的值
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        # 将源张量的内容填充到填充张量的右侧（采用左填充）
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # 采用左填充
        return padded_tensor.contiguous()  # 确保在连续的内存中
    

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        将模型的预测结果保存到 `output_dir` 中。

        这是一个在 Seq2SeqTrainer 中未包含的自定义行为。
        
        参数：
        - dataset: 用于预测的数据集。
        - predict_results: 包含预测结果的输出。
        """
        # 如果当前进程不是主进程，则不执行保存操作
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"将预测结果保存到 {output_prediction_file}")

        # 将 IGNORE_INDEX 替换为填充 token 的 id，以便于解码
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                # 将填充 token 移动到序列的末尾
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        # 解码输入、标签和预测结果，跳过特殊 tokens
        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                # 将每个结果以 JSON 格式保存
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
