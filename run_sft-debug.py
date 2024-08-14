
import sys
from typing import TYPE_CHECKING, List, Optional

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset # type: ignore
from llamafactory.extras.constants import IGNORE_INDEX # type: ignore
from llamafactory.extras.misc import get_logits_processor
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.callbacks import LogCallback
from llamafactory.hparams import get_train_args


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments



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



if __name__ == '__main__':
    callbacks: List["TrainerCallback"] = []
    # LLaMA-Factory 官方的全量微调命令
    sys.argv = ["FORCE_TORCHRUN=1", "examples/train_full/llama3_full_sft_ds3.yaml"]
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(None)
    callbacks.append(LogCallback())
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)