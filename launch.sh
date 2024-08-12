# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml


python -m torch.distributed.launch \
  --nproc_per_node 8 \
  ./src/train.py \
    --deepspeed ./examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/ma-user/work/01440561/01449344/output \
    --dataset_dir ./data \
    --dataset identity \
    --template qwen \
    --finetuning_type full \
    --output_dir ./output  \
    --overwrite_cache true \
    --overwrite_output_dir true \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --bf16 \
    --val_size 0.1 \
    --eval_steps 3 \
    --evaluation_strategy steps