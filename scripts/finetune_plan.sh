#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train_judge.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /mnt/afs/chenxuchuan/datasets/qwen_ft/plan.json \
    --val_data_path /mnt/afs/chenxuchuan/datasets/qwen_ft/plan_val.json \
    --calvin_datasets_dir /mnt/afs/chenxuchuan/datasets/calvin/task_ABC_D \
    --image_folder /mnt/afs/chenxuchuan/datasets/qwen_ft/imgs \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/plan \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 2e-6 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_steps 10000 \
    --save_total_limit 2 \
    --dataloader_num_workers 4