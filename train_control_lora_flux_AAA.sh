py3clean ./ && accelerate launch --config_file accelerate.yaml --gpu_ids 0 train_control_lora_flux_AAA.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --train_data_dir /home/kkang831/dataset/desobav2_full_res \
    --resolution 512 \
    --guidance_scale 3.5 \
    --learning_rate 3e-5 \
    --train_batch_size 1 \
    # --max_train_steps 22000 \
    --max_train_steps 10 \
    --rank 1024 \
    --gaussian_init_lora \
    --tracker_project_name flux_train_control_lora \
    --report_to tensorboard \
    --validation_steps 100 \
    --checkpointing_steps 1000 \
    --use_8bit_adam \
    --allow_tf32 \
    --scale_lr \
    --lr_scheduler cosine_with_restarts \
    --mixed_precision "bf16" \
    --gradient_accumulation_steps 8 \
    # --train_data_dir /data1/kkang/desobav2_full_res \
    # --num_train_epochs 20 \
    # --pretrained_lora_path /aaaidata/weirunpu/diffusers-0.33.0.dev0/flux_control_lora_RORD/pretrained_model/pytorch_lora_weights.safetensors \
    # --resume_from_checkpoint latest \
    # --pretrained_lora_path ckpt/LoRA-RORD/pytorch_lora_weights.safetensors \
    # --train_data_dir /huggingface/dataset_hub/VideoRemoval/train_dataset \
    # --offload \