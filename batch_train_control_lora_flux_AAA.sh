#! /bin/sh

#SBATCH -J train_control_lora_flux_AAA
#SBATCH -o train_control_lora_flux_AAA.out
#SBATCH -t 72:00:00


#### Select GPU
#SBATCH -q hpgpu
#SBATCH -p A100-80GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1              # number of nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo "Start"

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh    #경로

echo "conda activate pytorch260"
conda activate pytorch260          #사용할 conda env

# SAMPLES_DIR=$HOME/TensorFlow-2.x-Tutorials/
# python3  $SAMPLES_DIR/03-Play-with-MNIST/main.py  # run할 파일(python)

echo "which python3"
which python3

pyclean ./ && accelerate launch --config_file accelerate.yaml --gpu_ids 0 train_control_lora_flux_AAA.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --train_data_dir /home/kkang831/dataset/desobav2_full_res \
    --resolution 512 \
    --guidance_scale 3.5 \
    --learning_rate 3e-5 \
    --train_batch_size 1 \
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
    --max_train_steps 22000

date

echo " conda deactivate pytorch260"
conda deactivate     # 마무리 deactivate

date
squeue  --job  $SLURM_JOBID

echo  "##### END #####"
