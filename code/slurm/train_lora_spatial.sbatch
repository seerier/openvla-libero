#!/bin/bash
#SBATCH --job-name=lora-spatial
#SBATCH --partition=<your_partition>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=<your_log_dir>/%x-%j.out

# LoRA fine-tune of OpenVLA-7B on LIBERO-Spatial.
# Wraps the official `vla-scripts/finetune.py` with our paths and hparams.

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla
source $VLA_ROOT/code/env.sh

cd $VLA_ROOT/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path                openvla/openvla-7b \
    --data_root_dir           $VLA_DATASETS/modified_libero_rlds \
    --dataset_name            libero_spatial_no_noops \
    --run_root_dir            $VLA_RUNS \
    --adapter_tmp_dir         $VLA_RUNS/adapter_tmp \
    --use_lora                True \
    --lora_rank               32 \
    --batch_size              16 \
    --grad_accumulation_steps 1 \
    --learning_rate           5e-4 \
    --image_aug               True \
    --max_steps               50000 \
    --save_steps              5000 \
    --wandb_project           "$WANDB_PROJECT" \
    --wandb_entity            "$WANDB_ENTITY" \
    --run_id_note             lora_r32_lr5e-4
