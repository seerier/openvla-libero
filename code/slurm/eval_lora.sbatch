#!/bin/bash
#SBATCH --job-name=eval-lora-spatial
#SBATCH --partition=<your_partition>
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=<your_log_dir>/%x-%j.out

# Phase 3.1 — Your trained LoRA, merged. Expected SR >= 70% (target 80%+).
# Edit MERGED_CKPT below to the actual run directory output by training.

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla
source $VLA_ROOT/code/env.sh

# === EDIT THIS after training completes ===
MERGED_CKPT="${MERGED_CKPT:-$VLA_RUNS/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--lora_r32_lr5e-4/merged}"

cd $VLA_ROOT/openvla

python experiments/robot/libero/run_libero_eval.py \
    --model_family             openvla \
    --pretrained_checkpoint    "$MERGED_CKPT" \
    --task_suite_name          libero_spatial \
    --num_trials_per_task      50 \
    --center_crop              True \
    --run_id_note              self_lora
