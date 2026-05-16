#!/bin/bash
#SBATCH --job-name=eval-zeroshot-spatial
#SBATCH --partition=kostas-compute
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/kostas_home/gxzhao4/vla/logs/%x-%j.out

# Phase 1.2 — Zero-shot OpenVLA-7B on LIBERO-Spatial. Expected SR ~0-20%
# (no LIBERO in OXE pretrain). Lower bound for the 4-row results table.
# Safe to run in parallel with training (fast-path execution).

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla
source $VLA_ROOT/code/env.sh

cd $VLA_ROOT/openvla

python experiments/robot/libero/run_libero_eval.py \
    --model_family             openvla \
    --pretrained_checkpoint    openvla/openvla-7b \
    --task_suite_name          libero_spatial \
    --num_trials_per_task      50 \
    --center_crop              True \
    --run_id_note              zero_shot
