#!/bin/bash
#SBATCH --job-name=lang-ablation
#SBATCH --partition=kostas-compute
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/kostas_home/gxzhao4/vla/logs/%x-%j-%a.out

# Phase 3.2 — Language-perturbation ablation.
# Submit one job per mode:
#   sbatch slurm/lang_ablation.sbatch correct
#   sbatch slurm/lang_ablation.sbatch empty
#   sbatch slurm/lang_ablation.sbatch shuffled
#   sbatch slurm/lang_ablation.sbatch unrelated
#
# Runs against the *official* LIBERO-Spatial checkpoint (stronger reference than
# your LoRA — we want the cleanest signal about language grounding).

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla
source $VLA_ROOT/code/env.sh

LANG_MODE="${1:?missing lang_mode arg: correct|empty|shuffled|unrelated}"
CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"

cd $VLA_ROOT/openvla   # so 'experiments.robot.*' imports resolve

python $VLA_ROOT/code/run_lang_ablation.py \
    --lang_mode             "$LANG_MODE" \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name       libero_spatial \
    --num_trials_per_task   50 \
    --center_crop           True \
    --run_id_note           "lang_${LANG_MODE}"
