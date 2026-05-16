# Shared environment variables for the OpenVLA × LIBERO project.
# Source this from every interactive session and every sbatch script:
#   source /mnt/kostas_home/gxzhao4/vla/code/env.sh

# HuggingFace cache — single source of truth, never override per-script
export HF_HOME=/mnt/kostas_home/gxzhao4/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# LIBERO / MuJoCo headless rendering — egl is fast, osmesa is the fallback
export MUJOCO_GL=egl

# W&B routing — set once, reused by training + eval
export WANDB_PROJECT=openvla-libero
export WANDB_ENTITY=gxzhao4

# Project root on GRASP (the parent of openvla/, LIBERO/, runs/, datasets/, logs/)
export VLA_ROOT=/mnt/kostas_home/gxzhao4/vla

# Convenience derived paths
export VLA_RUNS=$VLA_ROOT/runs
export VLA_DATASETS=$VLA_ROOT/datasets
export VLA_LOGS=$VLA_ROOT/logs
export VLA_ROLLOUTS=$VLA_ROOT/rollouts
