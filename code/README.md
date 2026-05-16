# `code/` — OpenVLA × LIBERO fine-tune scripts

Thin layer on top of the upstream **openvla/openvla** and **Lifelong-Robot-Learning/LIBERO** repositories. This folder holds only what is custom to this LoRA-fine-tune project; everything else is invoked directly from the upstream clones.

## Layout

```
code/
├── README.md                       (this file)
├── env.sh                          source-able env vars
├── smoke_load_model.py             Phase 0.7 — proves 7B loads on GPU (M0)
├── merge_lora.py                   Phase 2.5 — fuse LoRA into base for eval
├── run_lang_ablation.py            Phase 3.2 — language-perturbation runner
├── capture_rollouts.py             Phase 1.4 + 3.4 — mp4 demos for the report
├── latency_bench.py                Phase 3.3 — per-action inference latency
├── aggregate_results.py            Phase 4 — collect SR numbers into a table
└── slurm/
    ├── train_lora_spatial.sbatch   Phase 2 — the 50k-step training job
    ├── eval_zero_shot.sbatch       Phase 1.2 — OpenVLA-7B, no fine-tune
    ├── eval_official.sbatch        Phase 1.3 — authors' LIBERO-Spatial ckpt
    ├── eval_lora.sbatch            Phase 3.1 — your merged LoRA
    └── lang_ablation.sbatch        Phase 3.2 — takes $1 = lang_mode
```

## Run order (fast path)

The execution path we chose submits training first and the two slow Phase 1 baselines in parallel, to minimize active time at the cost of slightly higher risk. The 45-minute Phase 1.1 single-task LIBERO env smoke is NOT skipped — it catches MuJoCo/EGL failures before a 12+ hour GPU run.

```bash
# 0. One-time env on the cluster (clone openvla + LIBERO as siblings under $VLA_ROOT,
#    pip install -e both, then activate the vla conda env)
source $VLA_ROOT/code/env.sh
conda activate vla

# 0.7  Model load smoke (interactive GPU session, ~30 min)
python code/smoke_load_model.py

# 1.1  Single-task LIBERO env smoke (interactive, ~45 min) — DO NOT SKIP
cd $VLA_ROOT/openvla
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --num_trials_per_task 1 --center_crop True

# 2.   Launch training (background, 12-15h)
sbatch code/slurm/train_lora_spatial.sbatch

# 1.2 + 1.3  Run the two slow baselines in parallel with training
sbatch code/slurm/eval_zero_shot.sbatch
sbatch code/slurm/eval_official.sbatch

# 2.5  After training finishes — merge LoRA
python code/merge_lora.py \
    --base openvla/openvla-7b \
    --lora $VLA_RUNS/<your_run_dir>/lora_adapter \
    --out  $VLA_RUNS/<your_run_dir>/merged

# 3.1  Eval your merged LoRA
MERGED_CKPT=$VLA_RUNS/<your_run_dir>/merged sbatch code/slurm/eval_lora.sbatch

# 3.2  Language ablation (4 runs in parallel)
for mode in correct empty shuffled unrelated; do
    sbatch code/slurm/lang_ablation.sbatch $mode
done

# 3.3  Latency bench (interactive)
python code/latency_bench.py \
    --checkpoint $VLA_RUNS/<your_run_dir>/merged \
    --unnorm_key libero_spatial

# 3.4  Rollout demos for the report (interactive)
python code/capture_rollouts.py \
    --checkpoint $VLA_RUNS/<your_run_dir>/merged \
    --task_suite_name libero_spatial \
    --num_rollouts 5 \
    --out_dir $VLA_ROOT/videos/lora_demo

# 4.   Aggregate everything for the report
python code/aggregate_results.py --rollouts_dir $VLA_ROLLOUTS --out results/summary.json
```

## Conventions

- All scripts assume the `vla` conda env (Python 3.10, OpenVLA-pinned PyTorch/transformers/peft).
- All paths live under `$VLA_ROOT=/mnt/kostas_home/gxzhao4/vla/` — see `env.sh`.
- All long-running jobs go through SLURM on `kostas-compute` with one L40S; the A40 nodes are an interchangeable fallback.
- Custom scripts deliberately *do not* import from `openvla/` package paths until inside `main()` — keeps `--help` fast and avoids loading the upstream world for trivial calls.

## Acknowledgements

This project depends on:

- **[openvla/openvla](https://github.com/openvla/openvla)** (MIT) — base model, training script, eval harness. Pin the upstream commit SHA used on the cluster in this README once setup completes.
- **[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** (MIT) — simulation benchmark and demo data.
- **[openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)** (MIT) — RLDS-formatted LIBERO data, ~10 GB.
