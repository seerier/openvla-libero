"""Phase 3.2 — language-perturbation ablation runner.

Wraps OpenVLA's official `experiments/robot/libero/run_libero_eval.py` and
monkey-patches `get_vla_action` so the language instruction passed into the
prompt is replaced according to `--lang_mode`:

    correct    — original instruction (sanity baseline)
    empty      — empty string
    shuffled   — random *different* instruction from the same suite
    unrelated  — fixed unrelated sentence

The upstream eval loop (`eval_libero`) runs unchanged; only the prompt-
builder sees the perturbed string. This is the lightest-touch way to ask
"does the policy actually use language?"

Assumes:
  - OpenVLA repo cloned at $VLA_ROOT/openvla and pip-installed
  - LIBERO installed in the same conda env
  - HF_HOME and MUJOCO_GL set via env.sh

Usage:
    python run_lang_ablation.py \\
        --lang_mode correct \\
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
        --task_suite_name libero_spatial \\
        --num_trials_per_task 50

Run all 4 modes via the matching sbatch wrapper (`slurm/lang_ablation.sbatch`).
"""

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

UNRELATED_SENTENCE = "the weather is nice today"


def build_instruction_pool(task_suite_name):
    """Collect all task descriptions in a LIBERO suite for the 'shuffled' mode."""
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[task_suite_name]()
    pool = []
    for i in range(bench.n_tasks):
        task = bench.get_task(i)
        pool.append(task.language)
    return pool


def mutate(label, mode, pool, rng):
    if mode == "correct":
        return label
    if mode == "empty":
        return ""
    if mode == "unrelated":
        return UNRELATED_SENTENCE
    if mode == "shuffled":
        others = [x for x in pool if x.lower().strip() != (label or "").lower().strip()]
        return rng.choice(others) if others else label
    raise ValueError(f"unknown lang_mode: {mode}")


def make_patched(orig_fn, lang_mode, pool, seed):
    """Wrap get_vla_action so task_label gets rewritten before delegation.

    Uses inspect.signature so we tolerate task_label being passed positionally
    or as a keyword.
    """
    rng = random.Random(seed)
    sig = inspect.signature(orig_fn)

    def patched(*args, **kwargs):
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            return orig_fn(*args, **kwargs)
        bound.apply_defaults()
        if "task_label" in bound.arguments:
            original = bound.arguments["task_label"]
            bound.arguments["task_label"] = mutate(original, lang_mode, pool, rng)
        return orig_fn(*bound.args, **bound.kwargs)

    patched.__wrapped__ = orig_fn
    return patched


def install_monkeypatch(patched):
    """Replace get_vla_action in its source module AND any module that has
    already imported it by name. Necessary because Python imports by value."""
    from experiments.robot import openvla_utils
    openvla_utils.get_vla_action = patched

    from experiments.robot.libero import run_libero_eval
    if hasattr(run_libero_eval, "get_vla_action"):
        run_libero_eval.get_vla_action = patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang_mode", choices=["correct", "empty", "shuffled", "unrelated"],
                   required=True)
    p.add_argument("--pretrained_checkpoint", required=True)
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--center_crop", default="True")
    p.add_argument("--run_id_note", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Ensure the OpenVLA repo is importable when launched from anywhere.
    openvla_root = Path(os.environ.get("VLA_ROOT", ".")) / "openvla"
    if openvla_root.exists() and str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))

    print(f"[lang_ablation] mode = {args.lang_mode}")
    print(f"[lang_ablation] checkpoint = {args.pretrained_checkpoint}")
    print(f"[lang_ablation] suite = {args.task_suite_name}")

    pool = build_instruction_pool(args.task_suite_name)
    print(f"[lang_ablation] instruction pool: {len(pool)} entries")

    from experiments.robot import openvla_utils
    patched = make_patched(openvla_utils.get_vla_action, args.lang_mode, pool, args.seed)
    install_monkeypatch(patched)
    print("[lang_ablation] monkey-patched get_vla_action")

    note = args.run_id_note or f"lang_{args.lang_mode}"
    from experiments.robot.libero.run_libero_eval import GenerateConfig, eval_libero
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        center_crop=(str(args.center_crop).lower() == "true"),
        run_id_note=note,
    )
    eval_libero(cfg)

    out = {
        "lang_mode": args.lang_mode,
        "checkpoint": args.pretrained_checkpoint,
        "task_suite_name": args.task_suite_name,
        "run_id_note": note,
        "instruction_pool_size": len(pool),
    }
    out_dir = Path(os.environ.get("VLA_ROLLOUTS", "./rollouts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lang_ablation_{args.lang_mode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[lang_ablation] wrote marker: {out_path}")


if __name__ == "__main__":
    main()
