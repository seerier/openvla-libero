"""Post-hoc rollout video capture for the Phase 4 report.

Decoupled from `run_libero_eval.py` because upstream lacks `--save_videos` /
`--videos_dir` flags. This script directly drives the LIBERO env and a
policy, writing one mp4 per rollout via imageio.

Usage:
    python capture_rollouts.py \\
        --checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
        --task_suite_name libero_spatial \\
        --num_rollouts 3 \\
        --out_dir $VLA_ROOT/videos/official_demo

Supports `--lang_mode` so the same script can capture ablation demos
(empty / shuffled / unrelated) for the report's qualitative section.
"""

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def get_libero_env(task_suite_name, task_idx, seed=0):
    """Minimal env init mirroring OpenVLA's get_libero_env helper."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[task_suite_name]()
    task = bench.get_task(task_idx)
    task_description = task.language
    task_bddl = os.path.join(
        bench.get_task_bddl_file_path(task_idx)
    ) if hasattr(bench, "get_task_bddl_file_path") else task.bddl_file

    env_args = {"bddl_file_name": task_bddl, "camera_heights": 256, "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def mutate_instruction(label, lang_mode, pool, rng):
    if lang_mode == "correct":
        return label
    if lang_mode == "empty":
        return ""
    if lang_mode == "unrelated":
        return "the weather is nice today"
    if lang_mode == "shuffled":
        others = [x for x in pool if x != label]
        return rng.choice(others) if others else label
    raise ValueError(lang_mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_rollouts", type=int, default=3,
                   help="Number of (task, seed) rollouts to record.")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--unnorm_key", default="libero_spatial")
    p.add_argument("--lang_mode", default="correct",
                   choices=["correct", "empty", "shuffled", "unrelated"])
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    openvla_root = Path(os.environ.get("VLA_ROOT", ".")) / "openvla"
    if openvla_root.exists() and str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(args.seed)

    # Build pool for 'shuffled' from the suite's task descriptions.
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[args.task_suite_name]()
    pool = [bench.get_task(i).language for i in range(bench.n_tasks)]

    proc = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda").eval()

    for rollout_idx in range(args.num_rollouts):
        task_idx = rollout_idx % bench.n_tasks
        env, task_description = get_libero_env(args.task_suite_name, task_idx, seed=args.seed + rollout_idx)
        instruction = mutate_instruction(task_description, args.lang_mode, pool, rng)

        obs = env.reset()
        frames = []
        success = False
        for t in range(args.max_steps):
            img = Image.fromarray(obs["agentview_image"][::-1])  # LIBERO returns flipped
            frames.append(np.array(img))

            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            inputs = proc(prompt, img).to("cuda", torch.bfloat16)
            action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break

        env.close()

        tag = "success" if success else "fail"
        out_path = out_dir / f"rollout_{rollout_idx:02d}_task{task_idx}_{args.lang_mode}_{tag}.mp4"
        imageio.mimwrite(out_path, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"  rollout {rollout_idx}: task={task_idx} steps={len(frames)} {tag} -> {out_path}")


if __name__ == "__main__":
    main()
