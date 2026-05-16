"""Phase 0.7 smoke test — confirm OpenVLA-7B loads on the assigned GPU.

Success criterion (Milestone M0): prints `loaded: ~7.5 B params` without
errors and the model dtype is bfloat16. Run inside an interactive SLURM
session that has at least one GPU and ~16 GB of free VRAM.

    srun --partition=<your_partition> --gres=gpu:l40s:1 --cpus-per-task=4 \\
         --mem=32G --time=00:30:00 --pty bash
    source $VLA_ROOT/code/env.sh && conda activate vla
    python $VLA_ROOT/code/smoke_load_model.py
"""

import argparse
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="openvla/openvla-7b")
    args = p.parse_args()

    assert torch.cuda.is_available(), "No CUDA device — request a GPU node."
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded: {n_params / 1e9:.2f} B params")
    print(f"dtype:  {model.dtype}")
    print(f"processor type: {type(proc).__name__}")
    print("OK")


if __name__ == "__main__":
    main()
