"""Phase 3.3 — per-action latency benchmark for an OpenVLA-format checkpoint.

Measures inference latency of greedy 7-action-token autoregressive decode,
the dominant cost of OpenVLA-style policies. Compare against OFT's reported
~10 ms (parallel decode + L1 head) — yours will be ~150-300 ms because you
are running discrete autoregressive decoding. That gap is the OFT pitch.

    python latency_bench.py \\
        --checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
        --unnorm_key libero_spatial
"""

import argparse
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unnorm_key", default="libero_spatial")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--n_iters", type=int, default=50)
    args = p.parse_args()

    assert torch.cuda.is_available(), "No CUDA device."
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint:  {args.checkpoint}")

    proc = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

    img = Image.new("RGB", (224, 224))
    prompt = "In: What action should the robot take to pick up the alphabet soup?\nOut:"
    inputs = proc(prompt, img).to("cuda", torch.bfloat16)

    print(f"Warming up ({args.warmup} iters)...")
    for _ in range(args.warmup):
        model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

    print(f"Timing ({args.n_iters} iters)...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.n_iters):
        model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    torch.cuda.synchronize()

    dt_ms = (time.time() - t0) / args.n_iters * 1000.0
    print(f"per-action latency: {dt_ms:.1f} ms")
    print(f"max control freq:   {1000.0 / dt_ms:.1f} Hz")


if __name__ == "__main__":
    main()
