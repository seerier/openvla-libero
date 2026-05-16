"""Phase 2.5 — merge a trained LoRA adapter back into the OpenVLA base model.

Produces a single self-contained checkpoint directory loadable by
`AutoModelForVision2Seq.from_pretrained`. Required for eval / inference.

Crucially, this also copies the `dataset_statistics.json` from the LoRA run
directory into the merged checkpoint, without which `predict_action(...,
unnorm_key=...)` returns all-zero actions at eval time.

    python merge_lora.py \\
        --base openvla/openvla-7b \\
        --lora /mnt/kostas_home/gxzhao4/vla/runs/<run_dir>/lora_adapter \\
        --out  /mnt/kostas_home/gxzhao4/vla/runs/<run_dir>/merged

The OpenVLA finetune.py also auto-merges after each checkpoint; this script
exists for explicit re-merging from a specific intermediate checkpoint.
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="HF id or path of base OpenVLA model")
    p.add_argument("--lora", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--out", required=True, help="Output directory for merged checkpoint")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora)
    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged checkpoint to: {out_dir}")
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(args.base, trust_remote_code=True).save_pretrained(out_dir)

    # Copy dataset_statistics.json from the LoRA run dir (may live next to the
    # adapter or one level up depending on OpenVLA version).
    lora_path = Path(args.lora)
    for candidate in [lora_path / "dataset_statistics.json",
                      lora_path.parent / "dataset_statistics.json"]:
        if candidate.exists():
            shutil.copy(candidate, out_dir / "dataset_statistics.json")
            print(f"Copied dataset_statistics.json from {candidate}")
            break
    else:
        print("WARNING: dataset_statistics.json not found. Eval will produce zero actions.")

    print("Done.")


if __name__ == "__main__":
    main()
