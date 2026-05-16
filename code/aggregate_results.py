"""Collect SR numbers from all eval runs into one markdown table + JSON.

Walks the LIBERO rollout directories produced by OpenVLA's eval script and
the language-ablation markers, then prints a digestible summary for the
Phase 4 report.

OpenVLA's `run_libero_eval.py` writes rollouts under `./rollouts/<DATE>/`
with per-run subdirectories named using `run_id_note`. Each subdirectory
typically contains a log file with the final success rate. This script
parses those.

Usage:
    python aggregate_results.py \\
        --rollouts_dir $VLA_ROLLOUTS \\
        --out results/summary.json
"""

import argparse
import json
import re
from pathlib import Path

SR_PATTERNS = [
    re.compile(r"success rate[^0-9]+([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"final SR[^0-9]+([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\"success_rate\"\s*:\s*([0-9]+\.?[0-9]*)"),
]


def extract_sr(text):
    """Return the last success rate (%) found in text, normalized to [0,100], or None."""
    found = []
    for pat in SR_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
                # Heuristic: if value <= 1, it's a fraction; convert to %
                if v <= 1.0:
                    v *= 100.0
                found.append(v)
            except ValueError:
                pass
    return found[-1] if found else None


def scan_rollouts(rollouts_dir):
    """Return list of {run_id, sr, source} for every run found."""
    rollouts_dir = Path(rollouts_dir)
    runs = []
    for sub in sorted(rollouts_dir.rglob("*")):
        if not sub.is_dir():
            continue
        # Skip non-leaf dirs that contain other run dirs
        if any(c.is_dir() for c in sub.iterdir()):
            continue
        # Aggregate any *.txt / *.log / *.json content in this dir
        blob = []
        for f in sub.iterdir():
            if f.suffix in (".txt", ".log", ".json"):
                try:
                    blob.append(f.read_text(errors="ignore"))
                except OSError:
                    pass
        if not blob:
            continue
        sr = extract_sr("\n".join(blob))
        if sr is None:
            continue
        runs.append({"run_id": sub.name, "sr": sr, "source": str(sub)})
    return runs


def scan_lang_markers(rollouts_dir):
    """Return list of {lang_mode, ...} marker dicts written by run_lang_ablation.py."""
    rollouts_dir = Path(rollouts_dir)
    markers = []
    for f in sorted(rollouts_dir.glob("lang_ablation_*.json")):
        try:
            markers.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return markers


def print_table(runs):
    print()
    print("| run_id | success rate (%) |")
    print("|--------|------------------:|")
    for r in sorted(runs, key=lambda x: x["run_id"]):
        print(f"| `{r['run_id']}` | {r['sr']:.1f} |")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts_dir", default=".", help="Parent of <DATE>/<run> trees")
    p.add_argument("--out", default="results/summary.json")
    args = p.parse_args()

    runs = scan_rollouts(args.rollouts_dir)
    markers = scan_lang_markers(args.rollouts_dir)

    print(f"Found {len(runs)} eval runs with parseable SR.")
    print(f"Found {len(markers)} language-ablation markers.")
    print_table(runs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs, "lang_markers": markers}, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
