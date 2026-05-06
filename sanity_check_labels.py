#!/usr/bin/env python3
"""
Scan label JSON under a dataset root and report coverage vs config class names.

Uses the same per-class label radius and strict_labels behavior as training when
building positive frame counts (JSON only — no video decode).

Exit code 1 when strict_labels is true in config and any unknown label strings appear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from utils.label_stats import (
    aggregate_frame_counts_from_json,
    compute_auto_pos_weight_numpy,
    count_events_per_class_from_json,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check label JSON files against config class names and radii."
    )
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root (videos/, labels/)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument(
        "--split",
        type=str,
        default="train.txt",
        help="Split list file relative to data_root (default: train.txt). Use empty string for auto scan.",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    class_names: List[str] = list(cfg["class_names"])
    strict = bool(cfg.get("strict_labels", False))
    clip_max = float((cfg.get("loss") or {}).get("pos_weight_clip_max", 20.0))

    split_arg: str | None = args.split.strip() if args.split else None
    if split_arg == "":
        split_arg = None

    root = Path(args.data_root)
    split_path = root / split_arg if split_arg else None
    if split_arg and not split_path.is_file():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Config classes ({len(class_names)}): {class_names}")
    print(f"strict_labels={strict}  split={split_arg!r}")

    counts, unknown_rows = count_events_per_class_from_json(cfg, root, split_arg)
    print("\nRaw JSON event counts (by string class field):")
    for n in class_names:
        print(f"  {n:20s}  {counts[n]}")

    if unknown_rows:
        print(f"\nUnknown label strings ({len(unknown_rows)} event(s), {len(unknown_names)} distinct):")
        for name, stem, path in unknown_rows[:50]:
            print(f"  {name!r}  stem={stem!r}  file={path}")
        if len(unknown_rows) > 50:
            print(f"  ... ({len(unknown_rows) - 50} more)")
    else:
        print("\nNo unknown label strings in JSON (by class/event/label/name fields).")

    if strict and unknown_rows:
        print(
            "\nFAIL: strict_labels is true but unknown labels exist. Fix JSON or set strict_labels: false.",
            file=sys.stderr,
        )
        sys.exit(1)

    pos, neg, T, label_paths = aggregate_frame_counts_from_json(cfg, root, split_arg)
    n_clips = len(label_paths)
    print(f"\nClips (label files) scanned: {n_clips}  frames_per_clip={T}")
    print("Positive frame counts (tensor labels with per-class radius):")
    for i, n in enumerate(class_names):
        print(f"  {n:20s}  pos={int(pos[i]):7d}  neg={int(neg[i]):7d}")

    pw = compute_auto_pos_weight_numpy(cfg, root, split_arg, clip_max=clip_max)
    print(f"\nApproximate BCE pos_weight (neg/max(pos,1), clip<={clip_max}):")
    for i, n in enumerate(class_names):
        print(f"  {n:20s}  {float(pw[i]):.6f}")

    print("\nOK.")
    sys.exit(0)


if __name__ == "__main__":
    main()
