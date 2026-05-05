"""
Run event detection on a single video clip and write JSON predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from models.event_model import build_event_model
from postprocess import PostprocessConfig, postprocess_clip
from utils.checkpoint import load_checkpoint
from utils.video import VideoPreprocessConfig, preprocess_clip_to_tensor


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML to override checkpoint config (defaults to embedded checkpoint config)",
    )
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg: Dict[str, Any] = ckpt.get("config") or {}
    if args.config:
        override = load_yaml(Path(args.config))
        cfg.update(override)

    if not cfg:
        raise RuntimeError("No config in checkpoint; pass --config path/to/config.yaml")

    device_str = str(cfg.get("inference", {}).get("device", cfg.get("training", {}).get("device", "cuda")))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = build_event_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    vpre = VideoPreprocessConfig(
        fps=int(cfg["fps"]),
        num_frames=int(cfg["num_frames"]),
        backend=str(cfg.get("video_backend", "opencv")),  # type: ignore[arg-type]
    )
    clip = preprocess_clip_to_tensor(args.video, vpre, device=device)  # [T,3,H,W]
    clip = clip.unsqueeze(0)  # [1,T,3,H,W]

    with torch.inference_mode():
        logits = model(clip)  # [1,T,C]

    pp_cfg = PostprocessConfig(
        fps=float(cfg["fps"]),
        activation=str(cfg.get("activation", "sigmoid")),  # type: ignore[arg-type]
        threshold=float(cfg.get("threshold", 0.5)),
        min_event_gap_sec=float(cfg.get("min_event_gap_sec", 1.0)),
        class_names=list(cfg["class_names"]),
        multi_label=bool(cfg.get("multi_label", True)),
    )
    events = postprocess_clip(logits, pp_cfg)
    # Stable JSON: round floats lightly
    for e in events:
        e["time"] = round(float(e["time"]), 4)
        e["confidence"] = round(float(e["confidence"]), 4)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Wrote {len(events)} events to {out_path.resolve()}")


if __name__ == "__main__":
    main()
