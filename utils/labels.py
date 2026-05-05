"""
Helpers to map JSON event annotations to per-frame targets.

Label JSON is intentionally generic (no SoccerNet). Example:

{
  "events": [
    {"time_sec": 4.8, "class": "pass"},
    {"time": 12.0, "event": "shot"}
  ]
}

Field aliases supported: time_sec / time / t for seconds; class / event / label for name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def load_events_json(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return list(data["events"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported label format in {path}")


def _event_time_seconds(ev: Dict[str, Any]) -> float:
    for key in ("time_sec", "time", "t", "timestamp_sec"):
        if key in ev:
            return float(ev[key])
    raise KeyError(f"Event has no time field: {ev}")


def _event_class_name(ev: Dict[str, Any]) -> str:
    for key in ("class", "event", "label", "name"):
        if key in ev:
            return str(ev[key])
    raise KeyError(f"Event has no class field: {ev}")


def events_to_frame_labels(
    events: List[Dict[str, Any]],
    class_to_idx: Dict[str, int],
    num_frames: int,
    fps: int,
    multi_label: bool,
    radius_frames: int,
) -> torch.Tensor:
    """
    Build dense frame labels of shape [T, C] (multi-label float 0/1) or [T] long (multi-class).

    multi_label True: returns float tensor [T, num_classes]
    multi_class: returns long tensor [T] with values in [0, num_classes] where index 0
        is reserved for background if "background" is in class_to_idx; otherwise uses 0
        as background only where no event spans a frame (see below).

    For multi-class without explicit background in class_names, unlabeled frames are 0
    and the first class index in class_to_idx may collide — users should add "background"
    as first class when using multi-class per-frame classification.
    """
    num_classes = len(class_to_idx)
    if multi_label:
        y = torch.zeros((num_frames, num_classes), dtype=torch.float32)
        for ev in events:
            t_sec = _event_time_seconds(ev)
            name = _event_class_name(ev)
            if name not in class_to_idx:
                continue
            c = class_to_idx[name]
            center = int(round(t_sec * fps))
            lo = max(0, center - radius_frames)
            hi = min(num_frames - 1, center + radius_frames)
            y[lo : hi + 1, c] = 1.0
        return y

    # Multi-class: single active class per frame; later frames override earlier overlaps
    if "background" in class_to_idx:
        bg = class_to_idx["background"]
    else:
        bg = 0
    y = torch.full((num_frames,), bg, dtype=torch.long)
    for ev in events:
        t_sec = _event_time_seconds(ev)
        name = _event_class_name(ev)
        if name not in class_to_idx:
            continue
        c = class_to_idx[name]
        center = int(round(t_sec * fps))
        lo = max(0, center - radius_frames)
        hi = min(num_frames - 1, center + radius_frames)
        y[lo : hi + 1] = int(c)
    return y


def build_class_to_idx(class_names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(class_names)}
