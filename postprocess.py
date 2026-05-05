"""
Turn per-frame logits into a list of detected events (JSON-serializable dicts).

- activation from config: sigmoid (multi-label) or softmax (multi-class)
- confidence threshold
- minimum time gap between detections of the same class (NMS-style)
- time_sec = frame_index / fps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PostprocessConfig:
    fps: float
    activation: Literal["sigmoid", "softmax"]
    threshold: float
    min_event_gap_sec: float
    class_names: List[str]
    multi_label: bool


def logits_to_probs(logits: torch.Tensor, activation: str) -> torch.Tensor:
    """
    logits: [T, num_classes] or [1, T, C] -> squeeze batch if present
    returns probs same shape [T, C]
    """
    if logits.dim() == 3:
        if logits.size(0) != 1:
            raise ValueError("Expected batch size 1 for inference postprocess")
        logits = logits.squeeze(0)
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    if activation == "softmax":
        return F.softmax(logits, dim=-1)
    raise ValueError(f"Unknown activation: {activation}")


def _nms_same_class(
    candidates: List[Dict[str, Any]],
    min_gap_frames: int,
) -> List[Dict[str, Any]]:
    """Greedy NMS on pre-sorted (by confidence desc) same-class or mixed list."""
    kept: List[Dict[str, Any]] = []
    for ev in candidates:
        f = int(ev["frame"])
        cls = ev["event"]
        ok = True
        for k in kept:
            if k["event"] != cls:
                continue
            if abs(int(k["frame"]) - f) < min_gap_frames:
                ok = False
                break
        if ok:
            kept.append(ev)
    return kept


def postprocess_multilabel(
    probs: np.ndarray,
    class_names: List[str],
    fps: float,
    threshold: float,
    min_gap_frames: int,
) -> List[Dict[str, Any]]:
    """
    probs: [T, C] — independent probabilities per class.
    """
    t, c = probs.shape
    assert c == len(class_names)
    candidates: List[Dict[str, Any]] = []
    for ci in range(c):
        for fi in range(t):
            p = float(probs[fi, ci])
            if p >= threshold:
                candidates.append(
                    {
                        "frame": fi,
                        "time": fi / fps,
                        "event": class_names[ci],
                        "confidence": p,
                    }
                )
    candidates.sort(key=lambda e: e["confidence"], reverse=True)
    selected = _nms_same_class(candidates, min_gap_frames)
    selected.sort(key=lambda e: e["frame"])
    return selected


def postprocess_multiclass(
    probs: np.ndarray,
    class_names: List[str],
    fps: float,
    threshold: float,
    min_gap_frames: int,
) -> List[Dict[str, Any]]:
    """
    probs: [T, C] from softmax — emit onset frames when argmax is class c with prob >= th.
    """
    t, c = probs.shape
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    events: List[Dict[str, Any]] = []
    prev = -1
    for fi in range(t):
        cls = int(pred[fi])
        p = float(conf[fi])
        name = class_names[cls]
        if name == "background":
            prev = cls
            continue
        if p < threshold:
            prev = cls
            continue
        # onset: class change into active class
        if cls != prev:
            events.append(
                {
                    "frame": fi,
                    "time": fi / fps,
                    "event": name,
                    "confidence": p,
                }
            )
        prev = cls
    events.sort(key=lambda e: e["confidence"], reverse=True)
    events = _nms_same_class(events, min_gap_frames)
    events.sort(key=lambda e: e["frame"])
    return events


def postprocess_clip(
    logits: torch.Tensor,
    cfg: PostprocessConfig,
) -> List[Dict[str, Any]]:
    """
    logits: [1, T, num_classes] on CPU or CUDA
    """
    probs = logits_to_probs(logits, cfg.activation)
    arr = probs.detach().float().cpu().numpy()
    min_gap_frames = max(1, int(round(cfg.min_event_gap_sec * cfg.fps)))
    if cfg.multi_label:
        return postprocess_multilabel(arr, cfg.class_names, cfg.fps, cfg.threshold, min_gap_frames)
    return postprocess_multiclass(arr, cfg.class_names, cfg.fps, cfg.threshold, min_gap_frames)
