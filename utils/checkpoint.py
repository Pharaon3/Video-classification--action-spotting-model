"""Save / load training checkpoints with embedded config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]],
    epoch: int,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "config": config,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
