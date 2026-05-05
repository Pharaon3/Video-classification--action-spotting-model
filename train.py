"""
Train temporal stack + prediction head on fixed-length clips.

Feature extractor is frozen by default (config: freeze_feature_extractor).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Allow `python path/to/train.py` from any working directory
_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from dataset import SoccerClipDataset
from models.event_model import build_event_model
from utils.checkpoint import save_checkpoint


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = torch.stack([b["video"] for b in batch], dim=0)  # [B,T,3,H,W]
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    paths = [b["video_path"] for b in batch]
    return {"video": videos, "labels": labels, "video_path": paths}


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
) -> float:
    model.train()
    multi_label = bool(cfg.get("multi_label", True))
    log_every = int(cfg.get("training", {}).get("log_every", 10))
    running = 0.0
    n = 0

    for step, batch in enumerate(loader):
        x = batch["video"].to(device)
        y = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # [B,T,C]

        if multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n += 1
        if step % log_every == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    return running / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_PKG / "config.yaml"))
    parser.add_argument("--data_root", type=str, required=True, help="Folder with videos/ and labels/")
    parser.add_argument("--split", type=str, default=None, help="Optional split list file under data_root")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg.setdefault("training", {})

    device_str = str(cfg["training"].get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    backend = str(cfg.get("video_backend", "opencv"))
    ds = SoccerClipDataset(
        args.data_root,
        cfg,
        split=args.split,
        video_backend=backend,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = build_event_model(cfg).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )

    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = int(cfg["training"]["num_epochs"])

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, device, cfg, epoch)
        print(f"epoch {epoch} mean_loss {avg_loss:.4f}")
        save_checkpoint(
            ckpt_dir / f"epoch_{epoch:03d}.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            config=cfg,
        )

    save_checkpoint(
        ckpt_dir / "last.pt",
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=num_epochs,
        config=cfg,
    )
    print(f"Done. Checkpoints in {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()
