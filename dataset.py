"""
PyTorch Dataset: paired `videos/` + `labels/` JSON annotations.

Each clip is resized temporally to `num_frames` at `fps` (see utils.video).
Frame targets are built via utils.labels.events_to_frame_labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from utils.dataset_video import load_stem_to_relpath, resolve_clip_video_path
from utils.labels import build_class_to_idx, events_to_frame_labels, load_events_json
from utils.video import VideoPreprocessConfig, preprocess_clip_to_tensor


class SoccerClipDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        config: Dict[str, Any],
        split: Optional[str] = None,
        video_dir: str = "videos",
        labels_dir: str = "labels",
        video_backend: str = "opencv",
    ) -> None:
        """
        Args:
            root: dataset root containing video_dir and labels_dir
            config: loaded YAML dict (must include fps, num_frames, class_names, multi_label, ...)
            split: optional; if provided, expects `root/split.txt` listing basenames (one per line)
        """
        self.root = Path(root)
        self.config = config
        self.video_dir = self.root / video_dir
        self.labels_dir = self.root / labels_dir
        self.multi_label = bool(config.get("multi_label", True))
        self.fps = int(config["fps"])
        self.num_frames = int(config["num_frames"])
        self.class_names: List[str] = list(config["class_names"])
        self.class_to_idx = build_class_to_idx(self.class_names)
        self.radius = int(config.get("label_radius_frames", 2))
        self.backend = video_backend
        self.vpre = VideoPreprocessConfig(
            fps=self.fps,
            num_frames=self.num_frames,
            backend=video_backend,  # type: ignore[arg-type]
        )

        self._stem_to_rel = load_stem_to_relpath(self.root)

        if split:
            split_file = self.root / split
            stems = [s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
            if not stems:
                raise RuntimeError(
                    f"Split file {split_file} is empty. Run `python dataset/materialize_from_manifest.py` "
                    f"to rebuild train.txt from your videos, or list one stem per line."
                )
            self.items = [self._resolve_item(stem) for stem in stems]
        else:
            exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
            vids = sorted(
                p for p in self.video_dir.iterdir() if p.suffix.lower() in exts
            )
            self.items = []
            seen_video_paths: set[str] = set()
            for vp in vids:
                lp = self.labels_dir / f"{vp.stem}.json"
                if lp.is_file():
                    self.items.append((vp, lp))
                    seen_video_paths.add(str(vp.resolve()))
                else:
                    # Skip silently or warn — training usually needs labels
                    continue
            # Also pick up nested manifest paths (e.g. clip_4/224p.mp4) when not copied to videos/
            for stem in self._stem_to_rel:
                vp = resolve_clip_video_path(self.root, stem, self.video_dir, self._stem_to_rel)
                if vp is None:
                    continue
                key = str(vp.resolve())
                if key in seen_video_paths:
                    continue
                lp = self.labels_dir / f"{stem}.json"
                if lp.is_file():
                    seen_video_paths.add(key)
                    self.items.append((vp, lp))

        if not self.items:
            raise RuntimeError(f"No video/label pairs found under {self.root}")

    def _resolve_item(self, stem: str) -> tuple[Path, Path]:
        lp = self.labels_dir / f"{stem}.json"
        if not lp.is_file():
            raise FileNotFoundError(f"Missing label JSON for stem '{stem}': {lp}")
        vp = resolve_clip_video_path(self.root, stem, self.video_dir, self._stem_to_rel)
        if vp is None:
            rel = self._stem_to_rel.get(stem)
            hints: list[str] = []
            if rel:
                r = str(rel).replace("\\", "/")
                hints.append(str(self.root / r))
                hints.append(str(self.video_dir / r))
            hint = f" (tried: {'; '.join(hints)})" if hints else ""
            raise FileNotFoundError(f"No video for stem '{stem}'{hint}")
        return vp, lp

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vpath, lpath = self.items[idx]
        events = load_events_json(lpath)
        video = preprocess_clip_to_tensor(vpath, self.vpre, device=None)  # [T,3,H,W]
        labels = events_to_frame_labels(
            events,
            self.class_to_idx,
            self.num_frames,
            self.fps,
            self.multi_label,
            self.radius,
        )
        return {
            "video": video,  # [T,3,H,W]
            "labels": labels,
            "video_path": str(vpath),
        }
