"""
PyTorch Dataset: paired `videos/` + `labels/` JSON annotations.

Each clip is resized temporally to `num_frames` at `fps` (see utils.video).
Frame targets are built via utils.labels.events_to_frame_labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from utils.dataset_video import load_stem_to_relpath, resolve_clip_video_path
from utils.labels import (
    build_class_to_idx,
    events_to_frame_labels,
    load_events_json,
    parse_label_radius_frames,
)
from utils.video import VideoPreprocessConfig, preprocess_clip_to_tensor


def discover_clip_items(
    root: Path,
    split: Optional[str],
    video_dir: Path,
    labels_dir: Path,
    stem_to_rel: Dict[str, str],
) -> List[Tuple[Path, Path]]:
    """
    Return (video_path, label_json_path) pairs the same way SoccerClipDataset does.

    Used by training utilities that need label paths without decoding video.
    """
    if split:
        split_file = root / split
        stems = [s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not stems:
            raise RuntimeError(
                f"Split file {split_file} is empty. Run `python dataset/materialize_from_manifest.py` "
                f"to rebuild train.txt from your videos, or list one stem per line."
            )
        items: List[Tuple[Path, Path]] = []
        for stem in stems:
            items.append(_resolve_item(root, stem, video_dir, labels_dir, stem_to_rel))
        return items

    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    vids = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)
    items = []
    seen_video_paths: set[str] = set()
    for vp in vids:
        lp = labels_dir / f"{vp.stem}.json"
        if lp.is_file():
            items.append((vp, lp))
            seen_video_paths.add(str(vp.resolve()))
        else:
            continue
    for stem in stem_to_rel:
        vp = resolve_clip_video_path(root, stem, video_dir, stem_to_rel)
        if vp is None:
            continue
        key = str(vp.resolve())
        if key in seen_video_paths:
            continue
        lp = labels_dir / f"{stem}.json"
        if lp.is_file():
            seen_video_paths.add(key)
            items.append((vp, lp))

    if not items:
        raise RuntimeError(f"No video/label pairs found under {root}")
    return items


def _resolve_item(
    root: Path,
    stem: str,
    video_dir: Path,
    labels_dir: Path,
    stem_to_rel: Dict[str, str],
) -> Tuple[Path, Path]:
    lp = labels_dir / f"{stem}.json"
    if not lp.is_file():
        raise FileNotFoundError(f"Missing label JSON for stem '{stem}': {lp}")
    vp = resolve_clip_video_path(root, stem, video_dir, stem_to_rel)
    if vp is None:
        rel = stem_to_rel.get(stem)
        hints: list[str] = []
        if rel:
            r = str(rel).replace("\\", "/")
            hints.append(str(root / r))
            hints.append(str(video_dir / r))
        hint = f" (tried: {'; '.join(hints)})" if hints else ""
        raise FileNotFoundError(f"No video for stem '{stem}'{hint}")
    return vp, lp


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
        self.label_radius_spec = parse_label_radius_frames(
            config.get("label_radius_frames", 10),
            self.class_names,
        )
        self.strict_labels = bool(config.get("strict_labels", False))
        self.backend = video_backend
        self.vpre = VideoPreprocessConfig(
            fps=self.fps,
            num_frames=self.num_frames,
            backend=video_backend,  # type: ignore[arg-type]
        )

        self._stem_to_rel = load_stem_to_relpath(self.root)
        self.items = discover_clip_items(
            self.root,
            split,
            self.video_dir,
            self.labels_dir,
            self._stem_to_rel,
        )

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
            self.label_radius_spec,
            strict_labels=self.strict_labels,
            label_source=lpath,
        )
        return {
            "video": video,  # [T,3,H,W]
            "labels": labels,
            "video_path": str(vpath),
        }
