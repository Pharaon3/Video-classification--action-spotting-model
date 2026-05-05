"""
Build SoccerClipDataset layout from train.json / valid.json manifests.

Creates:
  videos/<stem>.mp4   (copy from manifest path if source exists under this folder)
  labels/<stem>.json  ({"events": [{"time_sec", "label"}, ...]})
  train.txt / valid.txt  (basenames for train.py --split)

Run:
  python soccer_event_model/dataset/materialize_from_manifest.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

FPS = 25
HERE = Path(__file__).resolve().parent


def stem_for_path(rel: str) -> str:
    p = Path(rel.replace("\\", "/"))
    parent = p.parent.as_posix().replace("/", "_") if p.parent.as_posix() not in (".", "") else ""
    base = p.stem
    if parent:
        return f"{parent}_{base}"
    return base


def convert_manifest(manifest_path: Path) -> list[tuple[str, list[dict], str]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[tuple[str, list[dict], str]] = []
    for item in data.get("videos", []):
        rel = str(item["path"]).replace("\\", "/")
        stem = stem_for_path(rel)
        events = []
        for ann in item.get("annotations", []):
            fr = int(ann["frame"])
            events.append(
                {
                    "time_sec": fr / float(FPS),
                    "label": str(ann["label"]),
                }
            )
        events.sort(key=lambda e: e["time_sec"])
        out.append((stem, events, rel))
    return out


def materialize(entries: list[tuple[str, list[dict], str]], stems_seen: set[str]) -> list[str]:
    videos = HERE / "videos"
    labels = HERE / "labels"
    videos.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    stems_out: list[str] = []
    for stem, events, rel in entries:
        if stem in stems_seen:
            raise RuntimeError(f"Duplicate stem across manifests: {stem}")
        stems_seen.add(stem)
        (labels / f"{stem}.json").write_text(
            json.dumps({"events": events}, indent=2),
            encoding="utf-8",
        )
        stems_out.append(stem)
        src = HERE / rel
        dst = videos / f"{stem}.mp4"
        if src.is_file():
            shutil.copy2(src, dst)
    return stems_out


def main() -> None:
    train_path = HERE / "train.json"
    valid_path = HERE / "valid.json"
    if not train_path.is_file():
        print("Missing train.json", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()
    train_stems = materialize(convert_manifest(train_path), seen)
    valid_stems = materialize(convert_manifest(valid_path), seen) if valid_path.is_file() else []

    (HERE / "train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    if valid_stems:
        (HERE / "valid.txt").write_text("\n".join(valid_stems) + "\n", encoding="utf-8")

    n_vid = len(list((HERE / "videos").glob("*.mp4")))
    n_lbl = len(list((HERE / "labels").glob("*.json")))
    print(f"Labels: {n_lbl} files.")
    print(f"Videos: {n_vid} .mp4 (place clip_*/224p.mp4 under dataset/ then re-run to copy).")


if __name__ == "__main__":
    main()
