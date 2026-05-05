"""
Build SoccerClipDataset layout from train.json / valid.json manifests.

Creates:
  videos/<stem>.mp4   (copy from manifest path if source exists under this folder)
  labels/<stem>.json  ({"events": [{"time_sec", "label"}, ...]})
  train.txt / valid.txt  (basenames for train.py --split — only stems that have a video file)

Run:
  python dataset/materialize_from_manifest.py

Rewrite split lists to match existing files only (no manifest read):
  python dataset/materialize_from_manifest.py --sync-splits
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

FPS = 25
HERE = Path(__file__).resolve().parent

_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def stem_for_path(rel: str) -> str:
    p = Path(rel.replace("\\", "/"))
    parent = p.parent.as_posix().replace("/", "_") if p.parent.as_posix() not in (".", "") else ""
    base = p.stem
    if parent:
        return f"{parent}_{base}"
    return base


def stem_has_video(stem: str, videos_dir: Path) -> bool:
    """True if any supported video file exists for this stem (matches SoccerClipDataset lookup)."""
    for ext in _VIDEO_EXTS:
        if (videos_dir / f"{stem}{ext}").is_file():
            return True
    return False


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
        src = HERE / rel
        dst = videos / f"{stem}.mp4"
        if src.is_file():
            shutil.copy2(src, dst)
        if stem_has_video(stem, videos):
            stems_out.append(stem)
        else:
            try:
                hint = src.relative_to(HERE)
            except ValueError:
                hint = src
            print(
                f"materialize: omitted from split lists (no video under {videos}): stem={stem!r} "
                f"(place {hint})",
                file=sys.stderr,
            )
    return stems_out


def sync_split_lists() -> None:
    """Rewrite train.txt / valid.txt to only include stems that have a matching file in videos/."""
    videos = HERE / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    for name in ("train.txt", "valid.txt"):
        path = HERE / name
        if not path.is_file():
            continue
        stems = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        kept = [s for s in stems if stem_has_video(s, videos)]
        removed = len(stems) - len(kept)
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"{name}: kept {len(kept)} / {len(stems)} stems ({removed} removed without a video file).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize SoccerClipDataset layout from manifests.")
    parser.add_argument(
        "--sync-splits",
        action="store_true",
        help="Only rewrite train.txt and valid.txt: keep stems that have a file under videos/.",
    )
    args = parser.parse_args()

    if args.sync_splits:
        sync_split_lists()
        return

    train_path = HERE / "train.json"
    valid_path = HERE / "valid.json"
    if not train_path.is_file():
        print("Missing train.json", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()
    train_stems = materialize(convert_manifest(train_path), seen)
    valid_stems = materialize(convert_manifest(valid_path), seen) if valid_path.is_file() else []

    (HERE / "train.txt").write_text("\n".join(train_stems) + ("\n" if train_stems else ""), encoding="utf-8")
    if valid_stems:
        (HERE / "valid.txt").write_text("\n".join(valid_stems) + "\n", encoding="utf-8")

    n_vid = len(list((HERE / "videos").glob("*.mp4")))
    n_lbl = len(list((HERE / "labels").glob("*.json")))
    print(f"Labels: {n_lbl} files.")
    print(f"Videos: {n_vid} .mp4 (place clip_*/224p.mp4 under dataset/ then re-run to copy).")
    print(f"train.txt entries with a video file: {len(train_stems)}.")
    if valid_path.is_file():
        print(f"valid.txt entries with a video file: {len(valid_stems)}.")


if __name__ == "__main__":
    main()
