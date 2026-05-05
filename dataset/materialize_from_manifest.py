"""
Build SoccerClipDataset layout from train.json / valid.json manifests.

Creates:
  videos/<stem>.mp4   (copy from manifest path if source exists under this folder)
  labels/<stem>.json  ({"events": [{"time_sec", "label"}, ...]})
  train.txt / valid.txt  (basenames for train.py --split — only stems with a resolvable video)

A video counts as present if any of:
  - dataset/videos/<stem>.<ext>
  - dataset/<manifest_path> (e.g. clip_4/224p.mp4)
  - dataset/videos/<manifest_path> (e.g. videos/clip_4/224p.mp4)

Run:
  python dataset/materialize_from_manifest.py

Rewrite split lists to match resolvable videos only (no manifest read):
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
_ROOT = HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.dataset_video import load_stem_to_relpath, resolve_clip_video_path, stem_for_manifest_rel


def convert_manifest(manifest_path: Path) -> list[tuple[str, list[dict], str]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[tuple[str, list[dict], str]] = []
    for item in data.get("videos", []):
        rel = str(item["path"]).replace("\\", "/")
        stem = stem_for_manifest_rel(rel)
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


def materialize(
    entries: list[tuple[str, list[dict], str]],
    stems_seen: set[str],
    stem_rel_map: dict[str, str],
) -> list[str]:
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
        rel_norm = str(rel).replace("\\", "/")
        dst = videos / f"{stem}.mp4"
        # Copy only from dataset-root nested layout into flat videos/; clips already at
        # videos/<rel> are used in place (no duplicate flat file).
        root_nested = HERE / rel_norm
        if root_nested.is_file() and root_nested.resolve() != dst.resolve():
            shutil.copy2(root_nested, dst)
        if resolve_clip_video_path(HERE, stem, videos, stem_rel_map) is not None:
            stems_out.append(stem)
        else:
            try:
                hint = (HERE / rel_norm).relative_to(HERE)
            except ValueError:
                hint = HERE / rel_norm
            print(
                f"materialize: omitted from split lists (no video): stem={stem!r} "
                f"(add flat under {videos}, or nested {hint} or {videos / rel_norm})",
                file=sys.stderr,
            )
    return stems_out


def sync_split_lists() -> None:
    """Rewrite train.txt / valid.txt to only include stems with a resolvable video path."""
    videos = HERE / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    stem_rel_map = load_stem_to_relpath(HERE)
    for name in ("train.txt", "valid.txt"):
        path = HERE / name
        if not path.is_file():
            continue
        stems = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        kept = [
            s
            for s in stems
            if resolve_clip_video_path(HERE, s, videos, stem_rel_map) is not None
        ]
        removed = len(stems) - len(kept)
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"{name}: kept {len(kept)} / {len(stems)} stems ({removed} removed without a video file).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize SoccerClipDataset layout from manifests.")
    parser.add_argument(
        "--sync-splits",
        action="store_true",
        help="Only rewrite train.txt and valid.txt: keep stems with a resolvable video path.",
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

    stem_rel_map = load_stem_to_relpath(HERE)
    seen: set[str] = set()
    train_stems = materialize(convert_manifest(train_path), seen, stem_rel_map)
    valid_stems = (
        materialize(convert_manifest(valid_path), seen, stem_rel_map) if valid_path.is_file() else []
    )

    (HERE / "train.txt").write_text("\n".join(train_stems) + ("\n" if train_stems else ""), encoding="utf-8")
    if valid_stems:
        (HERE / "valid.txt").write_text("\n".join(valid_stems) + "\n", encoding="utf-8")

    n_flat = len(list((HERE / "videos").glob("*.mp4")))
    def _nested_clip_exists(rel: str) -> bool:
        r = str(rel).replace("\\", "/")
        return (HERE / r).is_file() or ((HERE / "videos") / r).is_file()

    n_nested = sum(1 for rel in stem_rel_map.values() if _nested_clip_exists(rel))
    n_lbl = len(list((HERE / "labels").glob("*.json")))
    print(f"Labels: {n_lbl} files.")
    print(f"Videos (flat under videos/): {n_flat} .mp4")
    print(f"Videos (nested manifest paths on disk): {n_nested} files.")
    print(f"train.txt entries with a resolvable video: {len(train_stems)}.")
    if valid_path.is_file():
        print(f"valid.txt entries with a resolvable video: {len(valid_stems)}.")


if __name__ == "__main__":
    main()
