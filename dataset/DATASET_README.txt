Layout for soccer_event_model/train.py (unchanged):

  dataset/
    videos/          <- optional flat copies: <stem>.mp4 (same stem as labels/*.json)
    clip_*/224p.mp4  <- nested paths from train.json / valid.json also work (no copy required)
    labels/          <- {"events":[{"time_sec", "label"}, ...]}  (from train/valid manifests)
    train.txt        <- basenames for --split train.txt
    valid.txt        <- basenames for --split valid.txt
    train.json       <- original manifest (optional archive)
    valid.json

Populate nested clips to match manifest paths, then materialize:

  dataset/clip_4/224p.mp4
  dataset/clip_6/224p.mp4
  ...

Then run:
  python dataset/materialize_from_manifest.py

That copies each file to videos/<stem>.mp4 when the nested file exists, writes labels/, and
refreshes train.txt / valid.txt. Split lists name stems that have a resolvable video: either
under videos/ or at the nested path from the manifest (e.g. clip_4/224p.mp4).

Training (dataset.py) resolves the same way via utils/dataset_video.py.

If labels are already materialized but train.txt lists missing clips, run:
  python dataset/materialize_from_manifest.py --sync-splits

Train:
  python train.py --data_root dataset --split train.txt

Frame indices in the original JSON are converted with time_sec = frame / 25.
