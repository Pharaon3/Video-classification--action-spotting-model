Layout for soccer_event_model/train.py (unchanged):

  dataset/
    videos/          <- one .mp4 per clip, basename matches labels/*.json stem
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

That copies each file to videos/<stem>.mp4, writes labels/, and refreshes train.txt / valid.txt.
Split lists only name stems that actually have a video file under videos/ (missing sources are skipped).

If labels are already materialized but train.txt lists missing clips, run:
  python dataset/materialize_from_manifest.py --sync-splits

Train:
  python train.py --data_root dataset --split train.txt

Frame indices in the original JSON are converted with time_sec = frame / 25.
