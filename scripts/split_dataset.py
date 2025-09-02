import os, random, shutil
from pathlib import Path

SRC = Path("dataset_updated")
DST = Path("dataset_split")

VAL_RATIO = 0.2
random.seed(42)

for label_dir in SRC.iterdir():
    if not label_dir.is_dir():
        continue

    images = [p for p in label_dir.iterdir() if p.is_file()]
    random.shuffle(images)

    split = int(len(images) * (1 - VAL_RATIO))

    for i, src_path in enumerate(images):
        subset = "train" if i < split else "val"
        dst_path = DST / subset / label_dir.name / src_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

print("Train/val folders are ready under dataset_split/")

