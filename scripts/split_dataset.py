import os, random, shutil
from pathlib import Path

SRC = Path("dataset_preprocessed")
DST = Path("dataset_split")

TEST_RATIO = 0.2
random.seed(42)

for label_dir in SRC.iterdir():
    if not label_dir.is_dir():
        continue

    images = [p for p in label_dir.iterdir() if p.is_file()]
    random.shuffle(images)

    split = int(len(images) * (1 - TEST_RATIO))

    for i, src_path in enumerate(images):
        subset = "train" if i < split else "test"
        dst_path = DST / subset / label_dir.name / src_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

print("Train/test folders are ready under dataset_split/")

