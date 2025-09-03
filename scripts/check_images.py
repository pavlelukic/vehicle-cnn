from pathlib import Path
from PIL import Image, UnidentifiedImageError

ROOT = Path("dataset_split")
bad_paths = []

for path in ROOT.rglob("*.*"):
    if path.is_dir():
        continue
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as ex:
        bad_paths.append(path)

if bad_paths:
    print("Corrupt / non-image files: ")
    for p in bad_paths:
        print(" ", p)

else:
    print("Every file opened successfully!")