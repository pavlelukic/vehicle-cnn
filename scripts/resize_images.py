import os
from PIL import Image

ROOT = "dataset_updated"
SIZE = (224, 224)

skipped = processed = 0

for label in os.listdir(ROOT):
    label_dir = os.path.join(ROOT, label)

    if not os.path.isdir(label_dir):
        continue

    for fname in os.listdir(label_dir):
        path = os.path.join(label_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(SIZE, Image.Resampling.LANCZOS)
                img.save(path, "JPEG", quality=90)
            processed += 1
        except Exception as ex:
            print(f"Image {path} skipped. ERROR: {ex}")
            skipped += 1

message = f"{processed} resized, {skipped} skipped." if skipped else f"All {processed} images resized successfully!"
print(message)