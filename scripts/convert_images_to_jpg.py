import os
from PIL import Image

vehicles = ["bicycle", "bus", "car", "scooter"]

SRC_ROOT = "dataset"
DST_ROOT = "dataset_updated"

for vehicle in vehicles:
    src_dir = os.path.join(SRC_ROOT, vehicle)
    dst_dir = os.path.join(DST_ROOT, vehicle)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)

        if not os.path.isfile(src_path):
            continue

        if not fname.lower().endswith(".jpg"):
            try:
                img = Image.open(src_path).convert("RGB")
                new_name = os.path.splitext(fname)[0] + ".jpg"
                dst_path = os.path.join(dst_dir, new_name)
                img.save(dst_path, "JPEG", quality=90)
                print(f"Converted image {fname} to {new_name}.")
            except Exception as ex:
                print(f"Image {fname} skipped. ERROR: {ex}")
        else:
            dst_path = os.path.join(dst_dir, fname)
            if not os.path.exists(dst_path):
                os.link(src_path, dst_path)