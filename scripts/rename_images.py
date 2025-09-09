import os

ROOT = "dataset_preprocessed"

for label in os.listdir(ROOT):
    label_dir = os.path.join(ROOT, label)

    if not os.path.isdir(label_dir):
        continue

    count = 1

    for fname in sorted(os.listdir(label_dir)):
        src = os.path.join(label_dir, fname)

        if not os.path.isfile(src):
            continue

        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        dst = os.path.join(label_dir, f"{label}_{count:05d}.jpg")
        if src != dst:
            os.rename(src, dst)

        count += 1

print("Images have been renamed successfully!")
