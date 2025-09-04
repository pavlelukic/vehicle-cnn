import sys, numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

IMG_SIZE = (224, 224)
MODEL_PATH = "vehicle_mobilenetv2.keras"
TEST_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples")

class_names = ["bicycle", "bus", "car", "scooter"]
model = load_model(MODEL_PATH)

def prep_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, 0)

for img_path in sorted(TEST_DIR.glob("*")):
    if not img_path.is_file():
        continue
    probs = model.predict(prep_image(img_path), verbose=0)[0]
    pred = class_names[np.argmax(probs)]
    conf = probs.max()
    print(f"{img_path.name:30s} -> {pred:8s} ({conf:.2%})")