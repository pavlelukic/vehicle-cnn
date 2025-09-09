import os, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_DIR = "dataset_split"
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS_FROZEN = 10
EPOCHS_FINE = 5
FINE_TUNE_LR = 1e-4

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True)

test_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True)

test_gen = test_aug.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False)

num_classes = train_gen.num_classes

base = MobileNetV2(input_shape=IMG_SIZE + (3,),
                   include_top=False,
                   weights="imagenet")
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint("vehicle_mobilenetv2.keras", save_best_only=True),
    ReduceLROnPlateau(patience=2, factor=0.3, verbose=1)
]

model.fit(train_gen,
          epochs=EPOCHS_FROZEN,
          validation_data=test_gen,
          callbacks=callbacks,
          verbose=2)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_gen,
          epochs=EPOCHS_FINE,
          validation_data=test_gen,
          callbacks=callbacks,
          verbose=2)

model.save("vehicle_mobilenetv2.keras")