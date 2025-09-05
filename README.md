# Vehicle-Type Classifier (CNN) 

Lightweight CNN that classifies images into **Car, Bicycle, Scooter, Bus** for traffic-monitoring use cases.

The project began with a **hand-built CNN trained from scratch** on a street-scene dataset of about 170 images per class.
Despite heavy data-augmentation, this baseline topped-out at **~58% validation accuracy**, which aren't satisfactory results. It was evident that it was under-fitting, so I had to try something else.

To overcome this ceiling, the pipeline was switched to **transfer-learning with a MobileNet V2 backbone**:

* freeze ImageNet-pretrained layers, train only a small GAP → Dropout → Dense head for 10 epochs,
* unfreeze the top 20 MobileNet layers and fine-tune 5 more epochs at a low learning-rate.

That simple change launched the model to an incredible **~91% validation accuracy**. This result is acceptable if we have in mind that the images are mostly in a busy urban environment, and that the dataset isn't that big. By adding more relevant pictures to the dataset, we could possibly achieve **95% or even higher**.

## Project goals

* The initial goal of this project is learning and understanding how image classification works by using neural networks
* Later on, it became finding a better way to get usable results on a small, noisy dataset like mine 
* Finally, it came down to providing a fully reproducible pipeline, with real world applicability

## Repository layout

vehicle-cnn/  
│  
├── **dataset/**          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  raw, unedited images from the web(4 categories)  
├── **dataset_split/**          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# train/ and val/ sub-folders (4 categories each)  
├── **dataset_updated/**          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# cleaned, renamed and resized JPGs (4 categories)  
├── **examples/**               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# test images  
├── **scripts/**  
│   ├── **check_images.py**     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# finds corrupt files<br>
│   ├── **convert_images_to_jpg.py**     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# converts all image types to JPEG<br>
│   ├── **predict_folder.py**     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# script that tests out our trained model<br>
│   ├── **rename_images.py**     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# every image gets a uniform name, depending on it's class  
│   ├── **resize_images.py**     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# resizes all images uniformly <br>
│   ├── **scrape_bing_images.py**     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Bing image scraper   
│   ├── **split_dataset.py** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# shuffles data into train/val (80:20)<br>
│   ├── **train_mobilenetv2.py**       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# final transfer-learning pipeline  
│   └── **train_model.py**    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# first CNN (from scratch)  
├── **requirements.txt**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Python dependencies 
├── **vehicle_cnn.keras**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# from scratch CNN report<br> 
├── **vehicle_mobilenetv2.keras**  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# MobileNetV2 report<br>
└── **README.md**  

## Requirements  
```bash  
python -m venv venv  
source venv/bin/activate      # Windows: venv\Scripts\activate  
pip install -r requirements.txt  
(TensorFlow 2.15+, Pillow, numpy, etc.)
```

## Quick start (when data is ready)

```bash
# train the transfer-learning model (10 + 5 epochs, batch 16)
python .\scripts\train_mobilenetv2.py

# run the predictions on the examples folder 
python .\scripts\predict_folder.py
```

## Step-by-step guide

### 1. Scrape Bing for images and manually add/remove them

**Script:** ` scrape_bing_images.py`

* Scrapes Bing for images matching the input keywords
* Manually remove images that aren't appropriate
* Download more images manually from websites like **unsplash.com**

### 2. Convert all images to JPEG

**Script:** `convert_images_to_jpg.py`

* Converts all images from `dataset/` to `.jpg`  
* Creates a new folder (for every category) and places the converted images there: `dataset_updated/`  
* Keeps the original folder structure by vehicle type

### 3. Resize all images

**Script:** `resize_images.py`

* Resize all images to **224x224**
* This resolution is light-weight and retains enough data, so it's the go-to solution
* Update and save them in `dataset_updated/`

### 4. Rename all images

**Script:** `rename_images.py`

* Renames all images to a uniform format
* They're named like this: vehicle_type_five_digit_number (e.x. **car_00023**) 
* Update and save them in `dataset_updated/`

### 5. Split Dataset

**Script:** `split_dataset.py`

* The content from `dataset_updated` is split into training and validation sets
* A split ratio of **80:20** is used
* Creates new folder `dataset_split` with two subfolders: `train/` and `val/`

### 6. Check images

**Script** `check_images.py`

* Checks the `dataset_split/` directory to see if any corrupt or non-image files have gone through
* If it finds something, it tells us which ones are faulty
* If nothing is found or the corrupt ones are deleted, we can start training the model

### 7. Train the model

**Script** `train_model.py` and `train_mobilenetv2.py`

* `train_model.py` is the initial script and `train_mobilenetv2.py` is the new one with better results
* `dataset_split/train/` and `dataset_split/val/` are loaded
* They build and train a **CNN** (Convolutional Neural Network)
* Finally, they save the trained models to `vehicle_cnn.keras` and `vehicle_mobilenetv2.keras` respectively

### 8. Test the model on a folder of images

**Script** `predict_folder.py`

* Loads `vehicle_mobilenetv2.keras` by default
* Prints the predicted class and it's confidence for every file in the folder
* You can also do a single-image test by passing a parameter

## Model Performance

For `train_model.py` after a bunch of tweaking, these were the best results I got:

```commandline
Epoch 1/30
68/68 - 21s - 311ms/step - accuracy: 0.2852 - loss: 1.4264 - val_accuracy: 0.3139 - val_loss: 1.3309 - learning_rate: 1.0000e-03
Epoch 2/30
68/68 - 20s - 294ms/step - accuracy: 0.3370 - loss: 1.3365 - val_accuracy: 0.2993 - val_loss: 1.4743 - learning_rate: 1.0000e-03
Epoch 3/30
68/68 - 20s - 296ms/step - accuracy: 0.3889 - loss: 1.3577 - val_accuracy: 0.4234 - val_loss: 1.2569 - learning_rate: 1.0000e-03
Epoch 4/30
68/68 - 21s - 302ms/step - accuracy: 0.4019 - loss: 1.2806 - val_accuracy: 0.4526 - val_loss: 1.2466 - learning_rate: 1.0000e-03
Epoch 5/30
68/68 - 21s - 302ms/step - accuracy: 0.4519 - loss: 1.2577 - val_accuracy: 0.4599 - val_loss: 1.1839 - learning_rate: 1.0000e-03
Epoch 6/30
68/68 - 20s - 291ms/step - accuracy: 0.4185 - loss: 1.2398 - val_accuracy: 0.4599 - val_loss: 1.1904 - learning_rate: 1.0000e-03
Epoch 7/30
68/68 - 21s - 310ms/step - accuracy: 0.4500 - loss: 1.2192 - val_accuracy: 0.4891 - val_loss: 1.1488 - learning_rate: 1.0000e-03
Epoch 8/30
68/68 - 20s - 290ms/step - accuracy: 0.5000 - loss: 1.2116 - val_accuracy: 0.4599 - val_loss: 1.1860 - learning_rate: 1.0000e-03
Epoch 9/30
68/68 - 20s - 299ms/step - accuracy: 0.5222 - loss: 1.1466 - val_accuracy: 0.5109 - val_loss: 1.1008 - learning_rate: 1.0000e-03
Epoch 10/30
68/68 - 20s - 301ms/step - accuracy: 0.5148 - loss: 1.1394 - val_accuracy: 0.5182 - val_loss: 1.0676 - learning_rate: 1.0000e-03
Epoch 11/30
68/68 - 20s - 300ms/step - accuracy: 0.5315 - loss: 1.0774 - val_accuracy: 0.5766 - val_loss: 0.9622 - learning_rate: 1.0000e-03
Epoch 12/30
68/68 - 20s - 291ms/step - accuracy: 0.5148 - loss: 1.1151 - val_accuracy: 0.5255 - val_loss: 1.0275 - learning_rate: 1.0000e-03
Epoch 13/30
68/68 - 21s - 303ms/step - accuracy: 0.5130 - loss: 1.0800 - val_accuracy: 0.5474 - val_loss: 0.9533 - learning_rate: 1.0000e-03
Epoch 14/30
68/68 - 22s - 322ms/step - accuracy: 0.5241 - loss: 1.0681 - val_accuracy: 0.5839 - val_loss: 0.9301 - learning_rate: 1.0000e-03
Epoch 15/30
68/68 - 20s - 290ms/step - accuracy: 0.5481 - loss: 1.0094 - val_accuracy: 0.4672 - val_loss: 1.0510 - learning_rate: 1.0000e-03
Epoch 16/30
68/68 - 20s - 298ms/step - accuracy: 0.5981 - loss: 0.9891 - val_accuracy: 0.5401 - val_loss: 0.9120 - learning_rate: 1.0000e-03
Epoch 17/30
68/68 - 20s - 292ms/step - accuracy: 0.5741 - loss: 0.9545 - val_accuracy: 0.5547 - val_loss: 0.9304 - learning_rate: 1.0000e-03
Epoch 18/30
68/68 - 20s - 292ms/step - accuracy: 0.5889 - loss: 0.9864 - val_accuracy: 0.5401 - val_loss: 0.9840 - learning_rate: 1.0000e-03
Epoch 19/30
Epoch 19: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
68/68 - 20s - 291ms/step - accuracy: 0.5907 - loss: 0.9531 - val_accuracy: 0.5912 - val_loss: 0.9448 - learning_rate: 1.0000e-03
Epoch 20/30
68/68 - 20s - 294ms/step - accuracy: 0.6148 - loss: 0.9060 - val_accuracy: 0.5693 - val_loss: 0.9295 - learning_rate: 3.0000e-04
Epoch 21/30
68/68 - 20s - 298ms/step - accuracy: 0.6019 - loss: 0.8956 - val_accuracy: 0.5839 - val_loss: 0.8956 - learning_rate: 3.0000e-04
Epoch 22/30
68/68 - 20s - 290ms/step - accuracy: 0.6204 - loss: 0.8621 - val_accuracy: 0.5985 - val_loss: 0.9033 - learning_rate: 3.0000e-04
Epoch 23/30
68/68 - 20s - 293ms/step - accuracy: 0.6222 - loss: 0.8557 - val_accuracy: 0.5766 - val_loss: 0.9079 - learning_rate: 3.0000e-04
Epoch 24/30
Epoch 24: ReduceLROnPlateau reducing learning rate to 9.000000427477062e-05.
68/68 - 20s - 292ms/step - accuracy: 0.6722 - loss: 0.7777 - val_accuracy: 0.5620 - val_loss: 0.9348 - learning_rate: 3.0000e-04
Epoch 25/30
68/68 - 21s - 304ms/step - accuracy: 0.6685 - loss: 0.7837 - val_accuracy: 0.5766 - val_loss: 0.8975 - learning_rate: 9.0000e-05
Epoch 26/30
68/68 - 20s - 293ms/step - accuracy: 0.6537 - loss: 0.7802 - val_accuracy: 0.5766 - val_loss: 0.8985 - learning_rate: 9.0000e-05
```

Which achieved a **validation accuracy** of just under **58%** in 26 epochs, with two learning rate reductions.

The new and improved script `train_mobilenetv2.py` achieved much better results:

```commandline
Epoch 1/10
34/34 - 28s - 817ms/step - accuracy: 0.5704 - loss: 1.0498 - val_accuracy: 0.8467 - val_loss: 0.4854 - learning_rate: 1.0000e-03
Epoch 2/10
34/34 - 16s - 460ms/step - accuracy: 0.8130 - loss: 0.4730 - val_accuracy: 0.8978 - val_loss: 0.3342 - learning_rate: 1.0000e-03
Epoch 3/10
34/34 - 15s - 456ms/step - accuracy: 0.8815 - loss: 0.3362 - val_accuracy: 0.9051 - val_loss: 0.2852 - learning_rate: 1.0000e-03
Epoch 4/10
34/34 - 16s - 465ms/step - accuracy: 0.9019 - loss: 0.2883 - val_accuracy: 0.8905 - val_loss: 0.2789 - learning_rate: 1.0000e-03
Epoch 5/10
34/34 - 16s - 462ms/step - accuracy: 0.9111 - loss: 0.2490 - val_accuracy: 0.9197 - val_loss: 0.2552 - learning_rate: 1.0000e-03
Epoch 6/10
34/34 - 15s - 456ms/step - accuracy: 0.9130 - loss: 0.2205 - val_accuracy: 0.9051 - val_loss: 0.2828 - learning_rate: 1.0000e-03
Epoch 7/10
34/34 - 17s - 492ms/step - accuracy: 0.9352 - loss: 0.1980 - val_accuracy: 0.9051 - val_loss: 0.2508 - learning_rate: 1.0000e-03
Epoch 8/10
34/34 - 16s - 472ms/step - accuracy: 0.9426 - loss: 0.1745 - val_accuracy: 0.9124 - val_loss: 0.2306 - learning_rate: 1.0000e-03
Epoch 9/10
34/34 - 16s - 458ms/step - accuracy: 0.9315 - loss: 0.1656 - val_accuracy: 0.9197 - val_loss: 0.2230 - learning_rate: 1.0000e-03
Epoch 10/10
34/34 - 15s - 448ms/step - accuracy: 0.9407 - loss: 0.1791 - val_accuracy: 0.9124 - val_loss: 0.2269 - learning_rate: 1.0000e-03
Epoch 1/5
34/34 - 24s - 707ms/step - accuracy: 0.8870 - loss: 0.3316 - val_accuracy: 0.9124 - val_loss: 0.2525 - learning_rate: 1.0000e-04
Epoch 2/5
Epoch 2: ReduceLROnPlateau reducing learning rate to 2.9999999242136255e-05.
34/34 - 17s - 508ms/step - accuracy: 0.9481 - loss: 0.1654 - val_accuracy: 0.8905 - val_loss: 0.3341 - learning_rate: 1.0000e-04
Epoch 3/5
34/34 - 17s - 498ms/step - accuracy: 0.9352 - loss: 0.1765 - val_accuracy: 0.8905 - val_loss: 0.3503 - learning_rate: 3.0000e-05
Epoch 4/5
Epoch 4: ReduceLROnPlateau reducing learning rate to 8.999999772640877e-06.
34/34 - 17s - 499ms/step - accuracy: 0.9648 - loss: 0.1150 - val_accuracy: 0.9124 - val_loss: 0.3606 - learning_rate: 3.0000e-05
```

A **validation accuracy** of slightly more than **91%** is a significant jump, which justifies the use of this training model instead of the from-scratch one. It managed to get these results in 10 epochs, plus 4 low-learning ones after freezing.

## Model Testing

To test out the trained model on a set of images (from the `examples/` directory), we're going to use this command:

```commandline
python .\scripts\predict_folder.py
```
And for a set of 7 photos from `examples/` we got the following results:
```
IK-202_GSP_Beograd.jpg         -> bus      (99.54%)
Ikarbus_IK-112LE_GSP_Beograd-3208.jpg -> bus      (99.86%)
images.jfif                    -> bicycle  (99.18%)
IMG_8989-scaled.jpg            -> car      (99.89%)
Porsche-911-Carrera-T.jpg      -> car      (99.93%)
s-l1200.jpg                    -> scooter  (99.79%)
unnamed.jpg                    -> scooter  (66.27%)
x5_5.jpg                       -> car      (86.91%)
```
The model managed to predict everything right, even with images with higher noise.

## How to run the `vehicle-cnn` project

Make sure you are in the project **root** folder, and follow the following steps to run the full pipeline.

First you need to run all of the preprocessing scripts in this order:

```bash
python .\scripts\scrape_bing_images.py
python .\scripts\convert_images_to_jpg.py
python .\scripts\resize_images.py
python .\scripts\rename_images.py
python .\scripts\split_dataset.py
```
This will pull an initial dataset of images for every category (you might need to delete unsuitable ones and download some manually for a larger, more complete dataset). After that all images will be converted to JPEG, resized and renamed. Finally, they will be split into two new datasets: `train` and `val`.
At that point, we may want to check if there are some corrupted or non-image files left in out `dataset_split/` directory, and to do that you can execute:

```python .\scripts\check_images.py```

You're going to have to create and activate a virtual environment, like this:

```bash
python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # Linux/MacOS
```
You also need all of the required libraries to successfully run this project

```bash
pip install tensorflow
pip install Pillow
```

Now you can finally train the model. I recommend using the now one:

```bash
    python .\scripts\train_mobilenetv2.py
```

But if you ever want to try, for testing or educational purposes, the old script is still there, and you can run with:

```bash
    python .\scripts\train_model.py
```

To test out the models predicting power you can populate the `examples/` directory with your own images, and run this:

```bash
    python .\scripts\predict_folder.py
```

You can also try it out on another folder or single image by adding the Path as a parameter.

# Conclusion

Building the **Vehicle-CNN** project gave me practical insight into every stage of an **image-classification** workflow. I scraped and cleaned a raw web dataset, automated conversion, resizing, renaming and train/validation splitting. But I learned the most by trying and comparing two modeling strategies: a small **CNN trained from scratch** and a **MobileNet V2 network** reused through **transfer-learning**. I managed to get a more than **33** percent (58 -> 91) increase in **validation accuracy**, by using the latter and careful fine-tuning. The result i got with `train_mobilenetv2.py` makes the script fully usable in a real-world setting. Future work could be adding more relevant images to the dataset to push the **validation accuracy** to **95+%** 

## License

---

#### Pavle Lukić, Fakultet Organizacionih Nauka
Feel free to use and modify this software!
Feel free to use and modify this software!