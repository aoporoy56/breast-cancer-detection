# Breast Cancer Histopathology Classifier

Independent validation Streamlit app for the fine-tuned DenseNet121 model,
adapted from the hair-disease-prediction app structure (Grad-CAM XAI included).

## Setup

```bash
cd breast-cancer-detection
pip install -r requirements.txt
```

Place your model file here (already done if you used the provided zip):

```
src/models/DenseNet121_Transfer_Learning_FineTuned.keras
```

## Run

```bash
streamlit run src/app.py
```

## Before you trust the predictions — check two things

1. **Preprocessing** (`src/utils/preprocessing.py`): defaults to
   `tf.keras.applications.densenet.preprocess_input`. If your training
   notebook used `rescale=1./255` or raw `[0, 255]` pixels instead, switch to
   OPTION B or C in that file — this is the #1 cause of silently-wrong
   predictions with transfer-learning models.
2. **Label order** (`src/app.py`, `CLASS_NAMES`): the model has a single
   sigmoid output, so `CLASS_NAMES = ["Benign", "Malignant"]` assumes
   `0 = Benign, 1 = Malignant`. Confirm this matches your dataset's
   `class_indices` (from `ImageDataGenerator`/`image_dataset_from_directory`)
   — if it's flipped, swap the two entries.

## Structure

```
src/
  app.py                    # Streamlit UI, prediction + Grad-CAM display
  utils/
    model_loader.py         # Loads .keras model(s) from src/models/
    preprocessing.py        # Resize + normalize uploaded images
    grad_cam.py             # Grad-CAM heatmap generation (binary-output aware)
  models/
    DenseNet121_Transfer_Learning_FineTuned.keras
```
