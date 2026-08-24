import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = (224, 224)


def load_and_preprocess_image(image_file, expand=True):
    """
    Load and preprocess an uploaded histopathology image for DenseNet121 inference.

    IMPORTANT: Your model was trained with a specific normalization scheme.
    The default below uses `densenet.preprocess_input` (Caffe-style: zero-centered
    per ImageNet channel means, no [0,1] scaling) — the standard choice when
    fine-tuning `tf.keras.applications.DenseNet121` with ImageNet weights.

    Double-check this against your training notebook before trusting predictions:
      - If you trained with `ImageDataGenerator(rescale=1./255)` -> use OPTION B below.
      - If you trained on raw [0, 255] pixel values (no rescale at all) -> use OPTION C.
    Getting this wrong won't throw an error — it will just silently produce
    unreliable predictions, so verify once and then trust it.
    """
    pil_img = Image.open(image_file).convert('RGB')
    original = np.array(pil_img)

    # Match TF's default bilinear interpolation used during training
    img = pil_img.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    img_array = np.array(img).astype('float32')

    # === OPTION A (default): DenseNet121 ImageNet-style preprocessing ===
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)

    # === OPTION B: rescale to [0, 1] (uncomment if you trained this way) ===
    # img_array = img_array / 255.0

    # === OPTION C: raw [0, 255] values, no normalization ===
    # (leave img_array as-is)

    if expand:
        img_array = np.expand_dims(img_array, axis=0)
    return img_array, original
