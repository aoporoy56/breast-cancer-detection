import os
import tensorflow as tf


def load_model(model_path: str, model_name: str):
    """Load a .keras model file. No custom layers are needed for the
    standard DenseNet121 transfer-learning architecture (GAP + Dense head)."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Loaded: {model_name}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None


def load_all_models(model_dir: str):
    """Load all .keras models found in model_dir, with friendlier display names
    for known filenames."""
    models = {}

    known_models = {
        'DenseNet121_Transfer_Learning_FineTuned.keras': 'DenseNet121 (Fine-tuned)',
    }

    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.keras'):
                path = os.path.join(model_dir, filename)
                name = known_models.get(
                    filename,
                    filename.replace('.keras', '').replace('_', ' ')
                )
                model = load_model(path, name)
                if model is not None:
                    models[name] = model

    return models
