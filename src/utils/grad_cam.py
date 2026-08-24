import numpy as np
import tensorflow as tf
import cv2
from matplotlib import cm


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap.

    Works for both:
      - multi-class softmax models (pred_index selects the class channel)
      - binary sigmoid models with a single output unit (this project's model) —
        in that case pred_index is ignored and the single logit/probability is used.
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)

        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]

        num_outputs = preds.shape[-1]

        if num_outputs == 1:
            # Binary sigmoid model — only one channel to explain
            class_channel = preds[:, 0]
        else:
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            else:
                pred_index = int(pred_index)
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        if grads is None:
            raise ValueError("Gradient is None. Check layer connectivity or dtype.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cm.jet):
    """
    Overlay heatmap on original image.

    Args:
        img: Original image array (H, W, 3), uint8
        heatmap: Heatmap array (H, W), float [0, 1]
        alpha: Overlay transparency
        colormap: Matplotlib colormap

    Returns:
        Overlay image (H, W, 3), uint8
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_colored = np.uint8(255 * colormap(heatmap_resized)[:, :, :3])
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def get_last_conv_layer_name(model):
    """
    Auto-detect the best layer to use for Grad-CAM: the LAST layer in the model
    that outputs a 4D feature map (batch, H, W, C) — i.e. right before the
    GlobalAveragePooling2D / Flatten that feeds the classification head.

    This is more reliable than a naive "name contains 'conv'" search, since in
    DenseNet121 the true final feature map is produced by a BatchNorm+ReLU
    ('relu') applied AFTER the last 'conv' layer, not by the last conv layer
    itself.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer.name
    return None
