#!/usr/bin/env python3
"""
Breast Cancer Histopathology Classifier - Streamlit WebApp
Thesis Independent Validation Module with XAI (Grad-CAM)
Model: DenseNet121 (transfer learning, fine-tuned), binary sigmoid output.
"""

import sys
import io
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import cv2

# Add src to path for imports (critical for src/ layout)
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st

# Import our utilities
try:
    from utils.model_loader import load_all_models
    from utils.grad_cam import make_gradcam_heatmap, overlay_heatmap, get_last_conv_layer_name
    from utils.preprocessing import load_and_preprocess_image, IMAGE_SIZE
except ImportError as e:
    st.error(f"❌ Failed to import utilities: {e}")
    st.info("💡 Make sure utils/ folder contains model_loader.py, grad_cam.py, and preprocessing.py")
    st.stop()

# ============================================================================
# CONFIG: binary classification labels
# ============================================================================
# Sigmoid output: values close to 0 -> class_names[0], close to 1 -> class_names[1].
# Update if your label encoding was flipped during training (check your
# ImageDataGenerator / dataset class_indices to confirm which is 0 and which is 1).
CLASS_NAMES = ["Benign", "Malignant"]
DECISION_THRESHOLD = 0.5

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="🎗️ Breast Cancer Histopathology Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #d16ba5 0%, #86a8e7 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .stProgress > div > div {
        background-color: #d16ba5;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# APP TITLE & DESCRIPTION
# ============================================================================
st.title("🎗️ Breast Cancer Histopathology Classification System")
st.markdown("""
**Independent Validation WebApp** — Upload a histopathology image (e.g. BreakHis-style patch) to get:
- 🎯 Benign / Malignant prediction with confidence score
- 📊 Probability breakdown
- 🔥 Grad-CAM explainability heatmap
- 📋 Detailed classification insights
""")

st.warning("⚠️ For research/thesis validation purposes only — **not** a clinical diagnostic tool.")

# ============================================================================
# SIDEBAR: MODEL MANAGEMENT
# ============================================================================
with st.sidebar:
    st.header("⚙️ Settings")

    model_dir = st.text_input(
        "📁 Model Directory",
        value="./src/models",
        help="Path to folder containing .keras model files"
    )

    if st.button("🔄 Load Models", type="primary", use_container_width=True):
        with st.spinner("Loading models from disk..."):
            try:
                models = load_all_models(model_dir)
                if models:
                    st.session_state.models = models
                    st.session_state.model_dir = model_dir
                    st.success(f"✅ Loaded {len(models)} model(s)!")
                else:
                    st.error("❌ No models found. Check the directory path and filenames.")
                    st.info("Expected file: `DenseNet121_Transfer_Learning_FineTuned.keras`")
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
                if "models" in st.session_state:
                    del st.session_state.models

    st.divider()

    if "models" in st.session_state and st.session_state.models:
        selected_model_name = st.selectbox(
            "🤖 Select Model",
            options=list(st.session_state.models.keys()),
            index=0,
            key="model_selector"
        )

        model = st.session_state.models[selected_model_name]
        with st.expander("📋 Model Information", expanded=False):
            st.write(f"**Name**: {selected_model_name}")
            st.write(f"**Parameters**: {model.count_params():,}")
            st.write(f"**Input Shape**: {model.input_shape[1:]}")
            st.write(f"**Output Units**: {model.output_shape[-1]} (sigmoid, binary)")
    else:
        st.warning("👈 Load models first to enable prediction")
        selected_model_name = None

    st.divider()

    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### 🔬 Thesis Validation Module

        **Features**:
        - ✅ Independent model validation (no training state)
        - ✅ Grad-CAM explainable AI visualizations
        - ✅ Confidence calibration display

        **Model**: `DenseNet121` (ImageNet transfer learning, fine-tuned)

        **Classes**: Benign vs. Malignant (binary, sigmoid output)
        """)

# ============================================================================
# MAIN CONTENT: IMAGE UPLOAD & PREDICTION
# ============================================================================

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a histopathology image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        help="Supported: JPG, JPEG, PNG, BMP, TIF"
    )

    if uploaded_file is not None:
        try:
            img_array, original_img = load_and_preprocess_image(uploaded_file)

            st.image(original_img, caption="📷 Uploaded Image", use_container_width=True)

            with st.expander("📋 Image Details"):
                st.write(f"- **Dimensions**: {original_img.shape[1]} × {original_img.shape[0]} px")
                st.write(f"- **Channels**: {original_img.shape[2]} (RGB)")
                st.write(f"- **File Size**: {uploaded_file.size / 1024:.1f} KB")
                st.write(f"- **Preprocessed**: Resized to {IMAGE_SIZE}")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            img_array, original_img = None, None
    else:
        st.info("👆 Upload an image to begin analysis")
        img_array, original_img = None, None

with col2:
    if uploaded_file is not None and img_array is not None:
        if "models" in st.session_state and st.session_state.models and selected_model_name:
            st.subheader("🎯 Prediction Results")

            with st.spinner("🔍 Running inference..."):
                try:
                    model = st.session_state.models[selected_model_name]
                    raw_output = model.predict(img_array, verbose=0)
                    malignant_prob = float(np.ravel(raw_output)[0])  # P(class == 1 == Malignant)
                    benign_prob = 1.0 - malignant_prob

                    predicted_class_idx = int(malignant_prob >= DECISION_THRESHOLD)
                    predicted_class_name = CLASS_NAMES[predicted_class_idx]
                    confidence = malignant_prob if predicted_class_idx == 1 else benign_prob

                    print(f"Malignant probability: {malignant_prob:.4f}, Predicted: {predicted_class_name}")

                    # === DISPLAY MAIN PREDICTION ===
                    st.metric(
                        label="🏆 Predicted Class",
                        value=predicted_class_name,
                        delta=f"{confidence * 100:.2f}% confidence"
                    )

                    if confidence >= 0.9:
                        st.progress(float(confidence))
                        st.caption(f"🟢 High confidence: {confidence * 100:.2f}%")
                    elif confidence >= 0.7:
                        st.progress(float(confidence))
                        st.caption(f"🟡 Moderate confidence: {confidence * 100:.2f}%")
                    else:
                        st.progress(float(confidence))
                        st.caption(f"🔴 Low confidence: {confidence * 100:.2f}% — review recommended")

                    # === PROBABILITY BREAKDOWN ===
                    st.markdown("### 📊 Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Class": CLASS_NAMES,
                        "Probability (%)": [f"{benign_prob * 100:.2f}", f"{malignant_prob * 100:.2f}"],
                        "Score": [f"{benign_prob:.4f}", f"{malignant_prob:.4f}"]
                    }).sort_values("Score", ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                    # Store for XAI section
                    st.session_state.last_prediction = {
                        'model_name': selected_model_name,
                        'malignant_prob': malignant_prob,
                        'predicted_idx': predicted_class_idx,
                        'class_names': CLASS_NAMES,
                        'img_array': img_array,
                        'original_img': original_img
                    }

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("💡 Ensure the model input shape matches the preprocessed image")
                    st.session_state.last_prediction = None

        elif "models" not in st.session_state or not st.session_state.models:
            st.info("👈 Load models from the sidebar first")
        else:
            st.info("👈 Select a model from the sidebar to run prediction")

    elif uploaded_file is not None and (img_array is None):
        st.warning("⚠️ Image processing failed. Try a different image format.")

# ============================================================================
# XAI SECTION: GRAD-CAM VISUALIZATION
# ============================================================================
if "last_prediction" in st.session_state and st.session_state.last_prediction:
    st.divider()
    st.subheader("🔥 Explainable AI (Grad-CAM)")

    pred_data = st.session_state.last_prediction
    model = st.session_state.models[pred_data['model_name']]

    with st.spinner("🎨 Generating Grad-CAM heatmap..."):
        try:
            last_conv_layer = get_last_conv_layer_name(model)

            if last_conv_layer:
                heatmap = make_gradcam_heatmap(
                    pred_data['img_array'],
                    model,
                    last_conv_layer,
                    pred_index=None  # ignored for binary sigmoid models
                )

                overlay = overlay_heatmap(pred_data['original_img'], heatmap)

                xai_col1, xai_col2 = st.columns(2)
                with xai_col1:
                    st.image(pred_data['original_img'], caption="Original Image", use_container_width=True)
                with xai_col2:
                    st.image(overlay, caption="Grad-CAM Heatmap 🔥", use_container_width=True)

                predicted_label = pred_data['class_names'][pred_data['predicted_idx']]
                st.caption(f"""
                *🔍 The **red/yellow regions** indicate areas most influential for predicting
                **"{predicted_label}"**.
                Blue/green areas had minimal impact on this prediction.*
                """)

                st.markdown("### 💾 Download Results")
                overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                overlay_pil.save(buf, format='PNG')
                buf.seek(0)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buf.getvalue(),
                        file_name=f"gradcam_{predicted_label}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with col_dl2:
                    import matplotlib.pyplot as plt

                    heatmap_buf = io.BytesIO()
                    plt.figure(figsize=(4, 4))
                    plt.imshow(heatmap, cmap='jet')
                    plt.axis('off')
                    plt.savefig(heatmap_buf, format='PNG', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    heatmap_buf.seek(0)

                    st.download_button(
                        label="📥 Download Raw Heatmap",
                        data=heatmap_buf.getvalue(),
                        file_name=f"heatmap_{predicted_label}.png",
                        mime="image/png",
                        use_container_width=True
                    )

            else:
                st.warning("""
                ⚠️ Could not auto-detect a 4D feature-map layer for Grad-CAM.

                **Fix**: Edit `utils/grad_cam.py` and manually set the target layer,
                e.g. `last_conv_layer = "relu"` (DenseNet121's final feature map layer).
                """)

        except Exception as e:
            st.error(f"❌ Error generating Grad-CAM: {str(e)}")
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                1. Ensure your model has standard Keras convolutional layers
                2. Check that `get_last_conv_layer_name()` finds the correct 4D feature map layer
                3. For custom architectures, manually specify the layer name in `grad_cam.py`
                4. Verify the model was saved with `save_format='keras'`
                """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em; padding: 10px;'>
    <b>🎗️ Breast Cancer Histopathology Classification System</b><br>
    Thesis Independent Validation Module • Built with Streamlit + TensorFlow + Grad-CAM<br>
    <i>For research purposes only • Not for clinical diagnosis</i>
</div>
""", unsafe_allow_html=True)
