# alzheimer_xai_app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tempfile

IMG_HEIGHT, IMG_WIDTH = 128, 128
MODEL_PATH = 'alzheimer_diagnosis_model.h5'
LAST_CONV_LAYER = 'conv2d_1'

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs, model.get_layer(last_conv_layer_name).output], model.output)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

st.title("🧠 Alzheimer’s Diagnosis with Explainable AI")
st.write("Upload an MRI scan to classify and explain the cognitive condition.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        img_path = tmp_file.name

    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model()
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, predicted_class)
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (IMG_WIDTH, IMG_HEIGHT))
    overlay_img = overlay_heatmap(original_img, heatmap)

    st.subheader(f"Prediction: Class {predicted_class} with {confidence:.2f} confidence")
    st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption="Grad-CAM Explanation", use_column_width=True)

    with st.expander("View Raw Confidence Scores"):
        for i, score in enumerate(preds[0]):
            st.write(f"Class {i}: {score:.4f}")
