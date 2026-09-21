import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import json


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "traffic_sign_cnn.keras"
    )

    return model


model = load_model()


# ============================================
# LOAD CLASS NAMES
# ============================================

with open(
    "class_names.json",
    "r"
) as f:

    class_names = json.load(f)


# ============================================
# TITLE
# ============================================

st.title(
    "🚦 Traffic Sign Recognition System"
)

st.write(
    "Upload an image of a traffic sign "
    "and the CNN model will predict the "
    "traffic sign."
)


# ============================================
# IMAGE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================
# PREDICTION
# ============================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    # Resize
    processed_image = image.resize(
        (32, 32)
    )

    # Convert to NumPy
    image_array = np.array(
        processed_image
    )

    # Normalize
    image_array = (
        image_array.astype("float32")
        / 255.0
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_class = np.argmax(
        probabilities
    )

    # Get confidence
    confidence = probabilities[
        predicted_class
    ]

    # Display result
    st.success(
        f"Prediction: "
        f"{class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )