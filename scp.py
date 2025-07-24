import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os

# Download Model if not exists
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1UKOUYNE52nUEDsrDbRgT1Ipds9cbGb1x'  # Replace with your Google Drive file ID
MODEL_PATH = 'skin_cancer_cnn.h5'

if not os.path.exists(MODEL_PATH):
    st.write('Downloading model...')
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)

model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit Interface
st.title("Skin Cancer Prediction")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        result = "Malignant" if prediction > 0.5 else "Benign"
        st.success(f"Prediction: {result}")
