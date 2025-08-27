import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
import gdown
import os

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Penyakit Padi", page_icon="🌾", layout="centered")

st.title("🌾 Klasifikasi Penyakit Daun Padi")

# Nama kelas
class_names = ["Ringan", "Sedang", "Berat"]

# Path model
MODEL_PATH = "no tuning.h5"
FILE_ID = "1vegErZ4PlOpdbZ7l3_cRj_8lz36KYSfQ"  

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Mengunduh model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load model (cache agar tidak berulang)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Fungsi preprocessing gambar
def preprocess_image(image_file):
    img = load_img(image_file, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img)

# Upload gambar
image_file = st.file_uploader("📷 Upload gambar daun padi", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Klasifikasi"):
        img_array = preprocess_image(image_file)

        with st.spinner("⏳ Model sedang memproses..."):
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds, axis=-1)[0]
            confidence = np.max(preds)

        st.success(f"✅ Prediksi: **{class_names[predicted_class]}**")
        st.write(f"📊 Confidence: **{confidence*100:.2f}%**")

