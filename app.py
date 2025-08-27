import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
import gdown
import os, zipfile

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Penyakit Padi", page_icon="🌾", layout="centered")

st.title("🌾 Klasifikasi Penyakit Daun Padi")

# Nama kelas
class_names = ["berat", "sedang", "sehat-ringan"]

# ==============================
# Bagian Model
# ==============================
MODEL_PATH = "no_tuning.h5"
MODEL_FILE_ID = "1vegErZ4PlOpdbZ7l3_cRj_8lz36KYSfQ"  # ganti dengan ID modelmu

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Mengunduh model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# Bagian Dataset (3 ZIP)
# ==============================
DATASETS = {
    "berat": "1pcpNJdZ5d9Y2a9zgh6AEQWMXNWDm1KWK",
    "sedang": "1rApUL_cT5zhWqJqgxGFjTLMAQLRhcltg",
    "sehat-ringan": "1vZjaT0GTk5UNgxGS4hzI72xIPTsahApt"  # ganti dengan file ID asli dari Google Drive
}

for name, file_id in DATASETS.items():
    zip_path = f"{name}.zip"
    extract_path = f"dataset/{name}"

    # download zip jika belum ada
    if not os.path.exists(zip_path):
        with st.spinner(f"🔽 Mengunduh dataset {name}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, zip_path, quiet=False)

    # ekstrak jika folder belum ada
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

st.success("✅ Semua dataset berhasil diunduh dan diekstrak!")

# ==============================
# Fungsi preprocessing gambar
# ==============================
def preprocess_image(image_file):
    img = load_img(image_file, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img)

# ==============================
# Upload & Prediksi
# ==============================
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
