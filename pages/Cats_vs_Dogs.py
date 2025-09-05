# pages/cats_dogs.py
import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from stqdm import stqdm
from PIL import Image
import numpy as np
import time

CLASS_NAMES = ["Cat (Kucing)", "Dog (Anjing)"]

st.set_page_config(page_title="Cats vs Dogs", page_icon="🐶")

st.header("🐱 vs 🐶 Classifier")
st.write("Upload gambar kucing atau anjing untuk diprediksi.")
st.info("⚠️ Model ini cuma dilatih untuk **Kucing vs Anjing**. "
        "Kalau kamu upload tikus, dinosaurus, atau lainnya, "
        "hasilnya pasti ngawur 💔")

IMG_SIZE = (128, 128)

@st.cache_resource
def load_cats_dogs_model():
    model_path = hf_hub_download(
        repo_id="fae-ld/cats-vs-dogs",
        filename="model.keras"
    )
    return load_model(model_path)

model = load_cats_dogs_model()

uploaded_file = st.file_uploader("Upload gambar", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    img = img.resize(IMG_SIZE)             # resize ke (128,128)
    arr = np.array(img)/255.0              # normalisasi
    arr = np.expand_dims(arr,0)            # tambah batch dimensi (1,128,128,3)


    # kasih progress bar simulasi
    for _ in stqdm(range(30), desc="Memprediksi..."):
        time.sleep(0.05)  # delay biar kelihatan jalan

    pred = model.predict(arr)[0]   # contoh [0.87, 0.13]

    pred_idx = np.argmax(pred)     # ambil index probabilitas terbesar
    label = CLASS_NAMES[pred_idx]
    confidence = pred[pred_idx]

    st.success(f"Prediksi: {label} (confidence={confidence:.2f})")