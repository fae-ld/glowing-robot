import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
from pycaret.classification import load_model, predict_model
from extractor.extractor import HFExtractor

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")
st.header("💬 Sentiment Classifier")

st.info(
    "⚠️ Model ini cuma dilatih untuk **5 jenis sentimen**: "
    "`SADNESS`, `ANGER`, `HOPE`, `SUPPORT`, `DISAPPOINTMENT`. "
    "mff kalau hasilnya ngawur"
)

# Labels mapping
labels = {0: "SADNESS", 1: "ANGER", 2: "HOPE", 3: "SUPPORT", 4: "DISAPPOINTMENT"}
LABEL2ID = {'SADNESS': 0, 'ANGER': 1, 'HOPE': 2, 'SUPPORT': 3, 'DISAPPOINTMENT': 4}

@st.cache_resource
def load_pycaret_model():
    path = hf_hub_download("fae-ld/sentiment-classifier", "model.pkl")
    model = load_model(path.replace('.pkl', ''))  # <- pakai PyCaret load_model
    return model

model = load_pycaret_model()
extractor = HFExtractor()

# Input teks
text_input = st.text_area("Masukkan teks untuk analisis sentimen", "")

if st.button("Prediksi"):
    if not text_input.strip():
        st.warning("Isi teks dulu ya 📝")
    else:
        df = pd.DataFrame({"text": [text_input]})
        
        features = extractor.transform(df)
        
        result = predict_model(model, data=features)
        
        pred_label_id = result['prediction_label'][0]
        prediction = labels[pred_label_id]
        conf = result['prediction_score'][0]
        
        st.success(f"Prediksi: {prediction} (confidence={conf:.2f})")