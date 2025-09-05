import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import pandas as pd

# --- Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Feature extractor
resnet = models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval().to(device)

# --- Load MLP classifier dari HF
@st.cache_resource
def load_mlp_model():
    path = hf_hub_download(repo_id="fae-ld/food-101", filename="model_ntm.keras")
    return load_model(path)

mlp_model = load_mlp_model()

labels = {
    "0": "apple_pie", "1": "baby_back_ribs", "2": "baklava", "3": "beef_carpaccio", "4": "beef_tartare",
    "5": "beet_salad", "6": "beignets", "7": "bibimbap", "8": "bread_pudding", "9": "breakfast_burrito",
    "10": "bruschetta", "11": "caesar_salad", "12": "cannoli", "13": "caprese_salad", "14": "carrot_cake",
    "15": "ceviche", "16": "cheesecake", "17": "cheese_plate", "18": "chicken_curry", "19": "chicken_quesadilla",
    "20": "chicken_wings", "21": "chocolate_cake", "22": "chocolate_mousse", "23": "churros", "24": "clam_chowder",
    "25": "club_sandwich", "26": "crab_cakes", "27": "creme_brulee", "28": "croque_madame", "29": "cup_cakes",
    "30": "deviled_eggs", "31": "donuts", "32": "dumplings", "33": "edamame", "34": "eggs_benedict",
    "35": "escargots", "36": "falafel", "37": "filet_mignon", "38": "fish_and_chips", "39": "foie_gras",
    "40": "french_fries", "41": "french_onion_soup", "42": "french_toast", "43": "fried_calamari", "44": "fried_rice",
    "45": "frozen_yogurt", "46": "garlic_bread", "47": "gnocchi", "48": "greek_salad", "49": "grilled_cheese_sandwich",
    "50": "grilled_salmon", "51": "guacamole", "52": "gyoza", "53": "hamburger", "54": "hot_and_sour_soup",
    "55": "hot_dog", "56": "huevos_rancheros", "57": "hummus", "58": "ice_cream", "59": "lasagna",
    "60": "lobster_bisque", "61": "lobster_roll_sandwich", "62": "macaroni_and_cheese", "63": "macarons", "64": "miso_soup",
    "65": "mussels", "66": "nachos", "67": "omelette", "68": "onion_rings", "69": "oysters",
    "70": "pad_thai", "71": "paella", "72": "pancakes", "73": "panna_cotta", "74": "peking_duck",
    "75": "pho", "76": "pizza", "77": "pork_chop", "78": "poutine", "79": "prime_rib",
    "80": "pulled_pork_sandwich", "81": "ramen", "82": "ravioli", "83": "red_velvet_cake", "84": "risotto",
    "85": "samosa", "86": "sashimi", "87": "scallops", "88": "seaweed_salad", "89": "shrimp_and_grits",
    "90": "spaghetti_bolognese", "91": "spaghetti_carbonara", "92": "spring_rolls", "93": "steak", "94": "strawberry_shortcake",
    "95": "sushi", "96": "tacos", "97": "takoyaki", "98": "tiramisu", "99": "tuna_tartare", "100": "waffles"
}

class_names = [labels[str(i)] for i in range(len(labels))]

# --- Class names
# class_names = ["Class 0", "Class 1", "..."]  # replace sesuai MLP training / bisa ambil json dari HF

def get_embedding(img: Image.Image):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(x)  # [1, 512, 1, 1]
        feat = feat.view(feat.size(0), -1)  # jadi [1, 512]
    return feat.cpu().numpy()

st.title("Food-101 Classifier")
st.info(
    "⚠️ Model ini cuma dilatih untuk **Food-101**. "
    "Kalau kamu upload kucing, motor, atau benda lain, "
    "hasilnya pasti ngawur 💔. "
    "Pastikan gambarnya makanan dari dataset pls"
)

with st.expander("ℹ️ Lihat semua labels Food-101"):
    df_labels = pd.DataFrame({
        "Index": [int(k) for k in labels.keys()],
        "Label": [v for v in labels.values()]
    })
    st.dataframe(df_labels, width=600, height=400)

uploaded_file = st.file_uploader("Upload gambar makanan", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    embedding = get_embedding(img)
    preds = mlp_model.predict(embedding)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]
    label = class_names[pred_idx]

    st.success(f"Prediksi: {label} ({confidence:.2f})")

    # Top-5
    top5_idx = preds.argsort()[-5:][::-1]
    st.subheader("🍽️ Top-5 Predictions")
    for i in top5_idx:
        st.write(f"- {class_names[i]} ({preds[i]:.2f})")