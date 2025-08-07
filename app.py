import streamlit as st
from PIL import Image
from utils import predict_species

st.set_page_config(page_title=" Tree Species Classifier", layout="wide")

st.markdown("<h1 style='text-align: center; color: green;'>🌲 Tree Species Classification App 🌳</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload a leaf or tree image to identify its species</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📷 Upload Tree Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        label, confidence = predict_species(img)
        st.success(f" **Prediction:** `{label}`")
        st.info(f" **Confidence:** `{confidence * 100:.2f}%`")
