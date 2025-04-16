import streamlit as st
from pickle import load
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar modelo
model = load(open("../models/spam_detector.sav", "rb"))

# Interfaz Streamlit
st.title("🔍 Detector de SPAM en URLs")
url_input = st.text_input("Introduce una URL para analizar:")
class_dict = {0: "NO spam ✅", 1: "SPAM 🚨"}
if st.button("Predecir"):
    if url_input.strip() == "":
        st.warning("Por favor, introduce una URL.")
    else:
        
        prediction = model.predict(model)[0]
        st.success(f"Predicción: {class_dict[prediction]}")
