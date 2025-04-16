import streamlit as st
st.write('que pasa bro')
from pickle import load
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import re          
model = load(open("../models/spam_detector.sav", "rb"))
@st.cache_resource
def load_vectorizer():
    data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")
    data["label"] = data["label"].map({"spam": 1, "ham": 0})
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data["url"])
    return vectorizer

st.title("DETECTOR SPAM URLS - Model prediction")

val1 = st.sidebar.checkbox("Show Analysis by STATUS", True, key=1)

class_dict = {0: "no spam", 1: "spam"}

if st.button("Predict"):
    prediction = str(model.predict([[val1]])[0])
    pred_class = class_dict[int(prediction)]
    st.write("Prediction:", pred_class)