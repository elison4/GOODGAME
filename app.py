import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_model():
    with open("modelo_final.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Classificador de Jogos Steam – Bom vs Ruim")

st.subheader("Insira os valores do jogo")

positive = st.number_input("Avaliações Positivas", min_value=0, value=100)
negative = st.number_input("Avaliações Negativas", min_value=0, value=10)
price = st.number_input("Preço (USD)", min_value=0.0, value=10.0)
english = st.selectbox("O jogo está em inglês?", [0, 1])

if st.button("Classificar"):
    X = np.array([[positive, negative, price, english]])
    pred = model.predict(X)[0]

    st.write("---")
    if pred == 1:
        st.success("O modelo prevê que este jogo é **BOM**!")
    else:
        st.error("O modelo prevê que este jogo é **RUIM**.")
    st.write("---")
