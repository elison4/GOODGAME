import streamlit as st
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

# Entradas do usuário
positive = st.number_input("Avaliações Positivas", min_value=0, value=100)
negative = st.number_input("Avaliações Negativas", min_value=0, value=10)
price = st.number_input("Preço (USD)", min_value=0.0, value=10.0)
required_age = st.number_input("Idade mínima recomendada", min_value=0, max_value=21, value=0)
release_year = st.number_input("Ano de lançamento", min_value=1980, max_value=2030, value=2020)
score = st.number_input("Score (positivas / total)", min_value=0.0, max_value=1.0, value=0.8)

if st.button("Classificar"):
    X = np.array([[positive, negative, price, required_age, release_year, score]])
    pred = model.predict(X)[0]

    st.write("---")
    if pred == 1:
        st.success("O modelo prevê que este jogo é **BOM**! 🎉")
    else:
        st.error("O modelo prevê que este jogo é **RUIM**. 😕")
    st.write("---")
