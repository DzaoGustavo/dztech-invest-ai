import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# CONFIG
st.set_page_config(page_title="DzTech Invest AI", layout="centered", page_icon="📈")

# CSS customizado
st.markdown("""
    <style>
        body { background-color: #0f1117; color: #fafafa; }
        .logo-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .logo-img {
            display: block;
            margin: 0 auto 20px;
            width: 180px;
        }
        footer {
            text-align: center;
            font-size: 12px;
            color: #888;
            margin-top: 50px;
        }
        footer a {
            color: #ccc;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# LOGO
st.image("DZtech_Final.png", use_column_width=False, width=180)

# TÍTULO
st.markdown("<div class='logo-title'>📈 DzTech Invest AI</div>", unsafe_allow_html=True)
st.write("Escolha um ativo, rode a IA e veja se ela compraria ou venderia com base na previsão!")

# OPÇÕES
opcoes = {
    "Petrobras (PETR4)": "PETR4.SA",
    "Vale (VALE3)": "VALE3.SA",
    "Magazine Luiza (MGLU3)": "MGLU3.SA",
    "Banco do Brasil (BBAS3)": "BBAS3.SA"
}
ativo = st.selectbox("Selecione o ativo:", list(opcoes.keys()))
ticker = opcoes[ativo]

if st.button("🚀 Rodar IA"):
    st.info(f"🔍 Coletando dados do ativo **{ticker}**...")
    df = yf.download(ticker, period="6mo", interval="1d").dropna()

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:-1], y[:-1])

    pred = model.predict([X.iloc[-1]])[0]
    acc = accuracy_score(y[:-1], model.predict(X[:-1])) * 100

    st.success(f"Acurácia da IA: {acc:.2f}%")

    if pred == 1:
        st.markdown("🟢 **A IA prevê que o preço vai subir.**")
        st.success(f"✅ Ordem simulada: COMPRAR {ticker}")
    else:
        st.markdown("🔴 **A IA prevê que o preço vai cair.**")
        st.error(f"🚫 Ordem simulada: VENDER {ticker}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Preço'))
    st.plotly_chart(fig)

# Rodapé
st.markdown("""
    <footer>
        DzTech Invest AI © 2025<br>
        <a href="#">LinkedIn</a> • <a href="#">Portfólio</a>
    </footer>
""", unsafe_allow_html=True)
