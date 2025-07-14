import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="DzTech Invest AI", layout="centered", page_icon="📈")

# CSS PERSONALIZADO
st.markdown(
    """
    <style>
        body {
            background-color: #0f1117;
            color: #fafafa;
        }
        .logo-title {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        .logo-title img {
            width: 90px;
            margin-right: 1rem;
            border-radius: 8px;
        }
        .logo-title h1 {
            font-size: 2.5rem;
            font-weight: 800;
        }
        .footer {
            text-align: center;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #333;
            color: #888;
        }
        .footer a {
            color: #bbb;
            text-decoration: none;
            margin: 0 10px;
        }
    </style>
    """, unsafe_allow_html=True
)

# LOGO + TÍTULO
st.markdown('''
<div class="logo-title">
    <img src="https://raw.githubusercontent.com/seuusuario/seurepo/main/DZtech_Final.png">
    <h1>DzTech Invest AI</h1>
</div>
''', unsafe_allow_html=True)

st.markdown("🚀 IA que simula decisões de investimento com base no histórico de preços. Selecione um ativo, rode a IA e veja a previsão!")

# INICIALIZA HISTÓRICO NA SESSÃO
if "historico" not in st.session_state:
    st.session_state.historico = []

# LISTA DE ATIVOS DISPONÍVEIS
ativos = {
    "Petrobras (PETR4)": "PETR4.SA",
    "Vale (VALE3)": "VALE3.SA",

