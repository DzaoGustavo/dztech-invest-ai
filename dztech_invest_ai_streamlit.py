import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
st.image("DZtech_Final.png", use_container_width=False, width=180)

# TÍTULO
st.markdown("<div class='logo-title'>📈 DzTech Invest AI</div>", unsafe_allow_html=True)
st.write("Escolha um ativo, rode a IA e veja se ela compraria ou venderia com base na previsão!")

# OPÇÕES DE ATIVOS
opcoes = {
    "Petrobras (PETR4)": "PETR4.SA",
    "Vale (VALE3)": "VALE3.SA",
    "Magazine Luiza (MGLU3)": "MGLU3.SA",
    "Banco do Brasil (BBAS3)": "BBAS3.SA",
    "Itaú Unibanco (ITUB4)": "ITUB4.SA",
    "B3 (B3SA3)": "B3SA3.SA",
    "WEG (WEGE3)": "WEGE3.SA",
    "Ambev (ABEV3)": "ABEV3.SA",
    "Lojas Renner (LREN3)": "LREN3.SA",
    "Suzano (SUZB3)": "SUZB3.SA",
    "Eletrobras (ELET3)": "ELET3.SA"
}
ativo_nome = st.selectbox("Selecione o ativo:", list(opcoes.keys()))
ticker = opcoes[ativo_nome]

# BOTÃO DE EXECUÇÃO
if st.button("🚀 Rodar IA"):
    st.info(f"🔍 Coletando dados do ativo **{ticker}**...")
    df = yf.download(ticker, period="6mo", interval="1d").dropna()

    if df.empty or len(df) < 20:
        st.error("⚠️ Dados insuficientes para análise. Tente outro ativo.")
    else:
        # Preparação dos dados
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
        y = df['Target']

        # Dividir dados entre treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test)) * 100

        # Previsão atual
        pred = model.predict([X.iloc[-1]])[0]

        st.success(f"Acurácia da IA nos dados de teste: {acc:.2f}%")

        if pred == 1:
            st.markdown("🟢 **A IA prevê que o preço vai subir.**")
            st.success(f"✅ Ordem simulada: COMPRAR {ticker}")
        else:
            st.markdown("🔴 **A IA prevê que o preço vai cair.**")
            st.error(f"🚫 Ordem simulada: VENDER {ticker}")

        # GRÁFICO
        if not df.empty and 'Close' in df.columns and df['Close'].notnull().sum() > 1:
            preco_inicio = float(df['Close'].iloc[0])
            preco_fim = float(df['Close'].iloc[-1])
            cor = 'limegreen' if preco_fim >= preco_inicio else 'crimson'

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df.index),
                y=df['Close'],
                mode='lines+markers',
                name=ativo_nome,
                line=dict(color=cor, width=3)
            ))

            fig.update_layout(
                title=f"Evolução do preço de fechamento - {ativo_nome}",
                xaxis_title="Data",
                yaxis_title="Preço (R$)",
                template="plotly_dark",
                showlegend=True,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Não foi possível gerar o gráfico.")
        
# Rodapé
st.markdown("""
    <footer>
        DzTech Invest AI © 2025<br>
        <a href="#">LinkedIn</a> • <a href="#">Portfólio</a>
    </footer>
""", unsafe_allow_html=True)
