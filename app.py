import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SAT-ML | Analítica Logística", layout="wide", page_icon="📈")

# --- LÓGICA DE DATOS ---
@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'Stock_Inicial': np.random.randint(200, 1000, n),
        'SAT_Score': np.random.uniform(1.5, 5.0, n),
        'Historico_Demanda': np.random.randint(150, 950, n)
    })
    df['Demanda_Real'] = (df['Historico_Demanda'] * (df['SAT_Score'] / 5.0)).astype(int)
    return df

# --- CONTROL DE SESIÓN ---
if 'logueado' not in st.session_state: st.session_state.logueado = False

# --- INTERFAZ SIDEBAR ---
st.sidebar.title("🔐 Acceso al Sistema")
if not st.session_state.logueado:
    user = st.sidebar.text_input("Usuario")
    pwd = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if user == "WCarvajal" and pwd == "1234":
            st.session_state.logueado = True
            st.rerun()
        else:
            st.sidebar.error("Usuario o contraseña incorrectos")
else:
    st.sidebar.success("Sesión activa: W. Carvajal")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logueado = False
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
if st.session_state.logueado:
    st.title("📈 Sistema Predictivo SAT-ML")
    st.markdown("""
    Bienvenido al sistema inteligente de gestión logística. 
    Esta herramienta utiliza Machine Learning para predecir la demanda de productos 
    basándose en el nivel de satisfacción del cliente (SAT) y el stock actual.
    """)
    
    df = generar_datos()
    
    # Simulación de entrenamiento
    X = df[['Stock_Inicial', 'SAT_Score', 'Historico_Demanda']]
    y = df['Demanda_Real']
    modelo = RandomForestRegressor().fit(X, y)
    
    st.subheader("Simulador de Escenarios")
    c1, c2, c3 = st.columns(3)
    s_stock = c1.slider("Stock Físico (Unidades)", 200, 1000, 500)
    s_sat = c2.slider("Nivel de Satisfacción (1-5)", 1.0, 5.0, 4.5)
    s_hist = c3.slider("Histórico Demanda", 150, 950, 600)
    
    if st.button("Calcular Predicción"):
        pred = modelo.predict([[s_stock, s_sat, s_hist]])[0]
        st.metric("Demanda Proyectada", f"{int(pred)} Unidades")
        if pred > s_stock:
            st.warning("⚠️ Alerta: Riesgo de quiebre de stock. Se recomienda reabastecer.")
        else:
            st.success("✅ Stock suficiente para la demanda proyectada.")

else:
    st.title("Bienvenido a SAT-ML Logística")
    st.info("Por favor, inicie sesión en el panel lateral izquierdo para comenzar a utilizar el sistema.")
    st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=2070") # Imagen estética de logística
