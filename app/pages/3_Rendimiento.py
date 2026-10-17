"""Rendimiento desde las tablas de Fernando; no carga ni entrena modelos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from estilo_graficos import AZUL, GRIS, aplicar_plantilla_plotly, color_modelo, nombre_modelo, algoritmo_de

st.set_page_config(page_title="Rendimiento", layout="wide")
st.title("Rendimiento de los modelos")
if not st.toggle("Mostrar información de rendimiento", value=True):
    st.info("La información de rendimiento está oculta.")
    st.stop()

carpeta = RAIZ / "results/tables"
archivos = [carpeta / nombre for nombre in ["08_metricas_prueba.csv", "08_predicciones_prueba.parquet", "08_validacion_por_ofertas.csv"]]
if any(not p.exists() for p in archivos):
    st.warning("Faltan tablas 08 de evaluación. Ejecuta el notebook 08 antes de consultar el rendimiento.")
    st.stop()

@st.cache_data
def leer_tablas(firmas):
    return pd.read_csv(archivos[0]), pd.read_parquet(archivos[1]), pd.read_csv(archivos[2])

metricas, predicciones, ofertas = leer_tablas(tuple(p.stat().st_mtime_ns for p in archivos))
requeridas = {"modelo", "auc", "precision", "sensibilidad", "f1"}
if not requeridas.issubset(metricas.columns) or not {"repeater", "id", "offer"}.issubset(predicciones.columns) or not {"algoritmo", "auc", "split"}.issubset(ofertas.columns):
    st.error("Las tablas 08 no cumplen el contrato: revisa sus encabezados con el responsable de evaluación.")
    st.stop()
modelos = metricas.modelo.tolist()
if any("probabilidad_" + algoritmo_de(m) not in predicciones for m in modelos):
    st.error("Faltan probabilidades de uno o más modelos en las predicciones de prueba.")
    st.stop()
y = predicciones.repeater
probabilidades = predicciones[["probabilidad_" + algoritmo_de(m) for m in modelos]]
if y.isna().any() or not y.isin([0, 1]).all() or not np.isfinite(probabilidades.to_numpy()).all() or not probabilidades.ge(0).all().all() or not probabilidades.le(1).all().all():
    st.error("Las etiquetas o probabilidades de prueba contienen valores inválidos.")
    st.stop()
plantilla = aplicar_plantilla_plotly()
st.caption(f"Semana de prueba: 24–30 de abril de 2013 · {len(y):,} clientes · {y.mean():.2%} positivos. El conjunto de prueba de Kaggle carece de etiquetas.")
mejor = metricas.loc[metricas.auc.idxmax()]
st.success(f"Mayor AUC temporal: {nombre_modelo(mejor.modelo)} ({mejor.auc:.3f}). AUC mide discriminación; no implica que las probabilidades estén calibradas.")
seleccion = st.multiselect("Modelos a comparar", modelos, default=modelos, format_func=nombre_modelo)
if not seleccion:
    st.info("Selecciona al menos un modelo.")
    st.stop()
fig = go.Figure()
for modelo in seleccion:
    prob = predicciones["probabilidad_" + algoritmo_de(modelo)]
    if y.nunique() == 2:
        fpr, tpr, _ = roc_curve(y, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{nombre_modelo(modelo)} · AUC {roc_auc_score(y, prob):.3f}", line_color=color_modelo(modelo)))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar · AUC 0.500", line=dict(color=GRIS, dash="dash")))
fig.update_layout(template=plantilla, title="Curvas ROC · semana de prueba", xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad")
fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, width="stretch")

st.subheader("Errores al cambiar el umbral")
modelo = st.selectbox("Modelo para la matriz de confusión", seleccion, format_func=nombre_modelo)
umbral = st.slider("Probabilidad mínima para clasificar como recompra", 0.0, 1.0, 0.5, 0.01)
clase = predicciones["probabilidad_" + algoritmo_de(modelo)].ge(umbral).astype(int)
matriz = confusion_matrix(y, clase, labels=[0, 1])
fig = go.Figure(go.Heatmap(z=matriz, x=["No recompra", "Recompra"], y=["No recompra", "Recompra"], text=matriz, texttemplate="%{text:,}", colorscale=[[0, "white"], [1, AZUL]], showscale=False))
fig.update_layout(template=plantilla, title=f"{nombre_modelo(modelo)} · umbral {umbral:.2f}", xaxis_title="Predicción", yaxis_title="Real")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width="stretch")
a, b, c = st.columns(3)
a.metric("Precisión", f"{precision_score(y, clase, zero_division=0):.3f}")
b.metric("Sensibilidad", f"{recall_score(y, clase, zero_division=0):.3f}")
c.metric("F1", f"{f1_score(y, clase, zero_division=0):.3f}")
st.caption("Mover el umbral cambia los errores y estas tres métricas; no cambia el AUC. Las métricas oficiales de la tabla inferior usan el umbral elegido en validación para cada modelo.")

metrica = st.selectbox("Métrica oficial a comparar", ["auc", "precision", "sensibilidad", "f1"])
tabla = metricas[metricas.modelo.isin(seleccion)].sort_values(metrica, ascending=False)
fig = go.Figure(go.Bar(x=[nombre_modelo(m) for m in tabla.modelo], y=tabla[metrica], marker_color=[color_modelo(m) for m in tabla.modelo], text=tabla[metrica].map(lambda x: f"{x:.3f}"), textposition="outside"))
fig.update_layout(template=plantilla, title=f"{metrica.upper()} oficial · semana de prueba", yaxis_title=metrica.upper())
fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, width="stretch")
st.dataframe(tabla, hide_index=True)

st.subheader("Generalización a ofertas no vistas")
st.caption("Validación agrupada por oferta: modelos temporales entrenados por división. Es un protocolo distinto de la semana de prueba temporal.")
fig = go.Figure()
for modelo in seleccion:
    grupo = ofertas[ofertas.algoritmo.map(algoritmo_de) == algoritmo_de(modelo)]
    valores = grupo.auc.dropna()
    if valores.empty:
        st.warning(f"No hay AUC agrupada para {nombre_modelo(modelo)}.")
        continue
    fig.add_trace(go.Bar(x=[nombre_modelo(modelo)], y=[valores.mean()], error_y=dict(type="data", array=[valores.std(ddof=1) if len(valores)>1 else 0]), marker_color=color_modelo(modelo), name=nombre_modelo(modelo), text=[f"{valores.mean():.3f} · n={len(valores)}"], textposition="outside"))
fig.update_layout(template=plantilla, title="AUC en ofertas no vistas · media y desviación estándar", showlegend=False, yaxis_title="AUC")
fig.update_yaxes(range=[0, 1]); fig.add_hline(y=.5, line_dash="dash", annotation_text="Azar")
st.plotly_chart(fig, width="stretch")
if {"auc_ofertas_vistas", "auc_ofertas_no_vistas"}.issubset(tabla.columns):
    st.write("En la semana de prueba temporal, por presencia de la oferta en entrenamiento:")
    st.dataframe(tabla[["modelo", "auc_ofertas_vistas", "auc_ofertas_no_vistas"]], hide_index=True)
with st.expander("Divisiones agrupadas por oferta"):
    st.dataframe(ofertas, hide_index=True)
