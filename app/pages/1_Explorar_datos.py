"""Exploración del conjunto limpio, sin entrenar modelos."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from estilo_graficos import AZUL, BERMELLON, GRIS, aplicar_plantilla_plotly

st.set_page_config(page_title="Explorar datos", layout="wide")
st.title("Explorar los datos limpios")
plantilla = aplicar_plantilla_plotly()

@st.cache_data
def leer_datos(firma):
    return pd.read_csv(RAIZ / "data/processed/dataset_modelo.csv")

ruta = RAIZ / "data/processed/dataset_modelo.csv"
if not ruta.exists():
    st.warning("Falta dataset_modelo.csv. Genera el conjunto limpio con los notebooks 01 y 02.")
    st.stop()
datos = leer_datos(ruta.stat().st_mtime_ns)
global_ = datos.repeater.mean()
filtrado = datos.copy()
with st.sidebar:
    st.header("Filtrar clientes")
    for columna, etiqueta in [("offer", "Oferta"), ("chain", "Cadena"), ("category", "Categoría")]:
        seleccion = st.multiselect(etiqueta, sorted(datos[columna].dropna().unique()))
        if seleccion:
            filtrado = filtrado[filtrado[columna].isin(seleccion)]
    st.caption("Sin selección se incluyen todos los grupos. Los filtros se combinan.")
if filtrado.empty:
    st.info("No hay clientes con esta combinación. Quita uno de los filtros.")
    st.stop()
a, b, c = st.columns(3)
a.metric("Clientes seleccionados", f"{len(filtrado):,}")
b.metric("Recompra seleccionada", f"{filtrado.repeater.mean():.2%}")
c.metric("Referencia global", f"{global_:.2%}")
st.caption("Los identificadores de oferta, cadena y categoría están anonimizados. Los resultados son asociaciones, no efectos causales.")

grupo = filtrado.groupby("nunca_compro_producto").repeater.agg(["mean", "size"]).reset_index()
grupo["Historial"] = grupo.nunca_compro_producto.map({0: "Compró antes el producto", 1: "Sin compra previa del producto"})
fig = px.bar(grupo, x="Historial", y="mean", text=grupo.apply(lambda r: f'{r["mean"]:.1%} · n={r["size"]:,}', axis=1),
             color="Historial", color_discrete_map={"Compró antes el producto": BERMELLON, "Sin compra previa del producto": GRIS},
             title="La recompra se relaciona con el historial del producto", template=plantilla)
fig.update_layout(showlegend=False, yaxis_title="Clientes que recompran (%)", xaxis_title=None)
fig.update_yaxes(tickformat=".0%", rangemode="tozero", range=[0, min(1.15, max(.35, grupo["mean"].max()*1.3))])
fig.add_hline(y=global_, line_dash="dash", line_color=AZUL, annotation_text=f"Global: {global_:.2%}")
st.plotly_chart(fig, width="stretch")
st.caption("Hallazgo del conjunto completo: la recompra pasa de aproximadamente 24.2% sin historial del producto a 42.8% con historial. Las barras se recalculan con los filtros.")

dimension = st.selectbox("Comparar grupos", ["category", "chain", "offer"], format_func=lambda x: {"category": "Categoría", "chain": "Cadena", "offer": "Oferta"}[x])
resumen = filtrado.groupby(dimension).repeater.agg(n="size", tasa="mean")
resumen = resumen.nlargest(15, "n").sort_values("tasa").reset_index()
resumen[dimension] = resumen[dimension].astype(str)
fig = px.bar(resumen, x=dimension, y="tasa", text=resumen.apply(lambda r: f'{r.tasa:.1%} · n={r.n:,.0f}', axis=1), template=plantilla,
             title="La recompra varía entre los 15 grupos con más clientes", color_discrete_sequence=[AZUL])
fig.update_yaxes(tickformat=".0%", rangemode="tozero")
fig.add_hline(y=global_, line_dash="dash", annotation_text="Tasa global")
st.plotly_chart(fig, width="stretch")

manifiesto = RAIZ / "results/modelos/manifiesto.json"
columnas_modelo = set()
if manifiesto.exists():
    for ficha in json.loads(manifiesto.read_text(encoding="utf-8"))["modelos"].values():
        columnas_modelo.update(ficha["columnas_entrada"])
excluidas = {"id", "repeater", "repeattrips"}
variables = [x for x in datos.select_dtypes("number").columns if x not in excluidas]
variables.sort(key=lambda x: (x not in columnas_modelo, x))
variable = st.selectbox("Distribución de una variable (★ usada por un modelo)", variables,
                        format_func=lambda x: ("★ " if x in columnas_modelo else "") + x)
fig = px.histogram(filtrado, x=variable, nbins=40, template=plantilla, color_discrete_sequence=[AZUL], title=f"Distribución de {variable} · clientes")
fig.update_yaxes(title="Número de clientes", rangemode="tozero")
st.plotly_chart(fig, width="stretch")
st.caption(f"Sin dato: {filtrado[variable].isna().sum():,}. Se conservan los valores extremos; consulta unidades y significado en el diccionario.")
st.dataframe(filtrado[variable].describe().rename("Resumen"), width="stretch")
with st.expander("Diccionario y datos seleccionados"):
    diccionario = RAIZ / "results/tables/02_diccionario_dataset_modelo.csv"
    if diccionario.exists():
        st.dataframe(pd.read_csv(diccionario), hide_index=True)
    st.dataframe(filtrado.head(100), hide_index=True)
    st.download_button("Descargar selección CSV", filtrado.to_csv(index=False).encode("utf-8"), "clientes_filtrados.csv", "text/csv")
