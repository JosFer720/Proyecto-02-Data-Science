"""Entrada de la aplicación; ejecutar desde cualquier directorio con Streamlit."""
import streamlit as st

st.set_page_config(page_title="Compradores recurrentes", page_icon="🛒", layout="wide")
st.title("¿Qué clientes vuelven a comprar?")
st.write("Explora el historial de compras, clasifica un cliente y compara cuatro modelos de recompra.")
st.info("La recompra global fue 27.14%. El historial previo con el producto ayuda a distinguir clientes que regresan.")
for titulo, ruta, descripcion in [
    ("1 · Explorar datos", "pages/1_Explorar_datos.py", "Filtra por oferta, cadena y categoría; conoce las variables y los hallazgos."),
    ("2 · Clasificar", "pages/2_Clasificar.py", "Carga transacciones o usa un cliente de ejemplo para comparar probabilidades."),
    ("3 · Rendimiento", "pages/3_Rendimiento.py", "Compara la semana de prueba, ajusta un umbral y consulta ofertas no vistas."),
]:
    st.page_link(ruta, label=titulo)
    st.caption(descripcion)
st.subheader("Datos y evaluación")
st.write("Acquire Valued Shoppers · 160,057 clientes con etiqueta. Entrenamiento: 1 de marzo–16 de abril de 2013; validación: 17–23 de abril; prueba: 24–30 de abril (35,499 clientes).")
st.caption("Los 151,484 clientes del conjunto de prueba de Kaggle no tienen etiqueta y no se usan para medir rendimiento. Las asociaciones exploratorias no demuestran causalidad.")
