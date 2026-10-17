"""Página Clasificar: ingresar un cliente nuevo y ver la predicción de cada modelo.

Cubre el ingreso de datos nuevos y el requisito b de la guía: ver el resultado de
cada algoritmo por separado o de todos. El preprocesamiento corre por detrás con
entrada_app.py y los modelos se cargan congelados: la página nunca los entrena.

Para probarla sola, desde la raíz del repositorio:
    streamlit run app/pages/2_Clasificar.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from config import DATASET_MODELO_PATH, TABLES_DIR  # noqa: E402
from entrada_app import EntradaInvalida, leer_transacciones, ofertas_disponibles, preparar_cliente  # noqa: E402
from estilo_graficos import (  # noqa: E402
    AZUL,
    GRIS_CLARO,
    ORDEN_MODELOS,
    TEXTO_SECUNDARIO,
    algoritmo_de,
    aplicar_plantilla_plotly,
    color_modelo,
    nombre_modelo,
)
from modelos_congelados import ModeloCongeladoError, cargar_modelos  # noqa: E402
from verificar_entrada_app import RUTA_CLIENTES, RUTA_TRANSACCIONES, RUTA_VERIFICACION  # noqa: E402

st.set_page_config(page_title="Clasificar un cliente", page_icon="🛒", layout="wide")
PLANTILLA = aplicar_plantilla_plotly()

# ---------------------------------------------------------------------------
# Datos que se cargan una sola vez
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando los modelos congelados...")
def modelos_congelados() -> dict:
    modelos = cargar_modelos(version=2, estricto=True)
    orden = {algoritmo: i for i, algoritmo in enumerate(ORDEN_MODELOS)}
    return dict(sorted(modelos.items(), key=lambda par: orden[algoritmo_de(par[0])]))


@st.cache_data
def clientes_ejemplo() -> tuple[pd.DataFrame, pd.DataFrame]:
    clientes = pd.read_csv(RUTA_CLIENTES)
    transacciones = pd.read_csv(RUTA_TRANSACCIONES)
    lineas = transacciones.groupby("id").size().rename("lineas")
    clientes = clientes.join(lineas, on="id")
    return clientes, transacciones


@st.cache_data
def datos_de_referencia() -> dict:
    """Mercados conocidos y el hallazgo central del análisis exploratorio."""
    datos = pd.read_csv(DATASET_MODELO_PATH, usecols=["market", "repeater", "nunca_compro_producto"])
    tasas = datos.groupby("nunca_compro_producto")["repeater"].agg(["mean", "size"])
    return {
        "mercados": sorted(datos["market"].unique().tolist()),
        "tasa_con_producto": float(tasas.loc[0, "mean"]),
        "clientes_con_producto": int(tasas.loc[0, "size"]),
        "tasa_sin_producto": float(tasas.loc[1, "mean"]),
        "clientes_sin_producto": int(tasas.loc[1, "size"]),
    }


@st.cache_data
def auc_de_prueba() -> dict:
    metricas = pd.read_csv(TABLES_DIR / "08_metricas_prueba.csv")
    return dict(zip(metricas["algoritmo"], metricas["auc"]))


@st.cache_data
def resumen_verificacion() -> dict | None:
    if not RUTA_VERIFICACION.exists():
        return None
    tabla = pd.read_csv(RUTA_VERIFICACION)
    diferencias = [c for c in tabla.columns if c.startswith("diferencia_")]
    return {
        "clientes": len(tabla),
        "coinciden": int(tabla["coincide"].sum()),
        "variables": int(tabla["variables_comparadas"].iloc[0]),
        "diferencia_maxima": float(tabla[diferencias].to_numpy().max()),
    }


def porcentaje(valor: float, decimales: int = 1) -> str:
    return f"{valor * 100:.{decimales}f}%"


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("Clasificar un cliente nuevo")
st.markdown(
    "Ingresa las compras de un cliente y la oferta que quieres hacerle. La aplicación prepara sus datos "
    "igual que en el entrenamiento y estima con cada modelo la probabilidad de que **vuelva a comprar "
    "el producto** después de usar el cupón."
)

try:
    modelos = modelos_congelados()
except ModeloCongeladoError as error:
    st.error(
        "No se pudieron cargar los modelos. Instala las versiones de `requirements.txt` "
        f"y vuelve a abrir la aplicación.\n\nDetalle: {error}"
    )
    st.stop()

referencia = datos_de_referencia()
ofertas = ofertas_disponibles()

# ---------------------------------------------------------------------------
# 1. Datos del cliente
# ---------------------------------------------------------------------------

st.header("1. Datos del cliente")
origen = st.radio(
    "¿De dónde vienen las compras del cliente?",
    ["Un cliente de ejemplo", "Subir un archivo de compras"],
    horizontal=True,
)

transacciones = None
real = None
if origen == "Un cliente de ejemplo":
    clientes, todas = clientes_ejemplo()
    st.caption(
        f"Son {len(clientes)} clientes reales de la semana de prueba, del 24 al 30 de abril de 2013. "
        "Los modelos no los vieron durante el entrenamiento."
    )
    etiquetas = {
        fila.id: f"Cliente {fila.id}: oferta {fila.offer}, {fila.lineas:,} líneas de compra"
        for fila in clientes.itertuples(index=False)
    }
    id_cliente = st.selectbox("Cliente", list(etiquetas), format_func=etiquetas.get)
    cliente = clientes.set_index("id").loc[id_cliente]
    transacciones = todas[todas["id"] == id_cliente]
    oferta = int(cliente["offer"])
    fecha_oferta = pd.Timestamp(cliente["offerdate"])
    mercado = int(cliente["market"])
    real = bool(cliente["repeater"])

    columnas = st.columns(3)
    columnas[0].markdown(f"**Oferta**  \n{ofertas.set_index('offer').loc[oferta, 'etiqueta']}")
    columnas[1].markdown(f"**Fecha de la oferta**  \n{fecha_oferta:%d/%m/%Y}")
    columnas[2].markdown(f"**Mercado**  \n{mercado}")
    st.download_button(
        "Descargar las compras de este cliente en CSV",
        transacciones.to_csv(index=False).encode("utf-8"),
        file_name=f"compras_cliente_{id_cliente}.csv",
        mime="text/csv",
        help="Sirve como modelo del formato para probar la opción de subir un archivo.",
    )
else:
    st.caption(
        "El archivo debe tener las columnas de transactions.csv de la competencia: chain, category, company, "
        "brand, date, purchasequantity y purchaseamount, con una fila por línea de compra de un solo cliente. "
        "Puedes descargar un ejemplo desde la opción anterior."
    )
    archivo = st.file_uploader("Archivo CSV con las compras del cliente", type="csv")
    if archivo is not None:
        try:
            transacciones = leer_transacciones(archivo)
        except EntradaInvalida as error:
            st.error(str(error))
    fecha_sugerida = pd.Timestamp("2013-04-24")
    if transacciones is not None and "date" in transacciones.columns:
        fechas = pd.to_datetime(transacciones["date"].astype(str), format="%Y-%m-%d", errors="coerce")
        if fechas.notna().any():
            fecha_sugerida = fechas.max() + pd.Timedelta(days=1)

    columnas = st.columns(3)
    oferta = columnas[0].selectbox(
        "Oferta que se le hará",
        ofertas["offer"].tolist(),
        format_func=ofertas.set_index("offer")["etiqueta"].get,
    )
    fecha_oferta = pd.Timestamp(
        columnas[1].date_input(
            "Fecha de la oferta",
            value=fecha_sugerida.date(),
            help="Solo se usan las compras anteriores a esta fecha.",
            format="DD/MM/YYYY",
        )
    )
    mercado = columnas[2].selectbox("Mercado", referencia["mercados"])

if transacciones is None:
    st.info("Elige un cliente de ejemplo o sube un archivo para continuar.")
    st.stop()

try:
    entrada = preparar_cliente(transacciones, oferta=oferta, fecha_oferta=fecha_oferta, market=mercado)
except EntradaInvalida as error:
    st.error(str(error))
    st.stop()

for aviso in entrada.avisos:
    st.warning(aviso)

fila = entrada.fila.iloc[0]

# ---------------------------------------------------------------------------
# 2. Ficha del cliente
# ---------------------------------------------------------------------------

st.header("2. Ficha del cliente")
st.caption(
    f"Se usaron {entrada.lineas_usadas:,} de {entrada.lineas_recibidas:,} líneas de compra, "
    f"todas de la cadena {entrada.cadena} y anteriores a la oferta."
)

compro_producto = int(fila["n_compras_producto"]) > 0
metricas = st.columns(4)
metricas[0].metric("Visitas a la tienda", f"{int(fila['n_visitas']):,}")
metricas[1].metric("Gasto total", f"${fila['gasto_total']:,.2f}")
metricas[2].metric(
    "Días desde la última compra",
    "Sin compras" if pd.isna(fila["dias_ultima_compra"]) else f"{int(fila['dias_ultima_compra'])}",
)
metricas[3].metric(
    "¿Ya compraba el producto?",
    f"Sí, {int(fila['n_compras_producto'])} veces" if compro_producto else "No",
)
metricas = st.columns(4)
metricas[0].metric("Compras de la categoría", f"{int(fila['n_compras_categoria']):,}")
metricas[1].metric("Compras de la marca", f"{int(fila['n_compras_marca']):,}")
metricas[2].metric(
    "Ticket promedio por visita",
    "Sin compras" if pd.isna(fila["ticket_promedio"]) else f"${fila['ticket_promedio']:,.2f}",
)
metricas[3].metric(
    "Días de historial",
    "Sin compras" if pd.isna(fila["antiguedad_dias"]) else f"{int(fila['antiguedad_dias']):,}",
)
with st.expander("Ver todas las variables que reciben los modelos"):
    variables = entrada.fila.drop(columns=["id"]).T.rename(columns={0: "valor"})
    variables["valor"] = variables["valor"].astype(str)
    st.dataframe(variables, width="stretch")

# ---------------------------------------------------------------------------
# 3. Resultado de los modelos
# ---------------------------------------------------------------------------

st.header("3. Probabilidad de que vuelva a comprar")
vista = st.radio("Mostrar", ["Los cuatro modelos", "Un solo modelo"], horizontal=True)
nombres = list(modelos)
if vista == "Un solo modelo":
    elegido = st.selectbox("Modelo", nombres, format_func=nombre_modelo)
    nombres = [elegido]

auc = auc_de_prueba()
resultados = []
for nombre in nombres:
    modelo = modelos[nombre]
    probabilidad = float(modelo.predecir_proba(entrada.fila)[0])
    resultados.append(
        {
            "nombre": nombre,
            "modelo": nombre_modelo(nombre),
            "probabilidad": probabilidad,
            "umbral": modelo.umbral,
            "recompra": probabilidad >= modelo.umbral,
            "auc": auc.get(algoritmo_de(nombre)),
        }
    )
resultados = pd.DataFrame(resultados).iloc[::-1]  # el primer modelo queda arriba en la gráfica

positivos = int(resultados["recompra"].sum())
if len(resultados) == 1:
    r = resultados.iloc[0]
    titulo = f"{r['modelo']}: {porcentaje(r['probabilidad'], 0)} de probabilidad de recompra"
else:
    titulo = f"{positivos} de {len(resultados)} modelos predicen que volverá a comprar"

figura = go.Figure()
figura.add_trace(
    go.Bar(
        x=resultados["probabilidad"],
        y=resultados["modelo"],
        orientation="h",
        marker_color=[color_modelo(n) for n in resultados["nombre"]],
        customdata=resultados[["umbral", "auc"]].to_numpy(),
        hovertemplate=(
            "<b>%{y}</b><br>Probabilidad: %{x:.1%}<br>Umbral del modelo: %{customdata[0]:.1%}"
            "<br>AUC en la semana de prueba: %{customdata[1]:.3f}<extra></extra>"
        ),
        showlegend=False,
    )
)
figura.add_trace(
    go.Scatter(
        x=resultados["umbral"],
        y=resultados["modelo"],
        mode="markers",
        marker={"symbol": "line-ns", "size": 34, "line": {"width": 3, "color": "#222222"}},
        name="Umbral de clasificación",
        hovertemplate="Umbral de %{y}: %{x:.1%}<extra></extra>",
    )
)
# Etiquetas directas en una columna fija, para que no choquen con la línea del umbral
for r in resultados.itertuples():
    figura.add_annotation(
        x=1.03,
        y=r.modelo,
        xref="x",
        text=f"<b>{porcentaje(r.probabilidad, 0)}</b> {'recompra' if r.recompra else 'no recompra'}",
        showarrow=False,
        xanchor="left",
        font={"color": "#222222" if r.recompra else TEXTO_SECUNDARIO, "size": 14},
    )
figura.update_layout(
    template=PLANTILLA,
    title=titulo,
    height=150 + 70 * len(resultados),
    xaxis={"range": [0, 1.32], "tickformat": ".0%", "tickvals": [0, 0.25, 0.5, 0.75, 1], "title": "Probabilidad de recompra"},
    yaxis={"showgrid": False, "title": None},
    legend={"orientation": "h", "y": -0.25, "x": 0},
    bargap=0.35,
)
st.plotly_chart(figura, width="stretch")
st.caption(
    "La línea negra marca el umbral de cada modelo, elegido en la semana de validación: si la barra lo "
    "alcanza, el modelo clasifica al cliente como recompra. La regresión logística es el modelo de "
    "referencia del proyecto, con el mayor AUC en la semana de prueba."
)

if real is not None and st.toggle("Mostrar qué pasó en realidad con este cliente"):
    if real:
        st.success("En la realidad, este cliente **sí volvió a comprar** el producto después de usar el cupón.")
    else:
        st.info("En la realidad, este cliente **no volvió a comprar** el producto después de usar el cupón.")

# ---------------------------------------------------------------------------
# 4. Contexto: el hallazgo central
# ---------------------------------------------------------------------------

st.header("4. ¿Por qué importa si ya compraba el producto?")
grupos = pd.DataFrame(
    {
        "grupo": ["Nunca había comprado el producto", "Ya había comprado el producto"],
        "tasa": [referencia["tasa_sin_producto"], referencia["tasa_con_producto"]],
        "clientes": [referencia["clientes_sin_producto"], referencia["clientes_con_producto"]],
        "es_cliente": [not compro_producto, compro_producto],
    }
)
contexto = go.Figure(
    go.Bar(
        x=grupos["tasa"],
        y=grupos["grupo"],
        orientation="h",
        marker_color=[AZUL if es else GRIS_CLARO for es in grupos["es_cliente"]],
        text=[
            f"  {porcentaje(t)} de {c:,} clientes" + ("  ← grupo de este cliente" if es else "")
            for t, c, es in zip(grupos["tasa"], grupos["clientes"], grupos["es_cliente"])
        ],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%} volvió a comprar<extra></extra>",
    )
)
contexto.update_layout(
    template=PLANTILLA,
    title="Haber comprado antes el producto casi duplica la recompra",
    height=260,
    xaxis={"range": [0, 0.62], "tickformat": ".0%", "title": "Clientes que volvieron a comprar"},
    yaxis={"showgrid": False, "title": None},
    showlegend=False,
)
st.plotly_chart(contexto, width="stretch")
st.caption(
    "Fuente: los 160,057 clientes con respuesta conocida de dataset_modelo.csv. Es el hallazgo central del "
    "análisis exploratorio y una de las variables que más pesa en los modelos."
)

# ---------------------------------------------------------------------------
# Pie: cómo se garantiza que la app usa los mismos modelos
# ---------------------------------------------------------------------------

st.divider()
verificacion = resumen_verificacion()
if verificacion:
    st.markdown(
        f"<span style='color:{TEXTO_SECUNDARIO}'>Verificación: en {verificacion['clientes']} clientes de la "
        f"semana de prueba, la aplicación calculó las mismas {verificacion['variables']} variables que el "
        f"entrenamiento y las mismas probabilidades que el cuaderno de evaluación en "
        f"{verificacion['coinciden']} de {verificacion['clientes']} casos. Los modelos son la versión 2, "
        "entrenada con las ofertas del 1 de marzo al 23 de abril de 2013, y se cargan congelados: la "
        "aplicación solo les pide predicciones.</span>",
        unsafe_allow_html=True,
    )
