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
    TEXTO,
    TEXTO_SECUNDARIO,
    algoritmo_de,
    aplicar_plantilla_plotly,
    color_modelo,
    nombre_modelo,
)
from modelos_congelados import ModeloCongeladoError, cargar_modelos  # noqa: E402
from verificar_entrada_app import RUTA_CLIENTES, RUTA_TRANSACCIONES, RUTA_VERIFICACION  # noqa: E402

st.set_page_config(page_title="Clasificar un cliente", page_icon=":material/person_search:", layout="wide")
PLANTILLA = aplicar_plantilla_plotly()
MODELO_REFERENCIA = "regresion_logistica"

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
        "clientes_total": len(datos),
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
    return {
        "clientes": len(tabla),
        "coinciden": int(tabla["coincide"].sum()),
        "variables": int(tabla["variables_comparadas"].iloc[0]),
    }


def porcentaje(valor: float, decimales: int = 0) -> str:
    return f"{valor * 100:.{decimales}f}%"


def tarjeta_resumen(resultado: dict, coincidencias: str | None, es_referencia: bool) -> None:
    """Tarjeta grande con la predicción del modelo de referencia o del modelo elegido."""
    with st.container(border=True):
        etiqueta = "Modelo de referencia" if es_referencia else "Modelo elegido"
        st.markdown(
            f"<div style='color:{TEXTO_SECUNDARIO};font-size:0.95rem'>{etiqueta}</div>"
            f"<div style='font-size:1.15rem;font-weight:600;color:{TEXTO}'>"
            f"<span style='color:{color_modelo(resultado['nombre'])}'>●</span> {resultado['modelo']}</div>"
            f"<div style='font-size:3.6rem;font-weight:700;line-height:1.1;color:{TEXTO};margin-top:0.4rem'>"
            f"{porcentaje(resultado['probabilidad'])}</div>"
            f"<div style='color:{TEXTO_SECUNDARIO};margin-bottom:0.6rem'>"
            "de probabilidad de que vuelva a comprar el producto</div>",
            unsafe_allow_html=True,
        )
        if resultado["recompra"]:
            st.badge("Volverá a comprar", icon=":material/check_circle:", color="blue")
            detalle = f"Su probabilidad alcanza el umbral de {porcentaje(resultado['umbral'])} de este modelo."
        else:
            st.badge("No volverá a comprar", icon=":material/cancel:", color="gray")
            detalle = f"Su probabilidad no alcanza el umbral de {porcentaje(resultado['umbral'])} de este modelo."
        st.caption(detalle if coincidencias is None else f"{detalle} {coincidencias}")


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("Clasificar un cliente nuevo")
st.markdown(
    "Ingresa las compras de un cliente y la oferta que quieres hacerle. La aplicación prepara sus datos "
    "igual que en el entrenamiento y estima con cada modelo la probabilidad de que **vuelva a comprar "
    "el producto** después de usar el cupón."
)
verificacion = resumen_verificacion()
with st.container(horizontal=True, gap="small"):
    st.badge("4 modelos de aprendizaje automático", icon=":material/model_training:", color="blue")
    st.badge("Modelos congelados, versión 2", icon=":material/lock:", color="gray")
    if verificacion:
        st.badge(
            f"Verificada en {verificacion['coinciden']} de {verificacion['clientes']} clientes",
            icon=":material/verified:",
            color="blue",
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
etiquetas_oferta = ofertas.set_index("offer")["etiqueta"]

izquierda, derecha = st.columns([5, 6], gap="large")

# ---------------------------------------------------------------------------
# 1. Datos del cliente
# ---------------------------------------------------------------------------

with izquierda:
    st.subheader("1. Datos del cliente")
    origen = st.segmented_control(
        "¿De dónde vienen las compras del cliente?",
        ["Cliente de ejemplo", "Subir un archivo"],
        default="Cliente de ejemplo",
        required=True,
    )

    transacciones = None
    real = None
    if origen == "Cliente de ejemplo":
        clientes, todas = clientes_ejemplo()
        etiquetas = {
            fila.id: f"Cliente {fila.id}: oferta {fila.offer}, {fila.lineas:,} líneas de compra"
            for fila in clientes.itertuples(index=False)
        }
        id_cliente = st.selectbox(
            "Cliente",
            list(etiquetas),
            format_func=etiquetas.get,
            help=(
                f"{len(clientes)} clientes reales de la semana de prueba, del 24 al 30 de abril de 2013. "
                "Los modelos no los vieron durante el entrenamiento."
            ),
        )
        cliente = clientes.set_index("id").loc[id_cliente]
        transacciones = todas[todas["id"] == id_cliente]
        oferta = int(cliente["offer"])
        fecha_oferta = pd.Timestamp(cliente["offerdate"])
        mercado = int(cliente["market"])
        real = bool(cliente["repeater"])

        with st.container(border=True):
            st.markdown(
                f"**Oferta:** {etiquetas_oferta.loc[oferta]}  \n"
                f"**Fecha de la oferta:** {fecha_oferta:%d/%m/%Y}  \n"
                f"**Mercado:** {mercado}"
            )
        st.download_button(
            "Descargar sus compras en CSV",
            transacciones.to_csv(index=False).encode("utf-8"),
            file_name=f"compras_cliente_{id_cliente}.csv",
            mime="text/csv",
            icon=":material/download:",
            help="Sirve como modelo del formato para probar la opción de subir un archivo.",
        )
    else:
        archivo = st.file_uploader(
            "Archivo CSV con las compras del cliente",
            type="csv",
            help=(
                "Una fila por línea de compra de un solo cliente, con las columnas de transactions.csv: "
                "chain, category, company, brand, date, purchasequantity y purchaseamount."
            ),
        )
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

        oferta = st.selectbox("Oferta que se le hará", ofertas["offer"].tolist(), format_func=etiquetas_oferta.get)
        columna_fecha, columna_mercado = st.columns(2)
        fecha_oferta = pd.Timestamp(
            columna_fecha.date_input(
                "Fecha de la oferta",
                value=fecha_sugerida.date(),
                help="Solo se usan las compras anteriores a esta fecha.",
                format="DD/MM/YYYY",
            )
        )
        mercado = columna_mercado.selectbox("Mercado", referencia["mercados"])

    if transacciones is None:
        st.info("Sube un archivo de compras para continuar, o elige un cliente de ejemplo.", icon=":material/upload_file:")
        with derecha:
            st.subheader("3. ¿Volverá a comprar?")
            with st.container(border=True):
                st.markdown(
                    f"<div style='color:{TEXTO_SECUNDARIO};padding:2.5rem 0;text-align:center'>"
                    "Aquí aparecerá la probabilidad de recompra según cada modelo "
                    "cuando subas las compras del cliente.</div>",
                    unsafe_allow_html=True,
                )
        st.stop()

    try:
        entrada = preparar_cliente(transacciones, oferta=oferta, fecha_oferta=fecha_oferta, market=mercado)
    except EntradaInvalida as error:
        st.error(str(error))
        st.stop()

    for aviso in entrada.avisos:
        st.warning(aviso, icon=":material/info:")

    fila = entrada.fila.iloc[0]
    compro_producto = int(fila["n_compras_producto"]) > 0

    # -----------------------------------------------------------------------
    # 2. Ficha del cliente
    # -----------------------------------------------------------------------

    st.subheader("2. Ficha del cliente")
    st.caption(
        f"Calculada con {entrada.lineas_usadas:,} de {entrada.lineas_recibidas:,} líneas de compra, "
        f"todas de la cadena {entrada.cadena} y anteriores a la oferta."
    )

    def sin_compras_o(valor, formato: str) -> str:
        return "Sin compras" if pd.isna(valor) else formato.format(valor)

    tarjetas = [
        ("Visitas a la tienda", f"{int(fila['n_visitas']):,}", ":material/storefront:"),
        ("Gasto total", f"${fila['gasto_total']:,.2f}", ":material/payments:"),
        ("Días desde la última compra", sin_compras_o(fila["dias_ultima_compra"], "{:,.0f}"), ":material/schedule:"),
        (
            "¿Ya compraba el producto?",
            f"Sí, {int(fila['n_compras_producto'])} veces" if compro_producto else "No",
            ":material/shopping_basket:",
        ),
        ("Compras de la categoría", f"{int(fila['n_compras_categoria']):,}", ":material/category:"),
        ("Compras de la marca", f"{int(fila['n_compras_marca']):,}", ":material/sell:"),
        ("Ticket promedio", sin_compras_o(fila["ticket_promedio"], "${:,.2f}"), ":material/receipt_long:"),
        ("Días de historial", sin_compras_o(fila["antiguedad_dias"], "{:,.0f}"), ":material/history:"),
    ]
    for inicio in range(0, len(tarjetas), 2):
        columnas = st.columns(2)
        for columna, (etiqueta, valor, icono) in zip(columnas, tarjetas[inicio : inicio + 2]):
            columna.metric(etiqueta, valor, border=True, icon=icono)

    with st.expander("Ver todas las variables que reciben los modelos", icon=":material/table_view:"):
        variables = entrada.fila.drop(columns=["id"]).T.rename(columns={0: "valor"})
        variables["valor"] = variables["valor"].astype(str)
        st.dataframe(variables, width="stretch", height=320)

# ---------------------------------------------------------------------------
# 3. Resultado de los modelos
# ---------------------------------------------------------------------------

with derecha:
    st.subheader("3. ¿Volverá a comprar?")
    vista = st.segmented_control(
        "Mostrar", ["Los cuatro modelos", "Un solo modelo"], default="Los cuatro modelos", required=True
    )
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
    resultados = pd.DataFrame(resultados)

    if len(resultados) == 1:
        principal = resultados.iloc[0].to_dict()
        tarjeta_resumen(principal, None, es_referencia=algoritmo_de(principal["nombre"]) == MODELO_REFERENCIA)
    else:
        principal = resultados[resultados["nombre"].map(algoritmo_de) == MODELO_REFERENCIA].iloc[0].to_dict()
        coinciden = int((resultados["recompra"] == principal["recompra"]).sum())
        tarjeta_resumen(principal, f"{coinciden} de {len(resultados)} modelos coinciden.", es_referencia=True)

    positivos = int(resultados["recompra"].sum())
    titulo = (
        "Probabilidad de recompra"
        if len(resultados) == 1
        else f"{positivos} de {len(resultados)} modelos predicen que volverá a comprar"
    )
    grafica = resultados.iloc[::-1]  # el primer modelo queda arriba
    figura = go.Figure()
    figura.add_trace(
        go.Bar(
            x=grafica["probabilidad"],
            y=grafica["modelo"],
            orientation="h",
            marker_color=[color_modelo(n) for n in grafica["nombre"]],
            customdata=grafica[["umbral", "auc"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Probabilidad: %{x:.1%}<br>Umbral del modelo: %{customdata[0]:.1%}"
                "<br>AUC en la semana de prueba: %{customdata[1]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figura.add_trace(
        go.Scatter(
            x=grafica["umbral"],
            y=grafica["modelo"],
            mode="markers",
            marker={"symbol": "line-ns", "size": 34, "line": {"width": 3, "color": TEXTO}},
            name="Umbral de clasificación",
            hovertemplate="Umbral de %{y}: %{x:.1%}<extra></extra>",
        )
    )
    # Etiquetas directas en una columna fija, para que no choquen con la línea del umbral
    for r in grafica.itertuples():
        figura.add_annotation(
            x=1.03,
            y=r.modelo,
            text=f"<b>{porcentaje(r.probabilidad)}</b> {'recompra' if r.recompra else 'no recompra'}",
            showarrow=False,
            xanchor="left",
            font={"color": TEXTO if r.recompra else TEXTO_SECUNDARIO, "size": 15},
        )
    figura.update_layout(
        template=PLANTILLA,
        title=titulo,
        height=150 + 66 * len(grafica),
        xaxis={
            "range": [0, 1.42],
            "tickformat": ".0%",
            "tickvals": [0, 0.25, 0.5, 0.75, 1],
            "title": "Probabilidad de recompra",
            "automargin": True,
        },
        yaxis={"showgrid": False, "title": None, "automargin": True, "ticksuffix": "   "},
        legend={"orientation": "h", "y": -0.3, "x": 0, "font": {"size": 13}},
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.35,
        margin={"l": 10, "r": 10, "t": 50, "b": 20},
    )
    st.plotly_chart(figura, width="stretch", theme=None)  # usa la plantilla del proyecto
    st.caption(
        "La línea negra marca el umbral de cada modelo, elegido en la semana de validación: si la barra lo "
        "alcanza, el modelo clasifica al cliente como recompra. La regresión logística es el modelo de "
        "referencia, con el mayor AUC en la semana de prueba. Pasa el cursor sobre una barra para ver su detalle."
    )

    if real is not None and st.toggle("Mostrar qué pasó en realidad con este cliente"):
        if real:
            st.success("En la realidad, este cliente **sí volvió a comprar** el producto.", icon=":material/check_circle:")
        else:
            st.info("En la realidad, este cliente **no volvió a comprar** el producto.", icon=":material/cancel:")

# ---------------------------------------------------------------------------
# 4. Contexto: el hallazgo central
# ---------------------------------------------------------------------------

st.divider()
st.subheader("4. ¿Por qué importa si ya compraba el producto?")
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
            f"  <b>{porcentaje(t, 1)}</b> de {c:,} clientes" + ("   ← grupo de este cliente" if es else "")
            for t, c, es in zip(grupos["tasa"], grupos["clientes"], grupos["es_cliente"])
        ],
        textposition="outside",
        textfont={"size": 15, "color": TEXTO},
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%} volvió a comprar<extra></extra>",
    )
)
contexto.update_layout(
    template=PLANTILLA,
    title="Haber comprado antes el producto casi duplica la recompra",
    height=320,
    xaxis={"range": [0, 0.68], "tickformat": ".0%", "title": "Clientes que volvieron a comprar", "automargin": True},
    yaxis={"showgrid": False, "title": None, "automargin": True, "ticksuffix": "   "},
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin={"l": 10, "r": 10, "t": 50, "b": 70},
)
st.plotly_chart(contexto, width="stretch", theme=None)  # usa la plantilla del proyecto
st.caption(
    f"Fuente: los {referencia['clientes_total']:,} clientes con respuesta conocida de dataset_modelo.csv. "
    "Es el hallazgo central del análisis exploratorio y una de las variables que más pesa en los modelos."
)

if verificacion:
    st.caption(
        f"Cómo se garantiza el resultado: en {verificacion['clientes']} clientes de la semana de prueba, esta "
        f"página calculó las mismas {verificacion['variables']} variables que el entrenamiento y las mismas "
        f"probabilidades que el cuaderno de evaluación en {verificacion['coinciden']} de "
        f"{verificacion['clientes']} casos. Los modelos son la versión 2, entrenada con las ofertas del 1 de "
        "marzo al 23 de abril de 2013, y se cargan congelados: la aplicación solo les pide predicciones."
    )
