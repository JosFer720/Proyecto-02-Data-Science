"""Paleta y estilo común para las figuras del informe y las gráficas de la aplicación.

Sigue la paleta Okabe-Ito, segura para personas con daltonismo:
- cada modelo tiene un color fijo, el mismo en el informe y en todas las páginas de la app;
- el azul es el acento principal, el gris es el contexto y el bermellón se reserva
  para señalar un hallazgo o un problema;
- nunca se usa el color como único canal: se acompaña con etiquetas directas.

Uso con matplotlib:
    from estilo_graficos import estilo_matplotlib, COLORES_MODELO
    estilo_matplotlib(ax, "AUC en la semana de prueba", fuente="08_metricas_prueba.csv")

Uso con Plotly:
    from estilo_graficos import aplicar_plantilla_plotly
    fig.update_layout(template=aplicar_plantilla_plotly())
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colores
# ---------------------------------------------------------------------------

AZUL = "#0072B2"  # acento principal
BERMELLON = "#D55E00"  # solo para el hallazgo o el problema
VERDE = "#009E73"
NARANJA = "#E69F00"
ROSA = "#CC79A7"
CELESTE = "#56B4E9"

GRIS = "#999999"  # contexto
GRIS_CLARO = "#C9CCD6"  # contexto con muchas series o fondo de barras
REJILLA = "#E5E5EC"
TEXTO = "#222222"
TEXTO_SECUNDARIO = "#777777"

ORDEN_MODELOS = (
    "regresion_logistica",
    "random_forest",
    "gradient_boosting",
    "perceptron_multicapa",
)

# Los mismos colores que usan las figuras 08 del informe
COLORES_MODELO = {
    "regresion_logistica": AZUL,
    "random_forest": NARANJA,
    "gradient_boosting": VERDE,
    "perceptron_multicapa": ROSA,
}

NOMBRES_MODELO = {
    "regresion_logistica": "Regresión logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "perceptron_multicapa": "Perceptrón multicapa",
}


def algoritmo_de(nombre_modelo: str) -> str:
    """Convierte '07_random_forest_v2' en 'random_forest'."""
    base = nombre_modelo.removeprefix("07_")
    for sufijo in ("_v2", "_v1"):
        base = base.removesuffix(sufijo)
    return base


def color_modelo(nombre_modelo: str) -> str:
    return COLORES_MODELO[algoritmo_de(nombre_modelo)]


def nombre_modelo(nombre_modelo: str) -> str:
    return NOMBRES_MODELO[algoritmo_de(nombre_modelo)]


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------

def estilo_matplotlib(ax, titulo: str, subtitulo: str | None = None, fuente: str | None = None):
    """Título a la izquierda, sin marco superior ni derecho, rejilla horizontal tenue
    y pie con la fuente."""
    ax.set_title(titulo, loc="left", fontsize=13, color=TEXTO, pad=18 if subtitulo else 8)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=10, color=TEXTO_SECUNDARIO)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    if fuente:
        ax.figure.text(0.01, -0.02, f"Fuente: {fuente}", color=TEXTO_SECUNDARIO, fontsize=9)
    return ax


def anotar(ax, x, y, texto: str, x_texto, y_texto):
    """Punto bermellón sobre el dato, flecha gris y frase que enuncia el hallazgo."""
    ax.plot(x, y, "o", color=BERMELLON, zorder=5)
    ax.annotate(
        texto,
        xy=(x, y),
        xytext=(x_texto, y_texto),
        color=TEXTO,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1},
    )


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

NOMBRE_PLANTILLA = "proyecto2"


def plantilla_plotly():
    """Plantilla de Plotly con la misma paleta y anatomía que las figuras estáticas."""
    import plotly.graph_objects as go

    return go.layout.Template(
        layout=go.Layout(
            font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 14, "color": TEXTO},
            title={"x": 0, "xanchor": "left", "font": {"size": 18}},
            colorway=[COLORES_MODELO[a] for a in ORDEN_MODELOS] + [CELESTE, BERMELLON, GRIS],
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis={"showgrid": False, "linecolor": GRIS, "ticks": "outside", "tickcolor": GRIS, "zeroline": False},
            yaxis={"showgrid": True, "gridcolor": REJILLA, "linecolor": GRIS, "zeroline": False},
            legend={"bgcolor": "rgba(0,0,0,0)"},
            hoverlabel={"bgcolor": "white", "font": {"color": TEXTO}},
            margin={"l": 10, "r": 10, "t": 60, "b": 40},
        )
    )


def aplicar_plantilla_plotly() -> str:
    """Registra la plantilla y devuelve su nombre para usarla en `template=`."""
    import plotly.io as pio

    if NOMBRE_PLANTILLA not in pio.templates:
        pio.templates[NOMBRE_PLANTILLA] = plantilla_plotly()
    return NOMBRE_PLANTILLA
