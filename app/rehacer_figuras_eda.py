"""Regenera las figuras categóricas de Nery desde las tablas originales.

Uso: python app/rehacer_figuras_eda.py
No cambia datos, modelos ni notebooks de otros integrantes.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from estilo_graficos import AZUL, BERMELLON, GRIS, estilo_matplotlib

NOMBRES = {"chain": "cadena", "category": "categoría", "brand": "marca", "company": "compañía", "market": "mercado"}
PLURALES = {"chain": "cadenas", "category": "categorías", "brand": "marcas", "company": "compañías", "market": "mercados"}
# Los rangos de descuento tienen orden propio: se grafican de menor a mayor valor, no por tasa.
ORDEN_OFERTA = ["<= 1.00", "1.00–1.50", "1.50–2.00", "> 2.00"]
ETIQUETAS_OFERTA = {"<= 1.00": "Hasta 1.00", "1.00–1.50": "1.00 a 1.50", "1.50–2.00": "1.50 a 2.00", "> 2.00": "Más de 2.00"}

def main():
    tablas = RAIZ / "results/tables"
    figuras = RAIZ / "results/figures"
    global_ = pd.read_csv(RAIZ / "data/processed/dataset_modelo.csv", usecols=["repeater"]).repeater.mean()
    for dimension in ["chain", "category", "brand", "company", "market"]:
        completa = pd.read_csv(tablas / f"04_frecuencia_{dimension}.csv")
        tabla = completa.nlargest(15, "frecuencia").sort_values("frecuencia")
        fila_mayor = completa.frecuencia.idxmax()
        grupo_mayor, proporcion_mayor = completa.grupo.loc[fila_mayor], completa.proporcion.loc[fila_mayor]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(tabla.grupo.astype(str), tabla.frecuencia, color=AZUL)
        ax.bar_label(ax.containers[0], labels=[f"n={n:,}" for n in tabla.frecuencia], padding=4)
        ax.set_xlim(0, tabla.frecuencia.max()*1.25)
        ax.set_xlabel("Número de clientes")
        ax.set_ylabel(f"{NOMBRES[dimension].capitalize()} (ID anónimo)")
        alcance = (f"las {len(tabla)} {PLURALES[dimension]} con más clientes, de {len(completa)}"
                   if len(completa) > len(tabla) else f"las {len(completa)} {PLURALES[dimension]}")
        estilo_matplotlib(ax, f"La {NOMBRES[dimension]} {grupo_mayor} concentra el {proporcion_mayor:.1%} de los clientes",
                          subtitulo=f"Se muestran {alcance}", fuente=f"04_frecuencia_{dimension}.csv")
        fig.savefig(figuras / f"04_frecuencia_{dimension}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    for dimension in ["chain", "category", "offervalue"]:
        completa = pd.read_csv(tablas / f"04_tasa_recompra_{dimension}.csv")
        if dimension == "offervalue":
            tabla = completa.set_index("grupo").loc[ORDEN_OFERTA].reset_index()
            nombres = [ETIQUETAS_OFERTA[g] for g in tabla.grupo]
            titulo = (f"La recompra va de {tabla.tasa_recompra.min():.0%} a {tabla.tasa_recompra.max():.0%} "
                      "según el valor del descuento")
            alcance = "los cuatro rangos de descuento, en orden de valor"
        else:
            tabla = completa.nlargest(15, "n").sort_values("tasa_recompra")
            nombres = tabla.grupo.astype(str).tolist()
            titulo = (f"La recompra va de {tabla.tasa_recompra.min():.0%} a {tabla.tasa_recompra.max():.0%} "
                      f"entre las {PLURALES[dimension]}")
            alcance = (f"las {len(tabla)} {PLURALES[dimension]} con más clientes, de {len(completa)}, ordenadas por tasa"
                       if len(completa) > len(tabla) else f"las {len(completa)} {PLURALES[dimension]}, ordenadas por tasa")
        fig, ax = plt.subplots(figsize=(12, 7))
        posiciones = range(len(tabla))
        colores = [BERMELLON if t == tabla.tasa_recompra.max() else AZUL for t in tabla.tasa_recompra]
        ax.bar(posiciones, tabla.tasa_recompra*100, color=colores)
        # El tamaño de cada grupo va bajo su etiqueta, para que no se encime con la tasa
        ax.set_xticks(list(posiciones), [f"{g}\nn={r.n:,.0f}" for g, r in zip(nombres, tabla.itertuples())], fontsize=8)
        ax.bar_label(ax.containers[0], labels=[f"{r.tasa_recompra:.1%}" for r in tabla.itertuples()], padding=4, fontsize=9)
        ax.axhline(global_*100, color=GRIS, linestyle="--", label=f"Global: {global_:.2%}")
        ax.set_ylim(0, max(tabla.tasa_recompra.max(), global_)*135)
        ax.set_ylabel("Clientes que recompran (%)")
        ax.set_xlabel({"chain": "Cadena (ID anónimo) y número de clientes", "category": "Categoría (ID anónimo) y número de clientes",
                       "offervalue": "Valor del descuento en dólares y número de clientes"}[dimension])
        ax.legend(loc="upper left")
        estilo_matplotlib(ax, titulo, subtitulo=f"Se muestran {alcance}", fuente=f"04_tasa_recompra_{dimension}.csv")
        fig.savefig(figuras / f"04_tasa_recompra_{dimension}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    # La matriz de correlación no representa tasas: su escala debe conservar el signo.
    matriz = pd.read_csv(tablas / "04_matriz_correlacion.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(17, 15))
    imagen = ax.imshow(matriz, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matriz)), matriz.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matriz)), matriz.index, fontsize=7)
    fig.colorbar(imagen, ax=ax, label="Correlación de Spearman (−1 a 1)", shrink=.7)
    ax.set_title("Asociaciones entre variables de comportamiento · Spearman", loc="left")
    fig.savefig(figuras / "04_matriz_correlacion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
