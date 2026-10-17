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

def main():
    tablas = RAIZ / "results/tables"
    figuras = RAIZ / "results/figures"
    global_ = pd.read_csv(RAIZ / "data/processed/dataset_modelo.csv", usecols=["repeater"]).repeater.mean()
    for dimension in ["chain", "category", "brand", "company", "market"]:
        tabla = pd.read_csv(tablas / f"04_frecuencia_{dimension}.csv").nlargest(15, "frecuencia").sort_values("frecuencia")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(tabla.grupo.astype(str), tabla.frecuencia, color=AZUL)
        ax.bar_label(ax.containers[0], labels=[f"n={n:,}" for n in tabla.frecuencia], padding=4)
        ax.set_xlim(0, tabla.frecuencia.max()*1.25)
        ax.set_xlabel("Número de clientes")
        estilo_matplotlib(ax, f"{dimension}: concentración de clientes en los 15 grupos más frecuentes", fuente=f"04_frecuencia_{dimension}.csv")
        fig.savefig(figuras / f"04_frecuencia_{dimension}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    for dimension in ["chain", "category", "offervalue"]:
        tabla = pd.read_csv(tablas / f"04_tasa_recompra_{dimension}.csv").nlargest(15, "n").sort_values("tasa_recompra")
        fig, ax = plt.subplots(figsize=(11, 7))
        posiciones = range(len(tabla))
        colores = [BERMELLON if t == tabla.tasa_recompra.max() else AZUL for t in tabla.tasa_recompra]
        ax.bar(posiciones, tabla.tasa_recompra*100, color=colores)
        ax.set_xticks(list(posiciones), tabla.grupo.astype(str), rotation=30)
        ax.bar_label(ax.containers[0], labels=[f"{r.tasa_recompra:.1%}\nn={r.n:,.0f}" for r in tabla.itertuples()], padding=5, fontsize=8)
        ax.axhline(global_*100, color=GRIS, linestyle="--", label=f"Global: {global_:.2%}")
        ax.set_ylim(0, max(tabla.tasa_recompra.max(), global_)*135)
        ax.set_ylabel("Clientes que recompran (%)")
        ax.set_xlabel({"chain": "Cadena (ID anónimo)", "category": "Categoría (ID anónimo)", "offervalue": "Valor del descuento (unidades monetarias)"}[dimension])
        ax.legend(loc="upper left")
        estilo_matplotlib(ax, "La recompra cambia entre grupos; el tamaño acompaña cada tasa", fuente=f"04_tasa_recompra_{dimension}.csv · top 15 por tamaño")
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
