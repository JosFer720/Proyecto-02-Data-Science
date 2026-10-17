"""Verificación del objetivo 5: la aplicación prepara y clasifica igual que el entrenamiento.

Paso 1, `--extraer`: recorre transactions.csv.gz y guarda las transacciones crudas
completas de los primeros 100 clientes de la semana de prueba que aparecen en el
archivo. Así la verificación puede repetirse sin el archivo de 3 GB, y esos
clientes sirven también como ejemplos en la aplicación.

Paso 2, `--verificar`: pasa a cada cliente por `entrada_app.preparar_cliente` y
compara su fila con la de dataset_modelo.csv. Después clasifica la fila con los
modelos congelados de la versión 2 y compara cada probabilidad con la que el
notebook 08 guardó en 08_predicciones_prueba.parquet.

Uso, desde la raíz del repositorio:
    python src/verificar_entrada_app.py --extraer --verificar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TABLES_DIR, TRANSACTIONS_PATH
from entrada_app import COLUMNAS_SALIDA, preparar_cliente
from modelo import dividir_en_tres_conjuntos
from preprocesamiento import TAMANO_BLOQUE, TRANSACTIONS_DTYPES, cargar_dataset_modelo

EJEMPLOS_DIR = DATA_DIR / "ejemplos"
RUTA_TRANSACCIONES = EJEMPLOS_DIR / "transacciones_clientes_prueba.csv.gz"
RUTA_CLIENTES = EJEMPLOS_DIR / "clientes_prueba.csv"
RUTA_VERIFICACION = TABLES_DIR / "09_verificacion_app.csv"
RUTA_PREDICCIONES = TABLES_DIR / "08_predicciones_prueba.parquet"

N_CLIENTES = 100
TOLERANCIA_PROBABILIDAD = 1e-9


def extraer_clientes_de_prueba(n: int = N_CLIENTES) -> pd.DataFrame:
    """Guarda las transacciones completas de los primeros n clientes de la semana de prueba."""
    prueba = dividir_en_tres_conjuntos(cargar_dataset_modelo()).prueba
    objetivo = set(prueba["id"])

    completos: list[int] = []
    filas: list[pd.DataFrame] = []
    actual = None
    lector = pd.read_csv(TRANSACTIONS_PATH, compression="gzip", dtype=TRANSACTIONS_DTYPES, chunksize=TAMANO_BLOQUE)
    for bloque in lector:
        # las filas de cada cliente vienen contiguas: un cliente queda completo cuando aparece otro id
        for id_cliente, grupo in bloque.groupby("id", sort=False):
            if actual is not None and id_cliente != actual:
                if actual in objetivo:
                    completos.append(actual)
                actual = None
            if len(completos) >= n:
                break
            if id_cliente in objetivo:
                filas.append(grupo)
            actual = id_cliente
        if len(completos) >= n:
            break

    transacciones = pd.concat(filas, ignore_index=True)
    transacciones = transacciones[transacciones["id"].isin(completos)]
    clientes = (
        prueba.set_index("id")
        .loc[completos, ["chain", "market", "offer", "offerdate", "repeater"]]
        .reset_index()
    )
    clientes["offerdate"] = clientes["offerdate"].dt.strftime("%Y-%m-%d")

    EJEMPLOS_DIR.mkdir(parents=True, exist_ok=True)
    transacciones.to_csv(RUTA_TRANSACCIONES, index=False, compression="gzip")
    clientes.to_csv(RUTA_CLIENTES, index=False)
    print(f"{len(clientes)} clientes y {len(transacciones):,} transacciones guardadas en {EJEMPLOS_DIR}")
    return clientes


def cargar_clientes_de_prueba() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clientes de ejemplo y sus transacciones crudas."""
    clientes = pd.read_csv(RUTA_CLIENTES)
    transacciones = pd.read_csv(RUTA_TRANSACCIONES)
    return clientes, transacciones


def _iguales(a: pd.Series, b: pd.Series) -> pd.Series:
    """Igualdad columna a columna, con los faltantes como iguales entre sí."""
    numerico = pd.to_numeric(a, errors="coerce").notna() | pd.to_numeric(b, errors="coerce").notna()
    resultado = pd.Series(False, index=a.index)
    for col in a.index:
        x, y = a[col], b[col]
        if pd.isna(x) and pd.isna(y):
            resultado[col] = True
        elif numerico[col] and not isinstance(x, pd.Timestamp):
            resultado[col] = bool(np.isclose(float(x), float(y), rtol=0, atol=1e-9))
        else:
            resultado[col] = x == y
    return resultado


def verificar(modelos: dict | None = None) -> pd.DataFrame:
    """Compara variables y probabilidades de la app contra el entrenamiento y el notebook 08."""
    from modelos_congelados import cargar_modelos
    from estilo_graficos import algoritmo_de

    clientes, transacciones = cargar_clientes_de_prueba()
    dataset = cargar_dataset_modelo().set_index("id")
    predicciones = pd.read_parquet(RUTA_PREDICCIONES).set_index("id")
    if modelos is None:
        modelos = cargar_modelos(version=2, estricto=True)

    filas = []
    for cliente in clientes.itertuples(index=False):
        resultado = preparar_cliente(
            transacciones[transacciones["id"] == cliente.id],
            oferta=cliente.offer,
            fecha_oferta=cliente.offerdate,
            market=cliente.market,
            cadena=cliente.chain,
        )
        app = resultado.fila.set_index("id").iloc[0]
        entrenamiento = dataset.loc[cliente.id, [c for c in COLUMNAS_SALIDA if c != "id"]]
        iguales = _iguales(app[entrenamiento.index], entrenamiento)

        fila = {
            "id": cliente.id,
            "variables_comparadas": int(len(iguales)),
            "variables_distintas": int((~iguales).sum()),
            "columnas_distintas": ", ".join(iguales.index[~iguales]),
        }
        for nombre, modelo in modelos.items():
            algoritmo = algoritmo_de(nombre)
            prob_app = float(modelo.predecir_proba(resultado.fila)[0])
            prob_notebook = float(predicciones.loc[cliente.id, f"probabilidad_{algoritmo}"])
            fila[f"probabilidad_app_{algoritmo}"] = prob_app
            fila[f"diferencia_{algoritmo}"] = abs(prob_app - prob_notebook)
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    diferencias = [c for c in tabla.columns if c.startswith("diferencia_")]
    tabla["coincide"] = (tabla["variables_distintas"] == 0) & (
        tabla[diferencias].max(axis=1) <= TOLERANCIA_PROBABILIDAD
    )
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extraer", action="store_true", help="extrae los clientes desde transactions.csv.gz")
    parser.add_argument("--verificar", action="store_true", help="compara la app con el entrenamiento")
    args = parser.parse_args()
    if not (args.extraer or args.verificar):
        parser.error("indica --extraer, --verificar o ambos")

    if args.extraer:
        extraer_clientes_de_prueba()
    if args.verificar:
        tabla = verificar()
        tabla.to_csv(RUTA_VERIFICACION, index=False)
        diferencias = [c for c in tabla.columns if c.startswith("diferencia_")]
        print(f"Clientes verificados: {len(tabla)}")
        print(f"Clientes con todas las variables iguales: {(tabla['variables_distintas'] == 0).sum()}")
        print(f"Mayor diferencia de probabilidad: {tabla[diferencias].to_numpy().max():.2e}")
        print(f"Clientes que coinciden en todo: {tabla['coincide'].sum()} de {len(tabla)}")
        print(f"Tabla guardada en {RUTA_VERIFICACION}")


if __name__ == "__main__":
    main()
