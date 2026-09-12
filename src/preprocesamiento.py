"""Preprocesamiento de transactions.csv.gz y construcción de dataset_modelo.csv.

El archivo tiene cerca de 350 millones de filas, así que se recorre una sola vez
por bloques y en esa misma pasada:
- se valida cada transacción contra el cliente que recibió la oferta
  (misma cadena y fecha anterior a su offerdate),
- se construyen las variables de comportamiento de cada cliente,
- se acumulan las estadísticas de limpieza que reporta el notebook 02,
- se guarda una muestra aleatoria de filas crudas para el EDA del notebook 03,
- se guardan las transacciones reducidas a las categorías y compañías ofertadas.

Supuesto que se verifica durante la pasada: las filas de cada cliente vienen
contiguas en el archivo. Por eso las filas del último cliente de cada bloque se
pasan al bloque siguiente y cada cliente se resume completo una sola vez.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carga import cargar_offers, cargar_test, cargar_train
from config import (
    DATASET_MODELO_PATH,
    DATASET_MODELO_TEST_PATH,
    RANDOM_STATE,
    TABLES_DIR,
    TRANSACCIONES_MUESTRA_PATH,
    TRANSACCIONES_REDUCIDAS_PATH,
    TRANSACTIONS_PATH,
)

TAMANO_BLOQUE = 2_000_000
FRACCION_MUESTRA = 0.001
VENTANAS_DIAS = (30, 60, 90, 180)

# nombre usado en las variables -> columna equivalente en offers y transactions
DIMENSIONES_OFERTA = {"categoria": "category", "compania": "company", "marca": "brand"}

# tipos reducidos: ~40% menos memoria que los que pandas infiere por defecto
TRANSACTIONS_DTYPES = {
    "id": "int64",
    "chain": "int16",
    "dept": "int16",
    "category": "int16",
    "company": "int64",
    "brand": "int32",
    "date": "str",
    "productsize": "float32",
    "productmeasure": "str",
    "purchasequantity": "int32",
    "purchaseamount": "float64",
}

VARIABLES_CRUDAS = ["purchaseamount", "purchasequantity", "productsize"]

TABLA_CONTEOS = TABLES_DIR / "02_conteos_limpieza.csv"
TABLA_ESTADISTICAS = TABLES_DIR / "02_estadisticas_transacciones.csv"
TABLA_PRODUCTMEASURE = TABLES_DIR / "02_productmeasure.csv"
TABLA_MENSUAL = TABLES_DIR / "02_transacciones_mensuales.csv"
TABLA_DESCUENTOS = TABLES_DIR / "02_lineas_descuento_por_categoria.csv"

SALIDAS = [
    DATASET_MODELO_PATH,
    DATASET_MODELO_TEST_PATH,
    TRANSACCIONES_MUESTRA_PATH,
    TABLA_CONTEOS,
    TABLA_ESTADISTICAS,
    TABLA_PRODUCTMEASURE,
    TABLA_MENSUAL,
    TABLA_DESCUENTOS,
]

CONCEPTOS_CONTEO = [
    "filas_leidas",
    "filas_sin_cliente",
    "filas_otra_cadena",
    "filas_en_o_despues_oferta",
    "filas_validas",
    "filas_reducidas",
    "filas_devolucion",
    "filas_descuento",
    "filas_monto_cero",
    "filas_cantidad_cero",
    "filas_productsize_cero_o_nulo",
    "filas_productmeasure_nulo",
    "clientes_no_contiguos",
]

COLUMNAS_CLIENTE = [
    "id", "chain", "market", "offer", "offerdate",
    "category", "company", "brand", "offervalue", "quantity",
]
COLUMNAS_OBJETIVO = ["repeattrips", "repeater"]
COLUMNAS_GENERALES = [
    "n_lineas", "n_compras", "n_devoluciones", "n_descuentos", "n_visitas",
    "gasto_total", "ticket_promedio", "dias_ultima_compra", "antiguedad_dias",
    "n_categorias_distintas", "n_marcas_distintas", "n_companias_distintas",
]
COLUMNAS_OFERTA = (
    [
        col
        for dim in DIMENSIONES_OFERTA
        for col in [f"n_compras_{dim}", f"gasto_{dim}"]
        + [f"n_compras_{dim}_{dias}d" for dias in VENTANAS_DIAS]
    ]
    + ["n_compras_producto", "gasto_producto", "dias_ultima_compra_categoria"]
    + [f"nunca_compro_{dim}" for dim in [*DIMENSIONES_OFERTA, "producto"]]
)


def cargar_clientes() -> pd.DataFrame:
    """Une train y test con el catálogo de ofertas: una fila por cliente,
    indexada por id, con la cadena, la fecha y el producto de su oferta."""
    offers = cargar_offers()
    train = cargar_train().assign(split="train")
    test = cargar_test().assign(split="test")
    clientes = pd.concat([train, test], ignore_index=True).merge(offers, on="offer", how="left")
    if clientes["id"].duplicated().any():
        raise ValueError("Hay clientes que aparecen más de una vez entre train y test")
    return clientes.set_index("id")


def comparar_memoria_dtypes(n_filas: int = 1_000_000) -> pd.DataFrame:
    """Memoria de las primeras n_filas con los tipos por defecto contra TRANSACTIONS_DTYPES."""
    por_defecto = pd.read_csv(TRANSACTIONS_PATH, compression="gzip", nrows=n_filas)
    reducido = pd.read_csv(TRANSACTIONS_PATH, compression="gzip", nrows=n_filas, dtype=TRANSACTIONS_DTYPES)
    reducido["date"] = pd.to_datetime(reducido["date"], format="%Y-%m-%d")
    tabla = pd.DataFrame(
        {
            "dtype_por_defecto": por_defecto.dtypes.astype(str),
            "mb_por_defecto": por_defecto.memory_usage(deep=True, index=False) / 1e6,
            "dtype_reducido": reducido.dtypes.astype(str),
            "mb_reducido": reducido.memory_usage(deep=True, index=False) / 1e6,
        }
    )
    tabla.loc["TOTAL", ["mb_por_defecto", "mb_reducido"]] = tabla[["mb_por_defecto", "mb_reducido"]].sum()
    return tabla.round(2)


def _acumular_estadisticas(acumulado: dict, tx: pd.DataFrame) -> None:
    """Suma, suma de cuadrados, extremos y conteos exactos de las variables crudas."""
    for col in VARIABLES_CRUDAS:
        serie = tx[col].astype("float64")
        validos = serie.dropna()
        a = acumulado.setdefault(
            col,
            {"n": 0, "nulos": 0, "suma": 0.0, "suma_cuadrados": 0.0,
             "min": np.inf, "max": -np.inf, "negativos": 0, "ceros": 0},
        )
        a["n"] += len(validos)
        a["nulos"] += len(serie) - len(validos)
        a["suma"] += float(validos.sum())
        a["suma_cuadrados"] += float((validos**2).sum())
        if len(validos):
            a["min"] = min(a["min"], float(validos.min()))
            a["max"] = max(a["max"], float(validos.max()))
        a["negativos"] += int((validos < 0).sum())
        a["ceros"] += int((validos == 0).sum())


def _tabla_estadisticas(acumulado: dict) -> pd.DataFrame:
    filas = []
    for col, a in acumulado.items():
        media = a["suma"] / a["n"]
        varianza = (a["suma_cuadrados"] - a["n"] * media**2) / (a["n"] - 1)
        filas.append(
            {"variable": col, "n": a["n"], "nulos": a["nulos"], "media": media,
             "desviacion": np.sqrt(varianza), "min": a["min"], "max": a["max"],
             "negativos": a["negativos"], "ceros": a["ceros"], "suma": a["suma"]}
        )
    return pd.DataFrame(filas)


def _agregar_por_cliente(tx: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Resume las transacciones válidas de un bloque en una fila por cliente.

    `info` trae, alineada fila a fila con `tx`, la oferta del cliente de cada
    transacción (offerdate, category, company, brand)."""
    monto = tx["purchaseamount"]
    cantidad = tx["purchasequantity"]
    # compra real: excluye devoluciones (cantidad < 0) y líneas de descuento (monto <= 0)
    compra = (cantidad > 0) & (monto > 0)
    dias_antes = (info["offerdate"] - tx["date"]).dt.days

    coincide = {
        nombre: tx[col].to_numpy() == info[col].to_numpy()
        for nombre, col in DIMENSIONES_OFERTA.items()
    }
    coincide["producto"] = coincide["categoria"] & coincide["compania"] & coincide["marca"]

    sumas = {
        "n_lineas": np.ones(len(tx), dtype="int32"),
        "n_compras": compra,
        "n_devoluciones": cantidad < 0,
        "n_descuentos": (monto < 0) & (cantidad >= 0),
        "gasto_total": monto,
    }
    for nombre, mascara in coincide.items():
        compra_dim = compra & mascara
        sumas[f"n_compras_{nombre}"] = compra_dim
        sumas[f"gasto_{nombre}"] = monto.where(mascara, 0.0)
        if nombre != "producto":
            for dias in VENTANAS_DIAS:
                sumas[f"n_compras_{nombre}_{dias}d"] = compra_dim & (dias_antes <= dias)

    tabla = pd.DataFrame(sumas, index=tx.index)
    tabla["id"] = tx["id"]
    resumen = tabla.groupby("id").sum()

    fechas = pd.DataFrame(
        {
            "id": tx["id"],
            "date": tx["date"],
            "dias_cat": dias_antes.where(compra & coincide["categoria"]),
        }
    )
    extremos = fechas.groupby("id").agg(
        primera_compra=("date", "min"),
        ultima_compra=("date", "max"),
        dias_ultima_compra_categoria=("dias_cat", "min"),
    )

    # drop_duplicates + size es bastante más rápido que groupby().nunique()
    distintos = {
        "n_visitas": "date",
        "n_categorias_distintas": "category",
        "n_marcas_distintas": "brand",
        "n_companias_distintas": "company",
    }
    for nombre, col in distintos.items():
        resumen[nombre] = tx[["id", col]].drop_duplicates().groupby("id").size()

    return resumen.join(extremos)


class _PasadaTransacciones:
    """Estado acumulado durante el recorrido por bloques de transactions."""

    def __init__(self, clientes: pd.DataFrame, ruta_reducidas: Path | None):
        offers = cargar_offers()
        self.categorias_ofertadas = offers["category"].unique()
        self.companias_ofertadas = offers["company"].unique()
        self.info_clientes = clientes[["chain", "offerdate", "category", "company", "brand"]]
        self.ruta_reducidas = ruta_reducidas
        self.escritor = None
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.conteos = dict.fromkeys(CONCEPTOS_CONTEO, 0)
        self.estadisticas = {}
        self.medidas, self.descuentos, self.mensual = [], [], []
        self.muestras, self.agregados = [], []
        self.vistos = set()

    def procesar(self, tx: pd.DataFrame) -> None:
        if tx.empty:
            return
        c = self.conteos
        c["filas_leidas"] += len(tx)

        ids_bloque = tx["id"].unique().tolist()
        c["clientes_no_contiguos"] += len(self.vistos.intersection(ids_bloque))
        self.vistos.update(ids_bloque)

        self._describir_crudo(tx)

        info = self.info_clientes.reindex(tx["id"].to_numpy())
        info.index = tx.index
        en_clientes = info["chain"].notna()
        misma_cadena = tx["chain"] == info["chain"]
        antes_oferta = tx["date"] < info["offerdate"]
        c["filas_sin_cliente"] += int((~en_clientes).sum())
        c["filas_otra_cadena"] += int((en_clientes & ~misma_cadena).sum())
        c["filas_en_o_despues_oferta"] += int((en_clientes & misma_cadena & ~antes_oferta).sum())

        valido = en_clientes & misma_cadena & antes_oferta
        c["filas_validas"] += int(valido.sum())
        tx, info = tx[valido], info[valido]
        if tx.empty:
            return

        reducida = tx["category"].isin(self.categorias_ofertadas) | tx["company"].isin(self.companias_ofertadas)
        c["filas_reducidas"] += int(reducida.sum())
        if self.ruta_reducidas is not None:
            self._escribir_reducidas(tx[reducida])

        self.agregados.append(_agregar_por_cliente(tx, info))

    def _describir_crudo(self, tx: pd.DataFrame) -> None:
        """Conteos de calidad, estadísticas y muestra sobre las filas tal como vienen."""
        c = self.conteos
        monto, cantidad = tx["purchaseamount"], tx["purchasequantity"]
        es_descuento = (monto < 0) & (cantidad >= 0)
        c["filas_devolucion"] += int((cantidad < 0).sum())
        c["filas_descuento"] += int(es_descuento.sum())
        c["filas_monto_cero"] += int((monto == 0).sum())
        c["filas_cantidad_cero"] += int((cantidad == 0).sum())
        c["filas_productsize_cero_o_nulo"] += int(((tx["productsize"] == 0) | tx["productsize"].isna()).sum())
        c["filas_productmeasure_nulo"] += int(tx["productmeasure"].isna().sum())

        _acumular_estadisticas(self.estadisticas, tx)
        self.medidas.append(tx["productmeasure"].fillna("SIN_MEDIDA").value_counts())
        self.descuentos.append(tx[es_descuento].groupby(["dept", "category"]).size())
        self.mensual.append(
            tx.groupby(tx["date"].dt.to_period("M")).agg(
                n_lineas=("id", "size"),
                monto=("purchaseamount", "sum"),
                n_clientes=("id", "nunique"),
            )
        )
        self.muestras.append(tx[self.rng.random(len(tx)) < FRACCION_MUESTRA])

    def _escribir_reducidas(self, tx: pd.DataFrame) -> None:
        if pa is None or pq is None:
            raise ImportError(
                "pyarrow es necesario para escribir transacciones reducidas en Parquet."
            )
        tabla = pa.Table.from_pandas(tx, preserve_index=False)
        if self.escritor is None:
            self.escritor = pq.ParquetWriter(self.ruta_reducidas, tabla.schema)
        else:
            tabla = tabla.cast(self.escritor.schema)
        self.escritor.write_table(tabla)

    def resultados(self, bloques: int, segundos: float) -> dict:
        if self.escritor is not None:
            self.escritor.close()
        if self.conteos["clientes_no_contiguos"]:
            raise RuntimeError(
                f"{self.conteos['clientes_no_contiguos']} clientes aparecen en bloques no contiguos; "
                "el supuesto de filas agrupadas por cliente no se cumple"
            )

        conteos = pd.Series(
            {**self.conteos, "bloques": bloques, "segundos": round(segundos)}, name="valor"
        ).rename_axis("concepto")

        mensual = pd.concat(self.mensual).groupby(level=0).sum()
        mensual.index = mensual.index.astype(str)

        medidas = pd.concat(self.medidas).groupby(level=0).sum().sort_values(ascending=False)
        medidas = medidas.rename_axis("productmeasure").rename("n_lineas").reset_index()
        medidas["proporcion"] = medidas["n_lineas"] / medidas["n_lineas"].sum()

        descuentos = pd.concat(self.descuentos).groupby(level=[0, 1]).sum().sort_values(ascending=False)
        descuentos = descuentos.rename("n_lineas").reset_index()
        descuentos["proporcion"] = descuentos["n_lineas"] / descuentos["n_lineas"].sum()
        descuentos["categoria_ofertada"] = descuentos["category"].isin(self.categorias_ofertadas)

        return {
            "features": pd.concat(self.agregados),
            "conteos": conteos,
            "estadisticas": _tabla_estadisticas(self.estadisticas),
            "productmeasure": medidas,
            "mensual": mensual.rename_axis("mes").reset_index(),
            "descuentos": descuentos,
            "muestra": pd.concat(self.muestras, ignore_index=True),
        }


def recorrer_transacciones(
    clientes: pd.DataFrame,
    ruta_reducidas: Path | None = None,
    max_bloques: int | None = None,
    verbose: bool = True,
) -> dict:
    """Recorre transactions.csv.gz por bloques. `max_bloques` limita la lectura
    para pruebas rápidas; con None se lee el archivo completo."""
    pasada = _PasadaTransacciones(clientes, ruta_reducidas)
    lector = pd.read_csv(
        TRANSACTIONS_PATH, compression="gzip", dtype=TRANSACTIONS_DTYPES, chunksize=TAMANO_BLOQUE
    )
    inicio = time.time()
    pendiente = None
    bloques = 0
    for bloque in lector:
        bloques += 1
        bloque["date"] = pd.to_datetime(bloque["date"], format="%Y-%m-%d")
        if pendiente is not None:
            bloque = pd.concat([pendiente, bloque], ignore_index=True)
        # el último cliente del bloque puede continuar en el siguiente: se retiene
        es_ultimo = bloque["id"].to_numpy() == bloque["id"].iat[-1]
        pendiente = bloque[es_ultimo]
        pasada.procesar(bloque[~es_ultimo])

        if verbose and bloques % 10 == 0:
            filas = pasada.conteos["filas_leidas"]
            print(f"  bloque {bloques}: {filas:,} filas procesadas en {time.time() - inicio:,.0f}s", flush=True)
        if max_bloques is not None and bloques >= max_bloques:
            break
    if pendiente is not None:
        pasada.procesar(pendiente)
    return pasada.resultados(bloques, time.time() - inicio)


def construir_dataset(clientes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Une las variables por cliente con su oferta y resuelve los faltantes.

    Un cliente sin compras de la categoría, compañía o marca ofertada no tiene
    un dato faltante sino un cero real: los conteos y montos se completan con 0 y
    se agrega la bandera nunca_compro_*. La recencia de categoría queda vacía en
    esos clientes porque ningún número de días describe una compra que no ocurrió."""
    ds = clientes.join(features, how="left")

    conteos = [c for c in features.columns if c.startswith("n_")]
    ds[conteos] = ds[conteos].fillna(0).astype("int64")
    gastos = [c for c in features.columns if c.startswith("gasto_")]
    ds[gastos] = ds[gastos].fillna(0.0).round(2)

    ds["ticket_promedio"] = (ds["gasto_total"] / ds["n_visitas"].where(ds["n_visitas"] > 0)).round(2)
    ds["dias_ultima_compra"] = (ds["offerdate"] - ds["ultima_compra"]).dt.days.astype("Int64")
    ds["antiguedad_dias"] = (ds["offerdate"] - ds["primera_compra"]).dt.days.astype("Int64")
    ds["dias_ultima_compra_categoria"] = ds["dias_ultima_compra_categoria"].astype("Int64")
    for dim in [*DIMENSIONES_OFERTA, "producto"]:
        ds[f"nunca_compro_{dim}"] = (ds[f"n_compras_{dim}"] == 0).astype("int8")

    columnas = COLUMNAS_CLIENTE + ["split"] + COLUMNAS_OBJETIVO + COLUMNAS_GENERALES + COLUMNAS_OFERTA
    return ds.reset_index()[columnas]


def procesar_transacciones(forzar: bool = False, verbose: bool = True) -> dict:
    """Ejecuta la pasada completa y guarda todas las salidas. Si ya existen y
    `forzar` es False, solo las carga de disco (la pasada tarda varios minutos)."""
    if not forzar and all(p.exists() for p in SALIDAS):
        if verbose:
            print("Las salidas ya existen: se cargan de disco. Usar forzar=True para recalcular.")
        return cargar_resultados()

    clientes = cargar_clientes()
    res = recorrer_transacciones(clientes, ruta_reducidas=TRANSACCIONES_REDUCIDAS_PATH, verbose=verbose)
    dataset = construir_dataset(clientes, res["features"])

    conteos = res["conteos"]
    conteos["clientes_con_transacciones"] = len(res["features"])
    conteos["clientes_sin_transacciones"] = len(clientes) - len(res["features"])

    DATASET_MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train = dataset[dataset["split"] == "train"].drop(columns="split")
    train = train.astype({"repeattrips": "int32", "repeater": "bool"})
    test = dataset[dataset["split"] == "test"].drop(columns=["split", *COLUMNAS_OBJETIVO])
    train.to_csv(DATASET_MODELO_PATH, index=False)
    test.to_csv(DATASET_MODELO_TEST_PATH, index=False)
    res["muestra"].to_csv(TRANSACCIONES_MUESTRA_PATH, index=False)

    conteos.to_csv(TABLA_CONTEOS)
    res["estadisticas"].to_csv(TABLA_ESTADISTICAS, index=False)
    res["productmeasure"].to_csv(TABLA_PRODUCTMEASURE, index=False)
    res["mensual"].to_csv(TABLA_MENSUAL, index=False)
    res["descuentos"].to_csv(TABLA_DESCUENTOS, index=False)
    return cargar_resultados()


def cargar_dataset_modelo(test: bool = False) -> pd.DataFrame:
    """Carga dataset_modelo.csv (train, con repeater) o dataset_modelo_test.csv."""
    ruta = DATASET_MODELO_TEST_PATH if test else DATASET_MODELO_PATH
    return pd.read_csv(ruta, parse_dates=["offerdate"])


def cargar_resultados() -> dict:
    """Carga de disco todas las salidas de procesar_transacciones()."""
    return {
        "dataset": cargar_dataset_modelo(),
        "dataset_test": cargar_dataset_modelo(test=True),
        "conteos": pd.read_csv(TABLA_CONTEOS, index_col="concepto")["valor"],
        "estadisticas": pd.read_csv(TABLA_ESTADISTICAS),
        "productmeasure": pd.read_csv(TABLA_PRODUCTMEASURE),
        "mensual": pd.read_csv(TABLA_MENSUAL),
        "descuentos": pd.read_csv(TABLA_DESCUENTOS),
        "muestra": pd.read_csv(TRANSACCIONES_MUESTRA_PATH, parse_dates=["date"]),
    }
