"""Funciones reutilizables para el análisis exploratorio del dataset de modelo."""

import numpy as np
import pandas as pd


def tabla_frecuencia(datos: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Calcula la frecuencia y la proporción de cada valor de una categoría."""
    if columna not in datos.columns:
        raise KeyError(f"La columna '{columna}' no existe en los datos.")

    tabla = (
        datos[columna]
        .value_counts(dropna=False)
        .rename_axis("grupo")
        .reset_index(name="frecuencia")
    )
    tabla["proporcion"] = tabla["frecuencia"] / len(datos)
    return tabla


def tasa_recompra_por_grupo(datos: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Resume tamaño, recompradores y tasa de recompra por cada grupo."""
    requeridas = {columna, "repeater"}
    faltantes = requeridas.difference(datos.columns)
    if faltantes:
        raise KeyError(f"Faltan las columnas requeridas: {sorted(faltantes)}")

    trabajo = datos[[columna, "repeater"]].copy()
    trabajo["repeater"] = trabajo["repeater"].astype(int)
    tabla = (
        trabajo.groupby(columna, dropna=False, observed=False)["repeater"]
        .agg(n="size", repeaters="sum", tasa_recompra="mean")
        .reset_index()
        .rename(columns={columna: "grupo"})
        .sort_values("n", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return tabla


def crear_rango_offervalue(offervalue: pd.Series) -> pd.Series:
    """Agrupa el valor de la oferta en rangos reproducibles de descuento."""
    return pd.cut(
        offervalue,
        bins=[-np.inf, 1.00, 1.50, 2.00, np.inf],
        labels=["<= 1.00", "1.00–1.50", "1.50–2.00", "> 2.00"],
        include_lowest=True,
    )


def matriz_correlacion_spearman(
    datos: pd.DataFrame, variables: list[str]
) -> pd.DataFrame:
    """Calcula Spearman entre variables de comportamiento y ``repeater``."""
    if "repeater" not in datos.columns:
        raise KeyError("La columna 'repeater' es necesaria para la correlación.")

    columnas = [*variables, "repeater"]
    faltantes = set(columnas).difference(datos.columns)
    if faltantes:
        raise KeyError(f"Faltan las columnas requeridas: {sorted(faltantes)}")

    trabajo = datos[columnas].copy()
    trabajo["repeater"] = trabajo["repeater"].astype(int)
    return trabajo.corr(method="spearman")


def resumen_correlaciones_repeater(matriz: pd.DataFrame) -> pd.DataFrame:
    """Ordena las correlaciones de cada variable respecto a ``repeater``."""
    if "repeater" not in matriz.columns:
        raise KeyError("La matriz no contiene la columna 'repeater'.")

    resumen = (
        matriz["repeater"]
        .drop(labels="repeater")
        .rename("correlacion_spearman")
        .rename_axis("variable")
        .reset_index()
    )
    resumen["correlacion_absoluta"] = resumen["correlacion_spearman"].abs()
    return resumen.sort_values(
        "correlacion_absoluta", ascending=False, kind="stable"
    ).reset_index(drop=True)
