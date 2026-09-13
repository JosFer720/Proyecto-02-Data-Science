"""Funciones para entrenar y evaluar los modelos de compradores recurrentes.

La validacion siempre respeta el orden de ``offerdate``. Las transformaciones
se ajustan solo con el bloque de entrenamiento para evitar fuga de informacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


COLUMNAS_LINEA_BASE = [
    "offervalue",
    "quantity",
    "dias_ultima_compra",
    "n_visitas",
    "gasto_total",
    "ticket_promedio",
    "antiguedad_dias",
]

COLUMNAS_CATEGORICAS_MEJORADAS = [
    "chain",
    "market",
    "category",
    "company",
    "brand",
]

COLUMNAS_DERIVADAS = [
    "visitas_por_30d",
    "compras_por_visita",
    "gasto_por_30d",
    "proporcion_compras_categoria",
    "proporcion_compras_marca",
    "proporcion_compras_producto",
    "proporcion_categoria_30_en_180d",
    "recencia_categoria_relativa",
    "lineas_por_30d",
    "volumen_extremo",
]

COLUMNAS_EXCLUIDAS = {
    "id",
    "offer",
    "offerdate",
    "repeattrips",
    "repeater",
    *COLUMNAS_CATEGORICAS_MEJORADAS,
}


@dataclass(frozen=True)
class DivisionTemporal:
    """Contenedores y metadatos de una particion cronologica."""

    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    fecha_corte: pd.Timestamp


def dividir_temporalmente(
    datos: pd.DataFrame,
    proporcion_validacion: float = 0.20,
    columna_fecha: str = "offerdate",
) -> DivisionTemporal:
    """Separa las fechas mas recientes y nunca divide un mismo dia.

    El punto de corte se aproxima a la proporcion solicitada. Todas las filas
    con la fecha de corte pasan a validacion para que un dia no aparezca en los
    dos conjuntos.
    """
    if not 0 < proporcion_validacion < 1:
        raise ValueError("proporcion_validacion debe estar entre 0 y 1")
    if columna_fecha not in datos.columns:
        raise KeyError(f"Falta la columna {columna_fecha}")

    ordenados = datos.copy()
    ordenados[columna_fecha] = pd.to_datetime(ordenados[columna_fecha])
    ordenados = ordenados.sort_values(columna_fecha, kind="stable").reset_index(drop=True)
    posicion = int(len(ordenados) * (1 - proporcion_validacion))
    fecha_corte = ordenados.loc[posicion, columna_fecha]

    entrenamiento = ordenados[ordenados[columna_fecha] < fecha_corte].copy()
    validacion = ordenados[ordenados[columna_fecha] >= fecha_corte].copy()
    if entrenamiento.empty or validacion.empty:
        raise ValueError("La division temporal produjo un conjunto vacio")
    if entrenamiento[columna_fecha].max() >= validacion[columna_fecha].min():
        raise AssertionError("Las fechas de entrenamiento y validacion se traslapan")

    return DivisionTemporal(entrenamiento, validacion, fecha_corte)


def crear_variables_avanzadas(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega tasas y proporciones comparables entre historiales distintos."""
    trabajo = datos.copy()
    dias = trabajo["antiguedad_dias"].clip(lower=1)
    visitas = trabajo["n_visitas"].replace(0, np.nan)
    compras = trabajo["n_compras"].replace(0, np.nan)
    compras_180 = trabajo["n_compras_categoria_180d"].replace(0, np.nan)

    trabajo["visitas_por_30d"] = 30.44 * trabajo["n_visitas"] / dias
    trabajo["compras_por_visita"] = trabajo["n_compras"] / visitas
    trabajo["gasto_por_30d"] = 30.44 * trabajo["gasto_total"] / dias
    trabajo["proporcion_compras_categoria"] = trabajo["n_compras_categoria"] / compras
    trabajo["proporcion_compras_marca"] = trabajo["n_compras_marca"] / compras
    trabajo["proporcion_compras_producto"] = trabajo["n_compras_producto"] / compras
    trabajo["proporcion_categoria_30_en_180d"] = (
        trabajo["n_compras_categoria_30d"] / compras_180
    )
    trabajo["recencia_categoria_relativa"] = (
        trabajo["dias_ultima_compra_categoria"] / dias
    )
    trabajo["lineas_por_30d"] = 30.44 * trabajo["n_lineas"] / dias
    trabajo["volumen_extremo"] = (trabajo["lineas_por_30d"] > 750).astype("int8")
    return trabajo


def _logaritmo_con_signo(valores: np.ndarray) -> np.ndarray:
    """Comprime colas largas y conserva los pocos montos netos negativos."""
    return np.sign(valores) * np.log1p(np.abs(valores))


def crear_pipeline_linea_base() -> Pipeline:
    """Regresion logistica sencilla sobre oferta y variables RFM."""
    numericas = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler()),
        ]
    )
    preparacion = ColumnTransformer(
        [("numericas", numericas, COLUMNAS_LINEA_BASE)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preparacion", preparacion),
            (
                "modelo",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )


def columnas_numericas_mejoradas(datos: pd.DataFrame) -> list[str]:
    """Selecciona comportamiento, condiciones de oferta y variables derivadas."""
    return [
        columna
        for columna in datos.select_dtypes(include=[np.number]).columns
        if columna not in COLUMNAS_EXCLUIDAS
    ]


def crear_pipeline_mejorado(datos: pd.DataFrame) -> Pipeline:
    """Regresion logistica con todo el comportamiento y categorias codificadas."""
    columnas_numericas = columnas_numericas_mejoradas(datos)
    numericas = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "logaritmo",
                FunctionTransformer(_logaritmo_con_signo, feature_names_out="one-to-one"),
            ),
            ("escalar", StandardScaler()),
        ]
    )
    categoricas = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            (
                "codificar",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
            ),
        ]
    )
    preparacion = ColumnTransformer(
        transformers=[
            ("numericas", numericas, columnas_numericas),
            ("categoricas", categoricas, COLUMNAS_CATEGORICAS_MEJORADAS),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preparacion", preparacion),
            (
                "modelo",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    C=0.25,
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluar_auc(modelo: Pipeline, datos: pd.DataFrame) -> dict:
    """Calcula probabilidades, AUC y puntos de la curva ROC."""
    y_real = datos["repeater"].astype(int)
    probabilidades = modelo.predict_proba(datos)[:, 1]
    auc = roc_auc_score(y_real, probabilidades)
    fpr, tpr, umbrales = roc_curve(y_real, probabilidades)
    return {
        "auc": float(auc),
        "probabilidades": probabilidades,
        "fpr": fpr,
        "tpr": tpr,
        "umbrales": umbrales,
    }


def tabla_coeficientes(modelo: Pipeline) -> pd.DataFrame:
    """Devuelve los coeficientes con el nombre generado por el preprocesamiento."""
    nombres = modelo.named_steps["preparacion"].get_feature_names_out()
    nombres = [
        nombre.replace("numericas__", "").replace("categoricas__", "")
        for nombre in nombres
    ]
    coeficientes = modelo.named_steps["modelo"].coef_.ravel()
    tabla = pd.DataFrame({"variable": nombres, "coeficiente": coeficientes})
    tabla["magnitud"] = tabla["coeficiente"].abs()
    return tabla.sort_values("magnitud", ascending=False).reset_index(drop=True)


def guardar_modelo(modelo: Pipeline, ruta: Path) -> None:
    """Guarda un pipeline completo, incluidas sus transformaciones ajustadas."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
