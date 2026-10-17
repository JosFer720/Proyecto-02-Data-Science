"""Funciones para entrenar y evaluar los modelos de compradores recurrentes.

La validacion siempre respeta el orden de ``offerdate``. Las transformaciones
se ajustan solo con el bloque de entrenamiento para evitar fuga de informacion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


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


# ---------------------------------------------------------------------------
# Tres conjuntos temporales, cuatro algoritmos y ajuste de hiperparametros
# ---------------------------------------------------------------------------

FIN_ENTRENAMIENTO = "2013-04-16"
FIN_VALIDACION = "2013-04-23"

ALGORITMOS = (
    "regresion_logistica",
    "random_forest",
    "gradient_boosting",
    "perceptron_multicapa",
)

SEMILLA = 42


@dataclass(frozen=True)
class DivisionTresConjuntos:
    """Entrenamiento, validacion y prueba ordenados en el tiempo."""

    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    prueba: pd.DataFrame


def dividir_en_tres_conjuntos(
    datos: pd.DataFrame,
    fin_entrenamiento: str = FIN_ENTRENAMIENTO,
    fin_validacion: str = FIN_VALIDACION,
    columna_fecha: str = "offerdate",
) -> DivisionTresConjuntos:
    """Separa por fecha de oferta: hasta el 16 de abril, del 17 al 23 y del 24 en adelante."""
    fechas = pd.to_datetime(datos[columna_fecha])
    corte_1 = pd.Timestamp(fin_entrenamiento)
    corte_2 = pd.Timestamp(fin_validacion)
    if not corte_1 < corte_2:
        raise ValueError("fin_entrenamiento debe ser anterior a fin_validacion")

    entrenamiento = datos[fechas <= corte_1].copy()
    validacion = datos[(fechas > corte_1) & (fechas <= corte_2)].copy()
    prueba = datos[fechas > corte_2].copy()
    if min(len(entrenamiento), len(validacion), len(prueba)) == 0:
        raise ValueError("La division produjo un conjunto vacio")
    if len(entrenamiento) + len(validacion) + len(prueba) != len(datos):
        raise AssertionError("La division perdio o duplico filas")
    return DivisionTresConjuntos(entrenamiento, validacion, prueba)


def columnas_numericas_modelo(datos: pd.DataFrame) -> list[str]:
    """Numericas que reciben los modelos, incluidas las derivadas."""
    return columnas_numericas_mejoradas(crear_variables_avanzadas(datos.head(100)))


def columnas_entrada_modelo(datos: pd.DataFrame) -> list[str]:
    """Columnas crudas que el pipeline necesita, en el orden de dataset_modelo.csv."""
    requeridas = (
        set(columnas_numericas_modelo(datos)) - set(COLUMNAS_DERIVADAS)
    ) | set(COLUMNAS_CATEGORICAS_MEJORADAS)
    return [columna for columna in datos.columns if columna in requeridas]


def _codificador_categorias() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            (
                "codificar",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=50, sparse_output=False
                ),
            ),
        ]
    )


def _preparacion_lineal(columnas_numericas: list[str]) -> ColumnTransformer:
    """Imputacion, logaritmo, escalado y one-hot para regresion y red neuronal."""
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
    return ColumnTransformer(
        [
            ("numericas", numericas, columnas_numericas),
            ("categoricas", _codificador_categorias(), COLUMNAS_CATEGORICAS_MEJORADAS),
        ],
        remainder="drop",
    )


def _preparacion_arboles(columnas_numericas: list[str]) -> ColumnTransformer:
    """Imputacion y one-hot sin escalar, porque los arboles no lo necesitan."""
    return ColumnTransformer(
        [
            (
                "numericas",
                SimpleImputer(strategy="median", add_indicator=True),
                columnas_numericas,
            ),
            ("categoricas", _codificador_categorias(), COLUMNAS_CATEGORICAS_MEJORADAS),
        ],
        remainder="drop",
    )


def _preparacion_boosting(columnas_numericas: list[str]) -> ColumnTransformer:
    """Numericas con sus faltantes y categorias ordinales que el modelo lee como grupos."""
    categoricas = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
        min_frequency=50,
    )
    return ColumnTransformer(
        [
            ("numericas", "passthrough", columnas_numericas),
            ("categoricas", categoricas, COLUMNAS_CATEGORICAS_MEJORADAS),
        ],
        remainder="drop",
    )


def crear_pipeline_modelo(
    algoritmo: str, columnas_numericas: list[str], **hiperparametros
) -> Pipeline:
    """Pipeline completo de un algoritmo: variables derivadas, preparacion y clasificador.

    Recibe las columnas de dataset_modelo.csv y devuelve probabilidades. Los
    hiperparametros reemplazan los valores por defecto del clasificador.
    """
    n_numericas = len(columnas_numericas)
    if algoritmo == "regresion_logistica":
        preparacion = _preparacion_lineal(columnas_numericas)
        base = {"solver": "liblinear", "max_iter": 1_000, "random_state": SEMILLA}
        clasificador = LogisticRegression(**{**base, **hiperparametros})
    elif algoritmo == "random_forest":
        preparacion = _preparacion_arboles(columnas_numericas)
        base = {"n_jobs": -1, "random_state": SEMILLA}
        clasificador = RandomForestClassifier(**{**base, **hiperparametros})
    elif algoritmo == "gradient_boosting":
        preparacion = _preparacion_boosting(columnas_numericas)
        # las categorias quedan al final del bloque numerico
        base = {
            "categorical_features": list(
                range(n_numericas, n_numericas + len(COLUMNAS_CATEGORICAS_MEJORADAS))
            ),
            "random_state": SEMILLA,
        }
        clasificador = HistGradientBoostingClassifier(**{**base, **hiperparametros})
    elif algoritmo == "perceptron_multicapa":
        preparacion = _preparacion_lineal(columnas_numericas)
        base = {"random_state": SEMILLA}
        clasificador = MLPClassifier(**{**base, **hiperparametros})
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    return Pipeline(
        steps=[
            ("variables", FunctionTransformer(crear_variables_avanzadas)),
            ("preparacion", preparacion),
            ("modelo", clasificador),
        ]
    )


def espacio_de_busqueda(algoritmo: str) -> dict:
    """Rangos de la seccion 5.4 de la revision bibliografica."""
    if algoritmo == "regresion_logistica":
        return {
            "modelo__C": loguniform(1e-3, 10),
            "modelo__penalty": ["l1", "l2"],
            "modelo__class_weight": ["balanced", None],
        }
    if algoritmo == "random_forest":
        return {
            "modelo__n_estimators": randint(300, 801),
            "modelo__max_depth": [8, 12, 16, 24],
            "modelo__min_samples_leaf": randint(1, 51),
            "modelo__max_features": ["sqrt", "log2", 0.2, 0.4],
            "modelo__class_weight": ["balanced", "balanced_subsample", None],
        }
    if algoritmo == "gradient_boosting":
        return {
            "modelo__learning_rate": loguniform(0.02, 0.2),
            "modelo__max_leaf_nodes": randint(15, 128),
            "modelo__min_samples_leaf": randint(20, 201),
            "modelo__l2_regularization": loguniform(1e-3, 10),
            "modelo__class_weight": ["balanced", None],
            "modelo__max_iter": [500],
            "modelo__early_stopping": [True],
            "modelo__n_iter_no_change": [20],
        }
    if algoritmo == "perceptron_multicapa":
        return {
            "modelo__hidden_layer_sizes": [(64,), (128, 64), (64, 32)],
            "modelo__alpha": loguniform(1e-5, 1e-2),
            "modelo__learning_rate_init": loguniform(1e-4, 1e-2),
            "modelo__max_iter": [200],
            "modelo__early_stopping": [True],
            "modelo__n_iter_no_change": [10],
        }
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def _sin_prefijo(parametros: dict) -> dict:
    """Quita el prefijo modelo__ de los parametros elegidos por la busqueda."""
    return {
        nombre.removeprefix("modelo__"): valor.item() if isinstance(valor, np.generic) else valor
        for nombre, valor in parametros.items()
    }


def buscar_hiperparametros(
    algoritmo: str,
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    n_iter: int,
    semilla: int = SEMILLA,
) -> tuple[pd.DataFrame, dict]:
    """Busqueda aleatoria: ajusta con entrenamiento y puntua con la semana de validacion.

    Devuelve la tabla de combinaciones probadas y los mejores hiperparametros.
    """
    columnas_numericas = columnas_numericas_modelo(entrenamiento)
    conjunto = pd.concat([entrenamiento, validacion], ignore_index=True)
    pliegue = np.r_[np.full(len(entrenamiento), -1), np.zeros(len(validacion), dtype=int)]
    busqueda = RandomizedSearchCV(
        crear_pipeline_modelo(algoritmo, columnas_numericas),
        espacio_de_busqueda(algoritmo),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=PredefinedSplit(pliegue),
        refit=False,
        random_state=semilla,
        error_score="raise",
        verbose=1,
    )
    busqueda.fit(conjunto, conjunto["repeater"].astype(int))

    resultados = pd.DataFrame(busqueda.cv_results_)
    tabla = pd.DataFrame(
        {
            "algoritmo": algoritmo,
            "combinacion": range(1, len(resultados) + 1),
            "hiperparametros": [
                str(_sin_prefijo(parametros)) for parametros in resultados["params"]
            ],
            "auc_validacion": resultados["mean_test_score"],
            "tiempo_ajuste_s": resultados["mean_fit_time"],
            "elegida": resultados["rank_test_score"] == 1,
        }
    )
    return tabla, _sin_prefijo(busqueda.best_params_)


def ajustar_y_evaluar(
    algoritmo: str,
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    **hiperparametros,
) -> dict:
    """Entrena una vez con el bloque de entrenamiento y mide AUC y tiempos en validacion."""
    pipeline = crear_pipeline_modelo(
        algoritmo, columnas_numericas_modelo(entrenamiento), **hiperparametros
    )
    inicio = time.perf_counter()
    pipeline.fit(entrenamiento, entrenamiento["repeater"].astype(int))
    tiempo_entrenamiento = time.perf_counter() - inicio

    inicio = time.perf_counter()
    probabilidades = pipeline.predict_proba(validacion)[:, 1]
    tiempo_prediccion = time.perf_counter() - inicio
    return {
        "pipeline": pipeline,
        "probabilidades": probabilidades,
        "auc": float(roc_auc_score(validacion["repeater"].astype(int), probabilidades)),
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "tiempo_prediccion_s": tiempo_prediccion,
    }


def umbral_maximo_f1(y_real: pd.Series, probabilidades: np.ndarray) -> float:
    """Umbral que maximiza el F1 de la clase recompra, elegido sin usar la prueba."""
    precision, sensibilidad, umbrales = precision_recall_curve(y_real, probabilidades)
    suma = np.clip(precision[:-1] + sensibilidad[:-1], 1e-12, None)
    f1 = 2 * precision[:-1] * sensibilidad[:-1] / suma
    return float(umbrales[int(np.argmax(f1))])


def importancia_por_permutacion(
    predecir: Callable[[pd.DataFrame], np.ndarray],
    datos: pd.DataFrame,
    columnas: list[str],
    n_repeticiones: int = 5,
    semilla: int = SEMILLA,
) -> pd.DataFrame:
    """Perdida de AUC al permutar cada columna cruda y recalcular las derivadas.

    ``predecir`` recibe un DataFrame con las columnas de dataset_modelo.csv y
    devuelve la probabilidad de recompra.
    """
    y_real = datos["repeater"].astype(int)
    auc_original = roc_auc_score(y_real, predecir(datos))
    generador = np.random.default_rng(semilla)

    filas = []
    for columna in columnas:
        perdidas = []
        for _ in range(n_repeticiones):
            permutado = datos.copy()
            permutado[columna] = generador.permutation(permutado[columna].to_numpy())
            perdidas.append(auc_original - roc_auc_score(y_real, predecir(permutado)))
        filas.append(
            {
                "variable": columna,
                "perdida_auc": float(np.mean(perdidas)),
                "desviacion": float(np.std(perdidas)),
            }
        )
    tabla = pd.DataFrame(filas).sort_values("perdida_auc", ascending=False)
    tabla["auc_sin_permutar"] = float(auc_original)
    return tabla.reset_index(drop=True)
