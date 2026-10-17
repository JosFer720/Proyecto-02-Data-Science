"""Evaluacion reproducible de modelos binarios de compradores recurrentes.

Las funciones de este modulo no ajustan los modelos congelados. La unica
excepcion es :func:`validar_por_ofertas`, que crea un pipeline nuevo para cada
algoritmo y division mediante la factoria que recibe.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve as sklearn_roc_curve,
)
from sklearn.model_selection import GroupKFold


def _etiquetas_binarias(y_real: Sequence[int] | pd.Series | np.ndarray) -> np.ndarray:
    """Normaliza etiquetas 0/1 y rechaza objetivos incompatibles."""
    etiquetas = np.asarray(y_real).reshape(-1)
    if etiquetas.size == 0:
        raise ValueError("y_real no puede estar vacio")
    if not np.isin(etiquetas, [0, 1, False, True]).all():
        raise ValueError("y_real debe contener solamente etiquetas binarias 0 y 1")
    return etiquetas.astype(int)


def _probabilidades_binarias(
    probabilidades: Sequence[float] | pd.Series | np.ndarray, n_esperado: int
) -> np.ndarray:
    """Normaliza y valida probabilidades de la clase positiva."""
    valores = np.asarray(probabilidades, dtype=float).reshape(-1)
    if len(valores) != n_esperado:
        raise ValueError("y_real y probabilidades deben tener la misma longitud")
    if not np.isfinite(valores).all() or ((valores < 0) | (valores > 1)).any():
        raise ValueError("Las probabilidades deben ser valores finitos entre 0 y 1")
    return valores


def _validar_umbral(umbral: float) -> float:
    corte = float(umbral)
    if not 0 <= corte <= 1:
        raise ValueError("umbral debe estar entre 0 y 1")
    return corte


def metricas_binarias(
    y_real: Sequence[int] | pd.Series | np.ndarray,
    probabilidades: Sequence[float] | pd.Series | np.ndarray,
    umbral: float = 0.5,
) -> dict[str, float | int]:
    """Calcula metricas binarias y matriz TN, FP, FN, TP para un umbral fijo.

    Si la muestra tiene una sola clase, AUC se informa como ``np.nan``: no es
    una estimacion definida y no se sustituye por una cifra enganosa.
    """
    etiquetas = _etiquetas_binarias(y_real)
    proba = _probabilidades_binarias(probabilidades, len(etiquetas))
    corte = _validar_umbral(umbral)
    predicciones = (proba >= corte).astype(int)
    tn, fp, fn, tp = confusion_matrix(etiquetas, predicciones, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(etiquetas, proba)) if np.unique(etiquetas).size == 2 else np.nan

    return {
        "auc": auc,
        "precision": float(precision_score(etiquetas, predicciones, zero_division=0)),
        "sensibilidad": float(recall_score(etiquetas, predicciones, zero_division=0)),
        "f1": float(f1_score(etiquetas, predicciones, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "umbral": corte,
    }


def curva_roc(
    y_real: Sequence[int] | pd.Series | np.ndarray,
    probabilidades: Sequence[float] | pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Devuelve los puntos de la curva ROC en una tabla normalizada."""
    etiquetas = _etiquetas_binarias(y_real)
    proba = _probabilidades_binarias(probabilidades, len(etiquetas))
    if np.unique(etiquetas).size != 2:
        raise ValueError("No se puede calcular una curva ROC con una sola clase")
    fpr, tpr, umbrales = sklearn_roc_curve(etiquetas, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "umbral": umbrales})


def _extraer_probabilidades(modelo: Any, datos: pd.DataFrame) -> np.ndarray:
    """Lee probabilidades tanto de ModeloCongelado como de estimadores sklearn."""
    if hasattr(modelo, "predecir_proba"):
        probabilidades = modelo.predecir_proba(datos)
    elif hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(datos)
        probabilidades = np.asarray(probabilidades)
        if probabilidades.ndim == 2:
            if probabilidades.shape[1] < 2:
                raise ValueError("predict_proba debe incluir la probabilidad de la clase positiva")
            probabilidades = probabilidades[:, 1]
    else:
        raise TypeError("El modelo debe exponer predecir_proba o predict_proba")
    return _probabilidades_binarias(probabilidades, len(datos))


def _resolver_objetivo(
    datos: pd.DataFrame,
    y_real: str | Sequence[int] | pd.Series | np.ndarray,
) -> np.ndarray:
    if isinstance(y_real, str):
        if y_real not in datos.columns:
            raise KeyError(f"Falta la columna objetivo {y_real}")
        return _etiquetas_binarias(datos[y_real])
    return _etiquetas_binarias(y_real)


def evaluar_modelo_congelado(
    nombre: str,
    modelo: Any,
    datos: pd.DataFrame,
    y_real: str | Sequence[int] | pd.Series | np.ndarray = "repeater",
    umbral: float | None = None,
) -> dict[str, Any]:
    """Evalua un modelo ya entrenado sin llamar nunca a ``fit``.

    El umbral por defecto procede de ``modelo.ficha['umbral']`` para conservar
    el valor elegido antes de la semana de prueba.
    """
    etiquetas = _resolver_objetivo(datos, y_real)
    ficha = getattr(modelo, "ficha", {})
    if umbral is None:
        if "umbral" not in ficha:
            raise ValueError("El modelo no tiene umbral en su ficha; indiquelo explicitamente")
        umbral = ficha["umbral"]
    corte = _validar_umbral(umbral)

    inicio = time.perf_counter()
    probabilidades = _extraer_probabilidades(modelo, datos)
    tiempo_prediccion = time.perf_counter() - inicio
    resultado = metricas_binarias(etiquetas, probabilidades, corte)
    resultado.update(
        {
            "modelo": nombre,
            "algoritmo": ficha.get("algoritmo", nombre),
            "tiempo_entrenamiento_s": ficha.get("tiempo_entrenamiento_s"),
            "tiempo_prediccion_s": tiempo_prediccion,
            "probabilidades": probabilidades,
            "clasificaciones": (probabilidades >= corte).astype(int),
        }
    )
    return resultado


def _iterar_splits(
    datos: pd.DataFrame,
    grupos: pd.Series,
    n_splits: int,
    splits: Iterable[tuple[Sequence[int], Sequence[int]]] | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if splits is not None:
        divisiones = [(np.asarray(entrenamiento), np.asarray(prueba)) for entrenamiento, prueba in splits]
        if not divisiones:
            raise ValueError("splits debe incluir al menos una division")
        return divisiones
    if n_splits < 5:
        raise ValueError("n_splits debe ser al menos 5 para validar por ofertas")
    if grupos.nunique() < n_splits:
        raise ValueError("Se requieren al menos n_splits ofertas distintas")
    divisor = GroupKFold(n_splits=n_splits)
    return list(divisor.split(datos, groups=grupos))


def validar_por_ofertas(
    datos: pd.DataFrame,
    algoritmos: Iterable[str] | Mapping[str, Any],
    factory: Callable[[str, pd.DataFrame], Any],
    *,
    columna_grupo: str = "offer",
    columna_objetivo: str = "repeater",
    n_splits: int = 5,
    splits: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
    umbral: float = 0.5,
) -> pd.DataFrame:
    """Reentrena pipelines temporales y los prueba en ofertas no vistas.

    ``factory`` recibe ``(algoritmo, entrenamiento)`` y debe devolver un
    pipeline nuevo. ``splits`` permite inyectar divisiones pequenas y
    deterministas en pruebas; sin el se usa ``GroupKFold`` con al menos cinco
    pliegues.
    """
    if columna_grupo not in datos.columns:
        raise KeyError(f"Falta la columna de ofertas {columna_grupo}")
    if columna_objetivo not in datos.columns:
        raise KeyError(f"Falta la columna objetivo {columna_objetivo}")
    if datos.empty:
        raise ValueError("datos no puede estar vacio")

    corte = _validar_umbral(umbral)
    grupos = datos[columna_grupo]
    if grupos.isna().any():
        raise ValueError("La columna de ofertas no puede contener valores faltantes")
    nombres_algoritmos = list(algoritmos.keys() if isinstance(algoritmos, Mapping) else algoritmos)
    if not nombres_algoritmos:
        raise ValueError("Debe indicarse al menos un algoritmo")

    divisiones_preparadas: list[tuple[int, pd.DataFrame, pd.DataFrame, set[Any], set[Any]]] = []
    for indice_split, (indices_entrenamiento, indices_prueba) in enumerate(
        _iterar_splits(datos, grupos, n_splits, splits), start=1
    ):
        entrenamiento = datos.iloc[indices_entrenamiento]
        prueba = datos.iloc[indices_prueba]
        ofertas_entrenamiento = set(entrenamiento[columna_grupo])
        ofertas_prueba = set(prueba[columna_grupo])
        if ofertas_entrenamiento.intersection(ofertas_prueba):
            raise AssertionError("Las ofertas de entrenamiento y prueba deben ser disjuntas")
        if entrenamiento.empty or prueba.empty:
            raise ValueError("Cada division debe tener entrenamiento y prueba")
        divisiones_preparadas.append(
            (indice_split, entrenamiento, prueba, ofertas_entrenamiento, ofertas_prueba)
        )

    filas: list[dict[str, Any]] = []
    for algoritmo in nombres_algoritmos:
        for indice_split, entrenamiento, prueba, ofertas_entrenamiento, ofertas_prueba in divisiones_preparadas:
            pipeline = factory(algoritmo, entrenamiento.copy())
            inicio = time.perf_counter()
            pipeline.fit(entrenamiento, entrenamiento[columna_objetivo].astype(int))
            tiempo_entrenamiento = time.perf_counter() - inicio
            inicio = time.perf_counter()
            probabilidades = _extraer_probabilidades(pipeline, prueba)
            tiempo_prediccion = time.perf_counter() - inicio
            metricas = metricas_binarias(prueba[columna_objetivo], probabilidades, corte)
            filas.append(
                {
                    "algoritmo": algoritmo,
                    "split": indice_split,
                    "n_entrenamiento": len(entrenamiento),
                    "n_prueba": len(prueba),
                    "n_ofertas_entrenamiento": len(ofertas_entrenamiento),
                    "n_ofertas_prueba": len(ofertas_prueba),
                    "grupos_disjuntos": True,
                    **metricas,
                    "tiempo_entrenamiento_s": tiempo_entrenamiento,
                    "tiempo_prediccion_s": tiempo_prediccion,
                }
            )

    return pd.DataFrame(filas)
