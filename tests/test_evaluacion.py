"""Contrato observable de la biblioteca de evaluacion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluacion import (
    curva_roc,
    evaluar_modelo_congelado,
    metricas_binarias,
    validar_por_ofertas,
)


class ModeloCongeladoDePrueba:
    """Doble minimo de la API publica de ModeloCongelado."""

    ficha = {"umbral": 0.7, "tiempo_entrenamiento_s": 12.5, "algoritmo": "prueba"}

    def predecir_proba(self, datos: pd.DataFrame) -> np.ndarray:
        return datos["probabilidad"].to_numpy()

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Un modelo congelado no debe reentrenarse al evaluarlo")


class PipelineTemporal:
    """Estimador pequeno cuya prediccion depende de una columna real de la muestra."""

    def fit(self, datos: pd.DataFrame, objetivo: pd.Series) -> "PipelineTemporal":
        self.clases_vistas = set(objetivo.astype(int))
        return self

    def predict_proba(self, datos: pd.DataFrame) -> np.ndarray:
        probabilidades = datos["senal"].to_numpy(dtype=float)
        return np.column_stack([1 - probabilidades, probabilidades])


def test_metricas_binarias_aplica_umbral_y_orden_tn_fp_fn_tp():
    # Rompe si se invierte una celda de la matriz, se ignora el umbral o se omite F1.
    resultado = metricas_binarias(
        y_real=[0, 0, 1, 1], probabilidades=[0.1, 0.8, 0.4, 0.9], umbral=0.7
    )

    assert resultado == {
        "auc": 0.75,
        "precision": 0.5,
        "sensibilidad": 0.5,
        "f1": 0.5,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
        "umbral": 0.7,
    }


def test_metricas_binarias_evitan_division_por_cero_y_auc_de_una_clase():
    # Rompe si una prediccion sin positivos produce NaN/error o AUC enganoso.
    resultado = metricas_binarias(y_real=[0, 0], probabilidades=[0.1, 0.2], umbral=0.5)

    assert resultado["precision"] == 0.0
    assert resultado["sensibilidad"] == 0.0
    assert resultado["f1"] == 0.0
    assert np.isnan(resultado["auc"])


def test_curva_roc_devuelve_tabla_normalizada_con_linea_completa():
    # Rompe si se pierden los puntos/umbrales devueltos por sklearn o cambia el contrato tabular.
    resultado = curva_roc(y_real=[0, 1, 0, 1], probabilidades=[0.1, 0.9, 0.4, 0.7])

    assert list(resultado.columns) == ["fpr", "tpr", "umbral"]
    assert resultado.iloc[0][["fpr", "tpr"]].tolist() == [0.0, 0.0]
    assert resultado.iloc[-1][["fpr", "tpr"]].tolist() == [1.0, 1.0]


def test_evaluar_modelo_congelado_mide_prediccion_sin_fit_y_usa_umbral_de_ficha():
    # Rompe si se llama fit, se usa 0.5 en lugar del manifiesto, o se pierde el tiempo registrado.
    datos = pd.DataFrame({"probabilidad": [0.6, 0.8], "repeater": [0, 1]})

    resultado = evaluar_modelo_congelado("07_prueba", ModeloCongeladoDePrueba(), datos)

    assert resultado["modelo"] == "07_prueba"
    assert resultado["umbral"] == 0.7
    assert resultado["tiempo_entrenamiento_s"] == 12.5
    assert resultado["tp"] == 1
    assert resultado["fp"] == 0
    assert resultado["tiempo_prediccion_s"] >= 0
    assert resultado["probabilidades"].tolist() == [0.6, 0.8]


def test_validar_por_ofertas_reentrena_factory_en_grupos_disjuntos_con_splits_inyectados():
    # Rompe si se reutiliza un pipeline, se mezclan ofertas o no se informan AUC y tiempos por split.
    datos = pd.DataFrame(
        {
            "offer": ["a", "a", "b", "b", "c", "c"],
            "repeater": [0, 1, 0, 1, 0, 1],
            "senal": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        }
    )
    fabricados: list[tuple[str, int]] = []

    def factory(algoritmo: str, entrenamiento: pd.DataFrame) -> PipelineTemporal:
        fabricados.append((algoritmo, len(entrenamiento)))
        return PipelineTemporal()

    tabla = validar_por_ofertas(
        datos,
        algoritmos=["lineal", "arbol"],
        factory=factory,
        splits=[(np.array([0, 1, 2, 3]), np.array([4, 5])), (np.array([2, 3, 4, 5]), np.array([0, 1]))],
    )

    assert len(tabla) == 4
    assert fabricados == [("lineal", 4), ("lineal", 4), ("arbol", 4), ("arbol", 4)]
    assert tabla["grupos_disjuntos"].all()
    assert (tabla["auc"] == 1.0).all()
    assert (tabla[["tiempo_entrenamiento_s", "tiempo_prediccion_s"]] >= 0).all().all()


def test_validar_por_ofertas_rechaza_splits_con_la_misma_oferta_en_ambos_lados():
    # Rompe si una fuga por oferta llega a entrenar o a producir una tabla aparentemente valida.
    datos = pd.DataFrame(
        {"offer": ["a", "a", "b", "b"], "repeater": [0, 1, 0, 1], "senal": [0.1, 0.9, 0.2, 0.8]}
    )

    with pytest.raises(AssertionError, match="ofertas"):
        validar_por_ofertas(
            datos,
            algoritmos=["lineal"],
            factory=lambda _algoritmo, _entrenamiento: PipelineTemporal(),
            splits=[(np.array([0, 2]), np.array([1, 3]))],
        )
