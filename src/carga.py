"""Funciones de carga de los CSV pequeños del reto (train/test history, offers)
con los dtypes correctos. transactions.csv.gz se carga aparte en
src/preprocesamiento.py (es demasiado grande para cargarlo entero)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OFFERS_PATH, TEST_HISTORY_PATH, TRAIN_HISTORY_PATH

_HISTORY_DTYPES = {
    "id": "int64",
    "chain": "int32",
    "offer": "int32",
    "market": "int32",
}

_OFFERS_DTYPES = {
    "offer": "int32",
    "category": "int32",
    "quantity": "int32",
    "company": "int64",
    "offervalue": "float64",
    "brand": "int64",
}


def _repeater_a_booleano(valor: object) -> bool:
    """El CSV crudo de Kaggle trae 'repeater' como 't'/'f', no como booleano nativo."""
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in {"t", "true", "1"}


def cargar_train() -> pd.DataFrame:
    """Carga trainHistory.csv con dtypes explícitos y `repeater` como booleano."""
    df = pd.read_csv(
        TRAIN_HISTORY_PATH,
        dtype={**_HISTORY_DTYPES, "repeattrips": "int32"},
        parse_dates=["offerdate"],
    )
    df["repeater"] = df["repeater"].apply(_repeater_a_booleano)
    return df


def cargar_test() -> pd.DataFrame:
    """Carga testHistory.csv con dtypes explícitos (sin repeater/repeattrips)."""
    return pd.read_csv(
        TEST_HISTORY_PATH,
        dtype=_HISTORY_DTYPES,
        parse_dates=["offerdate"],
    )


def cargar_offers() -> pd.DataFrame:
    """Carga offers.csv con dtypes explícitos."""
    return pd.read_csv(OFFERS_PATH, dtype=_OFFERS_DTYPES)
