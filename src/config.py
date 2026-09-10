"""Rutas y constantes compartidas por todo el proyecto."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "modelos"
TABLES_DIR = RESULTS_DIR / "tables"

RAW_ZIP_PATH = ROOT_DIR / "acquire-valued-shoppers-challenge.zip"

TRAIN_HISTORY_PATH = RAW_DATA_DIR / "trainHistory.csv"
TEST_HISTORY_PATH = RAW_DATA_DIR / "testHistory.csv"
OFFERS_PATH = RAW_DATA_DIR / "offers.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sampleSubmission.csv"
TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions.csv.gz"

TRAIN_OFFERS_PATH = PROCESSED_DATA_DIR / "train_offers.csv"
DATASET_MODELO_PATH = PROCESSED_DATA_DIR / "dataset_modelo.csv"

RANDOM_STATE = 42

# Fecha de corte oficial del reto: train son ofertas antes de esta fecha,
# test son ofertas en o después de esta fecha (split ya viene hecho por Kaggle).
OFFER_DATE_CUTOFF = "2013-05-01"

KAGGLE_COMPETITION = "acquire-valued-shoppers-challenge"
