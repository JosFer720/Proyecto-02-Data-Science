"""Obtiene los CSV crudos del reto Kaggle "Acquire Valued Shoppers Challenge" en data/raw/.

Orden de intento:
1. Si los archivos ya existen en data/raw/, no hace nada (idempotente).
2. Si hay un zip local del reto en la raíz del repo, extrae de ahí.
3. Si no, intenta descargar con la API de Kaggle (requiere el paquete `kaggle`
   instalado y credenciales en ~/.kaggle/kaggle.json).
4. Si nada de eso funciona, imprime instrucciones para descarga manual.

transactions.csv.gz se deja comprimido en data/raw/ (no se descomprime a CSV
plano): el archivo original son varias decenas de GB descomprimido y pandas
puede leerlo directo con `compression="gzip"`, en chunks, sin necesidad de
materializarlo en disco.
"""

import gzip
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    KAGGLE_COMPETITION,
    OFFERS_PATH,
    RAW_DATA_DIR,
    RAW_ZIP_PATH,
    SAMPLE_SUBMISSION_PATH,
    TEST_HISTORY_PATH,
    TRAIN_HISTORY_PATH,
    TRANSACTIONS_PATH,
)

# nombre de entrada dentro del zip -> ruta destino en data/raw/
_CSV_ENTRIES = {
    "trainHistory.csv.gz": TRAIN_HISTORY_PATH,
    "testHistory.csv.gz": TEST_HISTORY_PATH,
    "offers.csv.gz": OFFERS_PATH,
    "sampleSubmission.csv.gz": SAMPLE_SUBMISSION_PATH,
}
_TRANSACTIONS_ENTRY = "transactions.csv.gz"

_MANUAL_INSTRUCTIONS = f"""
No se encontró el zip del reto ni credenciales de Kaggle configuradas.

Para descargar manualmente:
1. Entra a https://www.kaggle.com/c/{KAGGLE_COMPETITION}/data (requiere unirte al reto).
2. Descarga el archivo del reto y colócalo en la raíz del repo como
   "acquire-valued-shoppers-challenge.zip".
3. Vuelve a correr: python src/download_dataset.py

Alternativa con la API de Kaggle:
1. pip install kaggle
2. Genera un token en https://www.kaggle.com/settings -> "Create New API Token"
   y guarda el kaggle.json descargado en ~/.kaggle/kaggle.json
   (en Windows: C:\\Users\\<usuario>\\.kaggle\\kaggle.json)
3. kaggle competitions download -c {KAGGLE_COMPETITION} -p .
""".strip()


def _ya_existen_los_csv() -> bool:
    return all(p.exists() for p in _CSV_ENTRIES.values())


def _extraer_desde_zip() -> None:
    print(f"Extrayendo desde {RAW_ZIP_PATH.name}...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP_PATH) as z:
        for entry_name, dest_path in _CSV_ENTRIES.items():
            if dest_path.exists():
                continue
            print(f"  {entry_name} -> {dest_path.relative_to(dest_path.parents[2])}")
            with z.open(entry_name) as gz_stream, gzip.GzipFile(fileobj=gz_stream) as f_in:
                with open(dest_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        if not TRANSACTIONS_PATH.exists():
            print(f"  {_TRANSACTIONS_ENTRY} -> {TRANSACTIONS_PATH.relative_to(TRANSACTIONS_PATH.parents[2])} (se deja comprimido)")
            with z.open(_TRANSACTIONS_ENTRY) as gz_stream:
                with open(TRANSACTIONS_PATH, "wb") as f_out:
                    shutil.copyfileobj(gz_stream, f_out)
    print("Extracción completa.")


def _intentar_descarga_kaggle() -> bool:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        return False

    import subprocess

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(RAW_DATA_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False

    downloaded_zip = RAW_DATA_DIR / f"{KAGGLE_COMPETITION}.zip"
    if downloaded_zip.exists():
        with zipfile.ZipFile(downloaded_zip) as z:
            z.extractall(RAW_DATA_DIR)
        downloaded_zip.unlink()
    return True


def main() -> None:
    if _ya_existen_los_csv() and TRANSACTIONS_PATH.exists():
        print("Los datos ya están en data/raw/, no se hace nada.")
        return

    if RAW_ZIP_PATH.exists():
        _extraer_desde_zip()
        return

    if _intentar_descarga_kaggle():
        print("Descarga vía Kaggle API completa.")
        return

    print(_MANUAL_INSTRUCTIONS)
    sys.exit(1)


if __name__ == "__main__":
    main()
