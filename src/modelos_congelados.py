"""Unica forma de guardar y cargar los modelos finales del proyecto.

Un modelo congelado se entrena una sola vez y nadie lo vuelve a ajustar. Cada
archivo ``.joblib`` tiene una ficha en ``results/modelos/manifiesto.json`` con su
huella SHA-256, las versiones con que se entreno y las columnas que espera.
``cargar_modelos`` verifica todo eso y devuelve objetos que solo permiten predecir.

Cargar un archivo de joblib ejecuta codigo, por eso solo se cargan archivos de
``results/modelos/`` cuya huella coincide con la ficha.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

# Los pipelines guardados apuntan a funciones de modelo.py, y este modulo
# tambien debe poder importarse desde la app sin que quien lo use toque sys.path.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import MODELS_DIR, RANDOM_STATE  # noqa: E402
import modelo as _modelo  # noqa: E402,F401  (necesario para deserializar los pipelines)

RUTA_MANIFIESTO = MODELS_DIR / "manifiesto.json"

PREPARACION_PREVIA = {"crear_variables_avanzadas": _modelo.crear_variables_avanzadas}


class ModeloCongeladoError(RuntimeError):
    """Error al guardar, verificar, cargar o usar un modelo congelado."""


def huella_sha256(ruta: Path) -> str:
    """Huella SHA-256 del contenido de un archivo."""
    resumen = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def versiones_actuales() -> dict:
    """Versiones del entorno que importan para que un modelo guardado sea confiable."""
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def leer_manifiesto(ruta: Path = RUTA_MANIFIESTO) -> dict:
    if not ruta.exists():
        return {"esquema": 1, "modelos": {}}
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def _a_json(valor):
    """Convierte tipos de numpy y tuplas que json no sabe escribir."""
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, tuple):
        return list(valor)
    raise TypeError(f"No se puede escribir {type(valor).__name__} en el manifiesto")


def _escribir_manifiesto(manifiesto: dict, ruta: Path = RUTA_MANIFIESTO) -> None:
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(manifiesto, archivo, ensure_ascii=False, indent=2, default=_a_json)
        archivo.write("\n")


class ModeloCongelado:
    """Pipeline guardado que solo permite predecir.

    Revisa que lleguen las columnas esperadas y bloquea cualquier intento de
    reentrenarlo o cambiarlo.
    """

    __slots__ = ("_pipeline", "nombre", "ficha")

    def __init__(self, pipeline, nombre: str, ficha: dict):
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "nombre", nombre)
        object.__setattr__(self, "ficha", ficha)

    def __setattr__(self, nombre, valor):
        raise ModeloCongeladoError(f"El modelo {self.nombre} esta congelado y no se puede modificar")

    @property
    def algoritmo(self) -> str:
        return self.ficha["algoritmo"]

    @property
    def columnas_entrada(self) -> list[str]:
        return list(self.ficha["columnas_entrada"])

    @property
    def umbral(self) -> float:
        return float(self.ficha["umbral"])

    def _preparar(self, datos: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(datos, pd.DataFrame):
            raise ModeloCongeladoError("La entrada debe ser un DataFrame con las columnas de dataset_modelo.csv")
        faltantes = [c for c in self.columnas_entrada if c not in datos.columns]
        if faltantes:
            raise ModeloCongeladoError(
                f"Al modelo {self.nombre} le faltan columnas de entrada: {', '.join(faltantes)}"
            )
        entrada = datos[self.columnas_entrada].copy()
        previa = self.ficha.get("preparacion_previa")
        if previa:
            entrada = PREPARACION_PREVIA[previa](entrada)
        return entrada

    def predecir_proba(self, datos: pd.DataFrame) -> np.ndarray:
        """Probabilidad de recompra de cada fila."""
        return self._pipeline.predict_proba(self._preparar(datos))[:, 1]

    def clasificar(self, datos: pd.DataFrame, umbral: float | None = None) -> np.ndarray:
        """1 si la probabilidad alcanza el umbral de la ficha o el indicado, 0 si no."""
        corte = self.umbral if umbral is None else umbral
        return (self.predecir_proba(datos) >= corte).astype(int)

    def _bloqueado(self, *args, **kwargs):
        raise ModeloCongeladoError(f"El modelo {self.nombre} esta congelado: solo permite predecir")

    fit = partial_fit = set_params = _bloqueado

    def __repr__(self) -> str:
        return f"ModeloCongelado({self.nombre!r}, version {self.ficha.get('version', 1)})"


def guardar_modelo_congelado(
    pipeline,
    nombre: str,
    *,
    algoritmo: str,
    columnas_entrada: list[str],
    hiperparametros: dict,
    auc_validacion: float,
    umbral: float,
    periodo_entrenamiento: str,
    periodo_validacion: str,
    tiempo_entrenamiento_s: float | None = None,
    semilla: int = RANDOM_STATE,
    preparacion_previa: str | None = None,
    nota: str | None = None,
    version: int = 1,
    directorio: Path = MODELS_DIR,
) -> dict:
    """Guarda un pipeline ya entrenado y registra su ficha en el manifiesto.

    Nunca sobrescribe: si el archivo existe hay que usar otro nombre, por
    ejemplo ``07_random_forest_v2``.
    """
    ruta = directorio / f"{nombre}.joblib"
    if ruta.exists():
        raise ModeloCongeladoError(
            f"{ruta.name} ya existe y nadie lo sobrescribe: guarda una version nueva con otro nombre"
        )
    directorio.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta, compress=3)
    return _registrar(
        ruta,
        nombre,
        algoritmo=algoritmo,
        columnas_entrada=columnas_entrada,
        hiperparametros=hiperparametros,
        auc_validacion=auc_validacion,
        umbral=umbral,
        periodo_entrenamiento=periodo_entrenamiento,
        periodo_validacion=periodo_validacion,
        tiempo_entrenamiento_s=tiempo_entrenamiento_s,
        semilla=semilla,
        preparacion_previa=preparacion_previa,
        nota=nota,
        version=version,
    )


def _registrar(ruta: Path, nombre: str, **datos) -> dict:
    ficha = {
        "archivo": ruta.name,
        "sha256": huella_sha256(ruta),
        "tamano_mb": round(ruta.stat().st_size / 1e6, 2),
        "fecha_guardado": datetime.now().isoformat(timespec="seconds"),
        "versiones": versiones_actuales(),
        **datos,
    }
    manifiesto = leer_manifiesto(ruta.parent / "manifiesto.json")
    if nombre in manifiesto["modelos"]:
        raise ModeloCongeladoError(f"{nombre} ya esta en el manifiesto")
    manifiesto["modelos"][nombre] = ficha
    _escribir_manifiesto(manifiesto, ruta.parent / "manifiesto.json")
    return ficha


def cargar_modelos(
    nombres: list[str] | None = None,
    prefijo: str | None = None,
    version: int | None = None,
    directorio: Path = MODELS_DIR,
    estricto: bool = True,
) -> dict[str, ModeloCongelado]:
    """Carga los modelos del manifiesto despues de verificarlos.

    ``version`` filtra por la version de la ficha: 1 son los entrenados hasta el
    16 de abril y 2 los reentrenados con la semana de validacion. Una ficha sin
    el campo cuenta como version 1.

    Se detiene con un mensaje claro si la huella de un archivo no coincide con
    su ficha o si scikit-learn no es la version con que se entreno. Con
    ``estricto=False`` esa diferencia de version solo avisa.
    """
    manifiesto = leer_manifiesto(directorio / "manifiesto.json")
    fichas = manifiesto["modelos"]
    if not fichas:
        raise ModeloCongeladoError("results/modelos/manifiesto.json no tiene modelos registrados")

    if nombres is not None:
        desconocidos = [n for n in nombres if n not in fichas]
        if desconocidos:
            raise ModeloCongeladoError(
                f"No estan en el manifiesto: {', '.join(desconocidos)}. Disponibles: {', '.join(fichas)}"
            )
        elegidos = nombres
    else:
        elegidos = [
            n
            for n in fichas
            if (prefijo is None or n.startswith(prefijo))
            and (version is None or fichas[n].get("version", 1) == version)
        ]
        if not elegidos:
            raise ModeloCongeladoError("Ningun modelo del manifiesto cumple el prefijo y la version pedidos")

    cargados = {}
    for nombre in elegidos:
        ficha = fichas[nombre]
        ruta = directorio / ficha["archivo"]
        if not ruta.exists():
            raise ModeloCongeladoError(f"Falta el archivo {ruta.name} que el manifiesto registra para {nombre}")
        if huella_sha256(ruta) != ficha["sha256"]:
            raise ModeloCongeladoError(
                f"{ruta.name} no coincide con su huella SHA-256 del manifiesto: "
                "el archivo cambio o se corrompio. No se carga; pide a Kevin el archivo correcto."
            )
        _verificar_versiones(nombre, ficha, estricto)
        cargados[nombre] = ModeloCongelado(joblib.load(ruta), nombre, ficha)
    return cargados


def _verificar_versiones(nombre: str, ficha: dict, estricto: bool) -> None:
    guardadas, actuales = ficha["versiones"], versiones_actuales()
    if guardadas["scikit-learn"] != actuales["scikit-learn"]:
        mensaje = (
            f"{nombre} se entreno con scikit-learn {guardadas['scikit-learn']} y aqui hay "
            f"{actuales['scikit-learn']}. Instala la version del requirements.txt."
        )
        if estricto:
            raise ModeloCongeladoError(mensaje)
        warnings.warn(mensaje)
    for paquete in ("numpy", "pandas"):
        if guardadas[paquete] != actuales[paquete]:
            warnings.warn(
                f"{nombre}: {paquete} {actuales[paquete]} difiere de {guardadas[paquete]} usado al entrenar"
            )


def registrar_modelos_anteriores() -> None:
    """Registra los dos modelos logisticos de los notebooks 05 y 06 para trabajar desde ya.

    Se entrenaron hasta el 23 de abril y su AUC se midio en la semana del 24 al
    30 de abril, la misma semana de prueba que usan los modelos 07.
    """
    comun = {
        "algoritmo": "regresion_logistica",
        "semilla": RANDOM_STATE,
        "periodo_entrenamiento": "2013-03-01 a 2013-04-23 (primer 80% por fecha)",
        "periodo_validacion": "2013-04-24 a 2013-04-30 (semana de prueba actual)",
        "tiempo_entrenamiento_s": None,
        "nota": (
            "Entrenado del 1 de marzo al 23 de abril. Su AUC se midio en la semana del 24 al 30 de abril, "
            "la misma semana de prueba de los modelos 07, que son los que se entrenaron solo hasta el 16 de abril"
        ),
    }
    registros = [
        (
            "05_regresion_logistica_linea_base",
            list(_modelo.COLUMNAS_LINEA_BASE),
            {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "liblinear"},
            0.561,
            None,
        ),
        (
            "06_regresion_logistica_mejorada",
            None,  # se calcula del dataset
            {"C": 0.25, "penalty": "l2", "class_weight": "balanced", "solver": "liblinear"},
            0.645,
            "crear_variables_avanzadas",
        ),
    ]
    from preprocesamiento import cargar_dataset_modelo

    datos = cargar_dataset_modelo()
    for nombre, columnas, hiperparametros, auc, previa in registros:
        if nombre in leer_manifiesto()["modelos"]:
            continue
        if columnas is None:
            columnas = _modelo.columnas_entrada_modelo(datos)
        _registrar(
            MODELS_DIR / f"{nombre}.joblib",
            nombre,
            columnas_entrada=columnas,
            hiperparametros=hiperparametros,
            auc_validacion=auc,
            umbral=0.5,
            preparacion_previa=previa,
            **comun,
        )


if __name__ == "__main__":
    modelos = cargar_modelos()
    for modelo in modelos.values():
        print(modelo, "OK")
