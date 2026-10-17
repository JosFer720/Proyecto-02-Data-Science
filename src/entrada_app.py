"""Preprocesamiento de los datos de un cliente nuevo para la aplicación.

Recibe las transacciones crudas de un cliente y la oferta que se le quiere hacer,
y devuelve una fila con las mismas columnas y tipos que `dataset_modelo.csv`, lista
para los modelos congelados. Para que la transformación sea idéntica a la del
entrenamiento no se reescribe ningún cálculo: se usan las mismas funciones de
`preprocesamiento.py`.

Las mismas reglas del entrenamiento se aplican aquí:
- solo cuentan las compras en la cadena del cliente;
- solo cuentan las compras anteriores a la fecha de la oferta;
- un cliente sin compras previas de la categoría, compañía, marca o producto
  ofertado tiene ceros en esos conteos y la bandera nunca_compro_* en 1.

Cualquier problema con los datos se informa con `EntradaInvalida`, cuyo mensaje
está escrito para el usuario de la aplicación, o con avisos que no detienen el
cálculo.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carga import cargar_offers
from preprocesamiento import (
    COLUMNAS_CLIENTE,
    COLUMNAS_GENERALES,
    COLUMNAS_OBJETIVO,
    COLUMNAS_OFERTA,
    _agregar_por_cliente,
    construir_dataset,
)

COLUMNAS_TRANSACCIONES = {
    "chain": "cadena",
    "category": "categoría",
    "company": "compañía",
    "brand": "marca",
    "date": "fecha",
    "purchasequantity": "cantidad",
    "purchaseamount": "monto",
}
COLUMNAS_SALIDA = COLUMNAS_CLIENTE + COLUMNAS_GENERALES + COLUMNAS_OFERTA

FORMATO_FECHA = "%Y-%m-%d"


class EntradaInvalida(ValueError):
    """Datos que no permiten clasificar al cliente. El mensaje es para el usuario."""


@dataclass
class ResultadoEntrada:
    """Fila lista para los modelos y lo que pasó al prepararla."""

    fila: pd.DataFrame
    avisos: list[str] = field(default_factory=list)
    lineas_recibidas: int = 0
    lineas_usadas: int = 0
    cadena: int | None = None

    @property
    def sin_historial(self) -> bool:
        return self.lineas_usadas == 0


@lru_cache(maxsize=1)
def ofertas_disponibles() -> pd.DataFrame:
    """Catálogo de ofertas de la competencia, con una etiqueta legible."""
    ofertas = cargar_offers().sort_values("offer").reset_index(drop=True)
    ofertas["etiqueta"] = [
        f"Oferta {oferta}: categoría {categoria}, marca {marca}, descuento de ${valor:.2f}"
        for oferta, categoria, marca, valor in zip(
            ofertas["offer"], ofertas["category"], ofertas["brand"], ofertas["offervalue"]
        )
    ]
    return ofertas


def leer_transacciones(archivo) -> pd.DataFrame:
    """Lee un CSV de transacciones desde una ruta o un archivo subido en la app."""
    try:
        datos = pd.read_csv(archivo)
    except UnicodeDecodeError as error:
        raise EntradaInvalida("El archivo no es un CSV de texto. Revisa que lo hayas exportado como CSV.") from error
    except pd.errors.EmptyDataError as error:
        raise EntradaInvalida("El archivo está vacío.") from error
    except pd.errors.ParserError as error:
        raise EntradaInvalida("No se pudo leer el archivo como CSV. Revisa que use comas como separador.") from error
    if datos.empty:
        raise EntradaInvalida("El archivo no trae ninguna transacción.")
    return datos


def _validar_columnas(datos: pd.DataFrame) -> None:
    faltantes = [col for col in COLUMNAS_TRANSACCIONES if col not in datos.columns]
    if faltantes:
        detalle = ", ".join(f"{col} ({COLUMNAS_TRANSACCIONES[col]})" for col in faltantes)
        raise EntradaInvalida(
            f"Al archivo le faltan columnas: {detalle}. Usa el formato de transactions.csv de la competencia."
        )


def _identificar_cliente(datos: pd.DataFrame, id_cliente: int | None) -> int:
    if "id" not in datos.columns:
        return int(id_cliente) if id_cliente is not None else 0
    ids = pd.to_numeric(datos["id"], errors="coerce").dropna().unique()
    if len(ids) > 1:
        raise EntradaInvalida(
            f"El archivo trae transacciones de {len(ids)} clientes distintos. Sube las compras de un solo cliente."
        )
    if len(ids) == 1:
        return int(ids[0])
    return int(id_cliente) if id_cliente is not None else 0


def _limpiar_tipos(datos: pd.DataFrame, avisos: list[str]) -> pd.DataFrame:
    """Convierte las columnas a números y fechas, y descarta las líneas ilegibles."""
    tx = datos[list(COLUMNAS_TRANSACCIONES)].copy()
    for col in ("chain", "category", "company", "brand", "purchasequantity", "purchaseamount"):
        tx[col] = pd.to_numeric(tx[col], errors="coerce")
    tx["date"] = pd.to_datetime(tx["date"].astype(str), format=FORMATO_FECHA, errors="coerce")

    ilegibles = tx.isna().any(axis=1)
    if ilegibles.all():
        raise EntradaInvalida(
            "Ninguna línea del archivo tiene números y fechas válidos. Las fechas deben tener el formato AAAA-MM-DD."
        )
    if ilegibles.any():
        avisos.append(
            f"Se descartaron {int(ilegibles.sum()):,} líneas con datos ilegibles, como fechas o montos vacíos."
        )
        tx = tx[~ilegibles]
    return tx.astype(
        {
            "chain": "int64",
            "category": "int64",
            "company": "int64",
            "brand": "int64",
            "purchasequantity": "int64",
            "purchaseamount": "float64",
        }
    )


def _elegir_cadena(tx: pd.DataFrame, cadena: int | None, avisos: list[str]) -> int:
    if cadena is not None:
        return int(cadena)
    conteo = tx["chain"].value_counts()
    elegida = int(conteo.index[0])
    if len(conteo) > 1:
        avisos.append(
            f"Las compras vienen de {len(conteo)} cadenas. Se usó la cadena {elegida}, "
            "la que tiene más compras, porque los modelos se entrenaron con una cadena por cliente."
        )
    return elegida


def preparar_cliente(
    transacciones: pd.DataFrame,
    oferta: int,
    fecha_oferta,
    market: int,
    cadena: int | None = None,
    id_cliente: int | None = None,
) -> ResultadoEntrada:
    """Convierte las transacciones crudas de un cliente en la fila que esperan los modelos.

    `transacciones` usa las columnas de transactions.csv. `cadena` puede omitirse:
    se toma la cadena con más compras. Devuelve la fila con las columnas de
    dataset_modelo.csv, sin repeattrips ni repeater.
    """
    avisos: list[str] = []
    if not isinstance(transacciones, pd.DataFrame) or transacciones.empty:
        raise EntradaInvalida("No hay transacciones para este cliente.")
    _validar_columnas(transacciones)
    id_cliente = _identificar_cliente(transacciones, id_cliente)

    ofertas = ofertas_disponibles().set_index("offer")
    if int(oferta) not in ofertas.index:
        raise EntradaInvalida(f"La oferta {oferta} no existe en el catálogo de la competencia.")
    detalle = ofertas.loc[int(oferta)]

    fecha_oferta = pd.Timestamp(fecha_oferta).normalize()
    tx = _limpiar_tipos(transacciones, avisos)
    lineas_recibidas = len(transacciones)
    cadena = _elegir_cadena(tx, cadena, avisos)

    otra_cadena = tx["chain"] != cadena
    if otra_cadena.any():
        avisos.append(f"Se ignoraron {int(otra_cadena.sum()):,} líneas de otras cadenas.")
    posteriores = (tx["date"] >= fecha_oferta) & ~otra_cadena
    if posteriores.any():
        avisos.append(
            f"Se ignoraron {int(posteriores.sum()):,} líneas en la fecha de la oferta o después, "
            "porque el modelo solo puede usar el historial previo."
        )
    tx = tx[~otra_cadena & ~posteriores].reset_index(drop=True)
    tx.insert(0, "id", id_cliente)

    if tx.empty:
        avisos.append(
            "El cliente no tiene compras válidas antes de la oferta. "
            "La clasificación usa solo los datos de la oferta y su confiabilidad es menor."
        )

    info = pd.DataFrame(
        {
            "offerdate": fecha_oferta,
            "category": int(detalle["category"]),
            "company": int(detalle["company"]),
            "brand": int(detalle["brand"]),
        },
        index=tx.index,
    )
    variables = _agregar_por_cliente(tx, info)

    cliente = pd.DataFrame(
        {
            "chain": [cadena],
            "market": [int(market)],
            "offer": [int(oferta)],
            "offerdate": [fecha_oferta],
            "category": [int(detalle["category"])],
            "company": [int(detalle["company"])],
            "brand": [int(detalle["brand"])],
            "offervalue": [float(detalle["offervalue"])],
            "quantity": [int(detalle["quantity"])],
            "split": ["nuevo"],
            **{col: [pd.NA] for col in COLUMNAS_OBJETIVO},
        },
        index=pd.Index([id_cliente], name="id"),
    )
    fila = construir_dataset(cliente, variables)[COLUMNAS_SALIDA]

    # Ida y vuelta por CSV: deja los mismos tipos que se obtienen al leer dataset_modelo.csv
    fila = pd.read_csv(io.StringIO(fila.to_csv(index=False)), parse_dates=["offerdate"])
    return ResultadoEntrada(
        fila=fila,
        avisos=avisos,
        lineas_recibidas=lineas_recibidas,
        lineas_usadas=len(tx),
        cadena=cadena,
    )
