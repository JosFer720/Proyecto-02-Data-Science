"""Pruebas del preprocesamiento de entradas de la aplicación."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entrada_app import COLUMNAS_SALIDA, EntradaInvalida, leer_transacciones, preparar_cliente
from preprocesamiento import cargar_dataset_modelo
from verificar_entrada_app import RUTA_CLIENTES, _iguales, cargar_clientes_de_prueba, verificar

OFERTA = 1208251  # categoría 2202, compañía 104460040, marca 3718
TRANSACCIONES_BASE = pd.DataFrame(
    {
        "id": [7] * 4,
        "chain": [205, 205, 205, 205],
        "dept": [22, 22, 99, 22],
        "category": [2202, 2202, 9999, 2202],
        "company": [104460040, 104460040, 1, 104460040],
        "brand": [3718, 3718, 1, 3718],
        "date": ["2013-03-01", "2013-04-01", "2013-04-10", "2013-04-20"],
        "productsize": [16, 16, 1, 16],
        "productmeasure": ["OZ", "OZ", "CT", "OZ"],
        "purchasequantity": [1, 2, 1, -1],
        "purchaseamount": [3.5, 7.0, 2.0, -3.5],
    }
)


def _preparar(transacciones, **kwargs):
    parametros = {"oferta": OFERTA, "fecha_oferta": "2013-04-24", "market": 34}
    parametros.update(kwargs)
    return preparar_cliente(transacciones, **parametros)


requiere_ejemplos = pytest.mark.skipif(not RUTA_CLIENTES.exists(), reason="faltan los clientes de ejemplo")


@requiere_ejemplos
def test_variables_iguales_a_dataset_modelo_en_100_clientes():
    clientes, transacciones = cargar_clientes_de_prueba()
    dataset = cargar_dataset_modelo().set_index("id")
    assert len(clientes) == 100
    for cliente in clientes.itertuples(index=False):
        resultado = preparar_cliente(
            transacciones[transacciones["id"] == cliente.id],
            cliente.offer,
            cliente.offerdate,
            cliente.market,
            cadena=cliente.chain,
        )
        app = resultado.fila.set_index("id").iloc[0]
        referencia = dataset.loc[cliente.id, [c for c in COLUMNAS_SALIDA if c != "id"]]
        iguales = _iguales(app[referencia.index], referencia)
        assert iguales.all(), f"cliente {cliente.id}: {list(iguales.index[~iguales])}"


@requiere_ejemplos
def test_probabilidades_iguales_al_notebook_08():
    from modelos_congelados import ModeloCongeladoError, cargar_modelos

    try:
        modelos = cargar_modelos(version=2, estricto=True)
    except ModeloCongeladoError as error:
        pytest.skip(f"el entorno no tiene la versión de scikit-learn de los modelos: {error}")
    tabla = verificar(modelos)
    assert tabla["coincide"].all()


def test_columnas_de_salida_y_reglas_basicas():
    resultado = _preparar(TRANSACCIONES_BASE)
    fila = resultado.fila.iloc[0]
    assert list(resultado.fila.columns) == COLUMNAS_SALIDA
    assert len(resultado.fila) == 1
    assert fila["n_lineas"] == 4
    assert fila["n_compras"] == 3
    assert fila["n_devoluciones"] == 1
    assert fila["n_compras_producto"] == 2
    assert fila["nunca_compro_producto"] == 0
    assert fila["dias_ultima_compra"] == 4
    assert resultado.avisos == []


def test_ignora_otras_cadenas_y_fechas_posteriores():
    extra = pd.DataFrame(
        {
            **{col: TRANSACCIONES_BASE[col].iloc[:2].tolist() for col in TRANSACCIONES_BASE.columns},
            "chain": [205, 999],
            "date": ["2013-04-24", "2013-04-01"],
        }
    )
    resultado = _preparar(pd.concat([TRANSACCIONES_BASE, extra], ignore_index=True))
    assert resultado.lineas_recibidas == 6
    assert resultado.lineas_usadas == 4
    assert resultado.cadena == 205
    assert any("otras cadenas" in a for a in resultado.avisos)
    assert any("fecha de la oferta o después" in a for a in resultado.avisos)


def test_cliente_sin_historial_previo():
    resultado = _preparar(TRANSACCIONES_BASE, fecha_oferta="2013-01-01")
    fila = resultado.fila.iloc[0]
    assert resultado.sin_historial
    assert fila["n_lineas"] == 0
    assert fila["nunca_compro_categoria"] == 1
    assert pd.isna(fila["dias_ultima_compra"])
    assert any("no tiene compras válidas" in a for a in resultado.avisos)


def test_lineas_ilegibles_se_descartan_con_aviso():
    datos = TRANSACCIONES_BASE.copy()
    datos.loc[0, "date"] = "01/03/2013"
    resultado = _preparar(datos)
    assert resultado.lineas_usadas == 3
    assert any("ilegibles" in a for a in resultado.avisos)


@pytest.mark.parametrize(
    ("cambio", "mensaje"),
    [
        (lambda d: d.drop(columns=["purchaseamount", "date"]), "faltan columnas"),
        (lambda d: d.assign(id=[1, 2, 3, 4]), "4 clientes distintos"),
        (lambda d: d.assign(date="sin fecha"), "Ninguna línea"),
        (lambda d: d.iloc[0:0], "No hay transacciones"),
    ],
)
def test_entradas_invalidas_dan_un_mensaje_claro(cambio, mensaje):
    with pytest.raises(EntradaInvalida, match=mensaje):
        _preparar(cambio(TRANSACCIONES_BASE.copy()))


def test_oferta_inexistente():
    with pytest.raises(EntradaInvalida, match="no existe"):
        _preparar(TRANSACCIONES_BASE, oferta=1)


def test_leer_transacciones_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text("")
    with pytest.raises(EntradaInvalida, match="vacío"):
        leer_transacciones(ruta)
