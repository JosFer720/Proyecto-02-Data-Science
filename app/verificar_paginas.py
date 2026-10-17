"""Prueba funcional reproducible: python app/verificar_paginas.py."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from estilo_graficos import algoritmo_de

def comprobar(app):
    assert not app.exception, [e.message for e in app.exception]

def main():
    inicio = AppTest.from_file(str(RAIZ / "app/Inicio.py")).run(timeout=30)
    comprobar(inicio)
    explorar = AppTest.from_file(str(RAIZ / "app/pages/1_Explorar_datos.py")).run(timeout=30)
    comprobar(explorar)
    datos = pd.read_csv(RAIZ / "data/processed/dataset_modelo.csv")
    categoria = int(datos.category.mode().iloc[0])
    explorar.sidebar.multiselect[2].set_value([categoria]).run(timeout=30)
    comprobar(explorar)
    assert explorar.metric[0].value == f"{(datos.category == categoria).sum():,}"
    # Construir una combinación imposible a partir de los datos, sin asumir IDs.
    pares = datos[["offer", "category"]].drop_duplicates()
    oferta = int(pares.offer.iloc[0])
    otra = next(int(x) for x in datos.category.unique() if x not in pares.loc[pares.offer == oferta, "category"].values)
    explorar.sidebar.multiselect[0].set_value([oferta])
    explorar.sidebar.multiselect[2].set_value([otra]).run(timeout=30)
    comprobar(explorar)
    assert any("No hay clientes" in x.value for x in explorar.info)
    rendimiento = AppTest.from_file(str(RAIZ / "app/pages/3_Rendimiento.py")).run(timeout=30)
    comprobar(rendimiento)
    assert len(rendimiento.metric) == 3
    inicial = [x.value for x in rendimiento.metric]
    rendimiento.slider[0].set_value(.7).run(timeout=30)
    comprobar(rendimiento)
    assert inicial != [x.value for x in rendimiento.metric]
    rendimiento.multiselect[0].set_value([]).run(timeout=30)
    comprobar(rendimiento)
    assert any("Selecciona al menos" in x.value for x in rendimiento.info)
    rendimiento.toggle[0].set_value(False).run(timeout=30)
    comprobar(rendimiento)
    assert not rendimiento.metric and not rendimiento.slider and not rendimiento.dataframe
    tablas = RAIZ / "results/tables"
    metricas = pd.read_csv(tablas / "08_metricas_prueba.csv")
    pred = pd.read_parquet(tablas / "08_predicciones_prueba.parquet")
    assert len(pred) == 35499 and pred.id.is_unique
    for r in metricas.itertuples():
        assert np.isclose(roc_auc_score(pred.repeater, pred["probabilidad_" + algoritmo_de(r.modelo)]), r.auc, atol=1e-12)
    clasificar = AppTest.from_file(str(RAIZ / "app/pages/2_Clasificar.py")).run(timeout=60)
    comprobar(clasificar)
    assert not clasificar.error, [x.value for x in clasificar.error]
    print("OK: cuatro páginas, filtros, combinación vacía, umbral, selección vacía, ocultar rendimiento y AUC de los cuatro modelos.")

if __name__ == "__main__":
    main()
