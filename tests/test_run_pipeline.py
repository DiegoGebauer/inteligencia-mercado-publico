"""Contratos del coordinador: API, inferencia y BigQuery se sustituyen por dobles."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import run_pipeline as pipeline


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    historial = tmp_path / "data/processed/history/historial_prueba"
    construir = Mock(return_value=historial)
    capturar = Mock(side_effect=AssertionError("Consulta API no esperada."))
    predecir = Mock(side_effect=AssertionError("Inferencia no esperada."))
    preparar = Mock(return_value=object())
    publicar = Mock(side_effect=AssertionError("Publicacion cloud no esperada."))
    for nombre, doble in [
        ("build_history", construir), ("capturar_lote", capturar),
        ("predecir_lote", predecir), ("prepare_history", preparar),
        ("apply_history", publicar),
    ]:
        monkeypatch.setattr(pipeline, nombre, doble)

    def crear_lote(estado="completo"):
        carpeta = tmp_path / "data/raw/batches/lote_prueba"
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "captura.json").write_text(
            json.dumps({"lote_id": carpeta.name, "estado": estado}),
            encoding="utf-8",
        )
        return carpeta

    def guardar_prediccion(carpeta, *, raiz=tmp_path):
        destino = raiz / "data/processed/batches" / carpeta.name / "intento_1"
        destino.mkdir(parents=True, exist_ok=True)
        csv = destino / "predicciones.csv"
        csv.write_text("CodigoExterno\n101-1-LE26\n", encoding="utf-8")
        (destino / "registro_ejecucion.json").write_text(
            json.dumps({"estado": "completo"}), encoding="utf-8",
        )
        return csv

    return SimpleNamespace(
        raiz=tmp_path, historial=historial, construir=construir,
        capturar=capturar, predecir=predecir, preparar=preparar,
        publicar=publicar, crear_lote=crear_lote,
        guardar_prediccion=guardar_prediccion,
    )


def autorizar_publicacion_simulada(entorno):
    entorno.publicar.side_effect = None
    entorno.publicar.return_value = {"job_id": "trabajo_simulado", "tablas": []}


def test_plan_de_fecha_no_ejecuta_etapas_aunque_se_solicite_publicacion(entorno):
    assert pipeline.main([
        "--fecha", "07112026", "--publicar-bigquery",
    ]) == 0
    entorno.construir.assert_not_called()
    entorno.capturar.assert_not_called()
    entorno.predecir.assert_not_called()
    entorno.preparar.assert_not_called()
    entorno.publicar.assert_not_called()


def test_lote_con_prediccion_se_reutiliza_sin_publicar_por_defecto(entorno):
    carpeta = entorno.crear_lote()
    csv = entorno.guardar_prediccion(carpeta)
    contenido = csv.read_bytes()
    assert pipeline.main(["--lote", carpeta.name]) == 0
    entorno.capturar.assert_not_called()
    entorno.predecir.assert_not_called()
    entorno.preparar.assert_not_called()
    entorno.publicar.assert_not_called()
    assert csv.read_bytes() == contenido


def test_lote_reutilizado_tambien_llega_a_bigquery(entorno):
    carpeta = entorno.crear_lote()
    entorno.guardar_prediccion(carpeta)
    autorizar_publicacion_simulada(entorno)
    assert pipeline.main([
        "--lote", carpeta.name, "--publicar-bigquery",
        "--project", "proyecto-prueba", "--dataset", "dataset_prueba",
        "--location", "us-central1", "--max-bytes", "123456",
    ]) == 0
    entorno.predecir.assert_not_called()
    entorno.capturar.assert_not_called()
    entorno.preparar.assert_called_once_with(entorno.historial, root=entorno.raiz)
    entorno.publicar.assert_called_once_with(
        entorno.preparar.return_value, "proyecto-prueba", "dataset_prueba",
        location="us-central1", max_bytes=123456,
    )


def test_publicacion_de_lote_pendiente_ocurre_despues_de_predecir(entorno):
    carpeta = entorno.crear_lote()
    eventos = []

    def predecir(carpeta, *, raiz):
        eventos.append("prediccion")
        return entorno.guardar_prediccion(carpeta, raiz=raiz)

    def preparar(historial, *, root):
        eventos.append("preparacion_bigquery")
        assert (root / "data/processed/batches/lote_prueba/intento_1/predicciones.csv").is_file()
        return object()

    entorno.predecir.side_effect = predecir
    entorno.preparar.side_effect = preparar
    autorizar_publicacion_simulada(entorno)
    assert pipeline.main(["--lote", carpeta.name, "--publicar-bigquery"]) == 0
    assert eventos == ["prediccion", "preparacion_bigquery"]
    entorno.publicar.assert_called_once()
    entorno.capturar.assert_not_called()


def test_fallo_de_publicacion_permite_continuar_sin_repetir_inferencia(entorno):
    carpeta = entorno.crear_lote()
    entorno.predecir.side_effect = entorno.guardar_prediccion
    entorno.publicar.side_effect = RuntimeError("Fallo simulado de carga.")
    argumentos = ["--lote", carpeta.name, "--publicar-bigquery"]
    with pytest.raises(RuntimeError, match="Fallo simulado"):
        pipeline.main(argumentos)

    csv = entorno.raiz / "data/processed/batches/lote_prueba/intento_1/predicciones.csv"
    contenido = csv.read_bytes()
    assert entorno.predecir.call_count == 1
    assert entorno.publicar.call_count == 1

    autorizar_publicacion_simulada(entorno)
    assert pipeline.main(argumentos) == 0
    assert entorno.predecir.call_count == 1
    assert entorno.publicar.call_count == 2
    entorno.capturar.assert_not_called()
    assert csv.read_bytes() == contenido


def test_lote_vacio_puede_publicar_su_historial_sin_cargar_modelo(entorno):
    carpeta = entorno.crear_lote("vacio")
    autorizar_publicacion_simulada(entorno)
    assert pipeline.main(["--lote", carpeta.name, "--publicar-bigquery"]) == 0
    entorno.predecir.assert_not_called()
    entorno.capturar.assert_not_called()
    entorno.publicar.assert_called_once()


def test_captura_simulada_continua_hasta_publicacion(entorno):
    carpeta = entorno.crear_lote()
    entorno.capturar.side_effect = None
    entorno.capturar.return_value = carpeta
    entorno.predecir.side_effect = entorno.guardar_prediccion
    autorizar_publicacion_simulada(entorno)
    assert pipeline.main([
        "--fecha", "07112026", "--limite", "5", "--capturar",
        "--publicar-bigquery",
    ]) == 0
    entorno.capturar.assert_called_once_with("07112026", 5, raiz=entorno.raiz)
    entorno.predecir.assert_called_once_with(carpeta, raiz=entorno.raiz)
    entorno.publicar.assert_called_once()


def test_lote_incompleto_detiene_todas_las_etapas(entorno):
    carpeta = entorno.crear_lote("incompleto")
    with pytest.raises(ValueError, match="incompleto"):
        pipeline.main(["--lote", carpeta.name, "--publicar-bigquery"])
    entorno.construir.assert_not_called()
    entorno.capturar.assert_not_called()
    entorno.predecir.assert_not_called()
    entorno.publicar.assert_not_called()


def test_capturar_sin_fecha_se_rechaza_antes_de_ejecutar(entorno):
    with pytest.raises(SystemExit) as error:
        pipeline.main(["--lote", "lote_prueba", "--capturar"])
    assert error.value.code == 2
    entorno.construir.assert_not_called()
    entorno.capturar.assert_not_called()


def test_historial_invalido_detiene_la_captura_antes_de_consultar_api(entorno):
    entorno.construir.side_effect = ValueError("Historial alterado.")
    with pytest.raises(ValueError, match="Historial alterado"):
        pipeline.main(["--fecha", "07112026", "--capturar", "--publicar-bigquery"])
    entorno.capturar.assert_not_called()
    entorno.predecir.assert_not_called()
    entorno.publicar.assert_not_called()
