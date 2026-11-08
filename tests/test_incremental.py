"""Contratos del backfill con API, inferencia y BigQuery simulados."""
from datetime import date
import json
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from src import run_incremental as inc
from src.ingestion import recent_batch as batch


def listing(codes, lot_id="origen"):
    raw = json.dumps({"Listado": [{"CodigoExterno": code} for code in codes]}).encode("utf-8")
    return inc.Listing(lot_id, raw, "2001-01-02T12:00:00+00:00", tuple(sorted(codes)))


def detail(code):
    return {"Listado": [{"CodigoExterno": code, "Nombre": "Servicio",
                          "Items": {"Listado": []}, "Fechas": {}, "Comprador": {}}]}


@pytest.fixture
def api(monkeypatch):
    client = ModuleType("src.api.mercado_publico_client")
    client.get_licitaciones_por_fecha = Mock(return_value={"Listado": [{"CodigoExterno": "1-1-LE26"}]})
    client.get_licitacion_por_codigo = Mock(side_effect=detail)
    monkeypatch.setitem(sys.modules, "src.api.mercado_publico_client", client)
    now = [0.0]
    original = batch._RequestControl
    def wait(seconds):
        now[0] += seconds
    monkeypatch.setattr(batch, "_RequestControl", lambda: original(clock=lambda: now[0], wait=wait))
    return client


@pytest.fixture
def args():
    return SimpleNamespace(limite=10, publicar_bigquery=False, reintentar_publicacion=False,
                           project="proyecto", dataset="mercado_publico", location="us-central1", max_bytes=1024)


def test_plan_avanza_desde_codigos_predichos_y_no_desde_cero():
    codes = [f"{i}-1-LE26" for i in range(1, 7)]
    snap = listing(codes)
    rows, work = inc.select_work([date(2001, 1, 1)], {"2001-01-01": snap},
                                 {"2001-01-01": set(codes[:2])}, [], 2)
    assert rows[0]["pendientes"] == 4
    assert work["desplazamiento"] == 2
    assert work["codigos"] == codes[2:4]


def test_hueco_no_recaptura_un_codigo_completo_en_el_mismo_tramo():
    snap = listing(["1-1-LE26", "2-1-LE26", "3-1-LE26", "4-1-LE26"])
    _, work = inc.select_work([date(2001, 1, 1)], {"2001-01-01": snap},
                              {"2001-01-01": {"1-1-LE26", "3-1-LE26"}}, [], 10)
    assert work["codigos"] == ["2-1-LE26"]
    assert work["limite"] == 1


def test_dia_sin_listado_se_planifica_sin_inventar_cobertura():
    rows, work = inc.select_work([date(2001, 1, 1)], {}, {}, [], 10)
    assert rows == [{"fecha": "2001-01-01", "listado": "pendiente_de_consultar"}]
    assert work["codigos"] is None


def test_listado_agotado_avanza_a_siguiente_fecha():
    snap = listing(["1-1-LE26"])
    _, work = inc.select_work([date(2001, 1, 1), date(2001, 1, 2)], {"2001-01-01": snap},
                              {"2001-01-01": {"1-1-LE26"}}, [], 10)
    assert work["fecha"] == "2001-01-02"


def test_lote_capturado_sin_prediccion_tiene_prioridad():
    _, work = inc.select_work([date(2001, 1, 1)], {}, {}, [("2001-01-01", "pendiente")], 10)
    assert work == {"tipo": "predecir_existente", "fecha": "2001-01-01", "lote": "pendiente"}


def test_listado_vacio_no_queda_eternamente_pendiente():
    rows, work = inc.select_work([date(2001, 1, 1)], {"2001-01-01": listing([])}, {}, [], 10)
    assert rows[0]["pendientes"] == 0
    assert work is None


@pytest.mark.parametrize("start,end", [("2001-02-30", "2001-03-01"),
                                       ("2001-02-02", "2001-02-01"),
                                       ("2001-01-01", "2001-02-01")])
def test_rechaza_rangos_invalidos(start, end):
    with pytest.raises(ValueError):
        inc.date_range(start, end)


def test_plan_no_escribe_no_captura_no_carga_modelo_no_publica(tmp_path, monkeypatch):
    monkeypatch.setattr(inc, "ROOT", tmp_path)
    monkeypatch.setattr(inc, "read_inventory", Mock(return_value=({}, {}, [])))
    execute = Mock(side_effect=AssertionError("No debe ejecutar"))
    monkeypatch.setattr(inc, "execute", execute)
    assert inc.main(["--desde", "2001-01-01", "--hasta", "2001-01-02", "--publicar-bigquery"]) == 0
    execute.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_tramo_guarda_listado_completo_y_omite_consulta_de_listado(tmp_path, api):
    snap = listing(["1-1-LE26", "2-1-LE26", "3-1-LE26"])
    folder = inc.capture_segment(tmp_path, "lote-nuevo", "2001-01-01", 1, 1, snap)
    manifest, records, _ = batch._load_capture(folder)
    assert manifest["seleccionados"] == ["2-1-LE26"]
    assert (folder / "listado.json").read_bytes() == snap.raw
    assert records[0]["CodigoExterno"] == "2-1-LE26"
    api.get_licitaciones_por_fecha.assert_not_called()
    api.get_licitacion_por_codigo.assert_called_once_with("2-1-LE26")


def test_alterar_desplazamiento_no_permite_cambiar_seleccion(tmp_path, api):
    folder = inc.capture_segment(tmp_path, "lote-nuevo", "2001-01-01", 1, 1,
                                 listing(["1-1-LE26", "2-1-LE26"]))
    manifest = inc._read(folder / "captura.json")
    manifest["seleccion_incremental"]["desplazamiento"] = 0
    (folder / "captura.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selección"):
        batch._load_capture(folder)


def test_listado_vacio_se_conserva_con_una_sola_consulta(tmp_path, api):
    api.get_licitaciones_por_fecha.return_value = {"Listado": []}
    folder = inc.capture_segment(tmp_path, "lote-vacio", "2001-01-01", 10, 0)
    assert inc._read(folder / "captura.json")["estado"] == "vacio"
    api.get_licitacion_por_codigo.assert_not_called()


def test_recuperar_tramo_reutiliza_detalle_correcto_y_solo_consulta_fallido(tmp_path, api):
    snap = listing(["1-1-LE26", "2-1-LE26", "3-1-LE26"])
    def partial(code):
        if code == "3-1-LE26":
            raise RuntimeError("https://ejemplo/?ticket=secreto")
        return detail(code)
    api.get_licitacion_por_codigo.side_effect = partial
    with pytest.raises(RuntimeError, match="Captura pendiente"):
        inc.capture_segment(tmp_path, "lote-parcial", "2001-01-01", 2, 1, snap)
    folder = inc._folder(tmp_path, "lote-parcial")
    original, _, _ = batch._load_capture(folder, permitir_incompleto=True)
    assert "secreto" not in (folder / "captura.json").read_text(encoding="utf-8")
    first_bytes = (folder / "details/2-1-LE26.json").read_bytes()
    api.get_licitacion_por_codigo.reset_mock()
    api.get_licitacion_por_codigo.side_effect = detail
    recovered = inc.capture_segment(tmp_path, "lote-recuperado", "2001-01-01", 2, 1, snap, original)
    manifest, _, _ = batch._load_capture(recovered)
    assert manifest["estado"] == "completo"
    assert manifest["detalles"][0]["captura_utc"] == original["detalles"][0]["captura_utc"]
    assert (recovered / "details/2-1-LE26.json").read_bytes() == first_bytes
    api.get_licitacion_por_codigo.assert_called_once_with("3-1-LE26")


def test_429_persistente_corta_y_conserva_pendientes(tmp_path, api):
    api.get_licitacion_por_codigo.side_effect = RuntimeError("Error al consultar Mercado Público. Estado: 429")
    with pytest.raises(RuntimeError):
        inc.capture_segment(tmp_path, "lote-limitado", "2001-01-01", 2, 0,
                            listing(["1-1-LE26", "2-1-LE26"]))
    manifest, _, _ = batch._load_capture(inc._folder(tmp_path, "lote-limitado"), permitir_incompleto=True)
    assert manifest["estado"] == "incompleto"
    assert manifest["control_api"]["bloqueado_por_429"] is True
    assert api.get_licitacion_por_codigo.call_count == 2
    assert manifest["detalles"][1]["motivo"] == "no_intentado"


def test_bloqueo_no_permite_dos_ejecutores_y_libera_al_salir(tmp_path):
    with inc.single_writer(tmp_path):
        with pytest.raises(RuntimeError, match="otra ejecución"):
            with inc.single_writer(tmp_path):
                pass
    assert not (tmp_path / inc.STATE_REL.parent / "ejecucion.lock").exists()


def test_cierre_durante_captura_conserva_manifiesto_recuperable(tmp_path, api):
    api.get_licitacion_por_codigo.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        inc.capture_segment(tmp_path, "lote-interrumpido", "2001-01-01", 1, 0,
                            listing(["1-1-LE26"]))
    manifest, _, _ = batch._load_capture(inc._folder(tmp_path, "lote-interrumpido"), permitir_incompleto=True)
    assert manifest["estado"] == "incompleto"


def test_inventario_real_lee_tramo_sin_escribir_historial(tmp_path, api):
    snap = listing(["1-1-LE26", "2-1-LE26"])
    inc.capture_segment(tmp_path, "lote-inventario", "2001-01-01", 1, 1, snap)
    snapshots, done, pending = inc.read_inventory(tmp_path)
    assert snapshots["2001-01-01"].codes == snap.codes
    assert done == {}
    assert pending == [("2001-01-01", "lote-inventario")]
    assert not (tmp_path / "data/processed/history").exists()


def test_finish_local_reutiliza_prediccion_sin_cargar_modelo(tmp_path, api, args, monkeypatch):
    folder = inc.capture_segment(tmp_path, "lote-listo", "2001-01-01", 1, 0, listing(["1-1-LE26"]))
    monkeypatch.setattr(inc, "_prediction", Mock(return_value=tmp_path / "predicciones.csv"))
    predict = Mock(side_effect=AssertionError("No debe predecir"))
    monkeypatch.setattr(batch, "predecir_lote", predict)
    monkeypatch.setattr(inc.history, "build_history", Mock(return_value=tmp_path / "historial"))
    publish = Mock(side_effect=AssertionError("No debe publicar"))
    monkeypatch.setattr(inc, "publish", publish)
    state = {"lote": folder.name, "fase": "prediccion_pendiente"}
    inc.finish(tmp_path, state, args)
    predict.assert_not_called()
    publish.assert_not_called()
    assert state["fase"] == "completo"
    assert state["publicado_bigquery"] is False


def test_fallo_publicacion_persiste_fase_y_bloquea_nueva_api(tmp_path, args, monkeypatch):
    snapshot = tmp_path / "data/processed/history/snapshot"
    snapshot.mkdir(parents=True)
    args.publicar_bigquery = True
    monkeypatch.setattr(inc, "prepare_history", Mock(return_value=object()))
    monkeypatch.setattr(inc, "apply_history", Mock(side_effect=RuntimeError("resultado desconocido")))
    state = {"lote": "lote-listo", "fase": "prediccion_pendiente"}
    with pytest.raises(RuntimeError):
        inc.publish(tmp_path, snapshot, state, args)
    assert inc._read(tmp_path / inc.STATE_REL)["fase"] == "publicacion_en_curso"
    monkeypatch.setattr(inc, "read_inventory", Mock(return_value=({}, {}, [])))
    capture = Mock(side_effect=AssertionError("No debe consultar"))
    monkeypatch.setattr(inc, "capture_segment", capture)
    with pytest.raises(RuntimeError, match="sin confirmación"):
        inc.execute(tmp_path, [date(2001, 1, 1)], args)
    capture.assert_not_called()


def test_reanudar_publicacion_sin_lote_no_consulta_api(tmp_path, args, monkeypatch):
    snapshot = tmp_path / "data/processed/history/snapshot"
    snapshot.mkdir(parents=True)
    args.publicar_bigquery = True
    args.reintentar_publicacion = True
    state = {"lote": None, "fase": "publicacion_en_curso", "destino": inc._target(args),
             "historial": snapshot.relative_to(tmp_path).as_posix()}
    inc._atomic_json(tmp_path / inc.STATE_REL, state)
    monkeypatch.setattr(inc, "read_inventory", Mock(return_value=({}, {}, [])))
    monkeypatch.setattr(inc, "prepare_history", Mock(return_value=object()))
    apply = Mock(return_value={"job_id": "simulado"})
    monkeypatch.setattr(inc, "apply_history", apply)
    capture = Mock(side_effect=AssertionError("No debe consultar"))
    monkeypatch.setattr(inc, "capture_segment", capture)
    assert inc.execute(tmp_path, [date(2001, 1, 1)], args) == 0
    capture.assert_not_called()
    apply.assert_called_once()
    assert inc._read(tmp_path / inc.STATE_REL)["fase"] == "completo"


def test_reintentar_sin_publicacion_pendiente_no_inicia_captura(tmp_path, args, monkeypatch):
    args.reintentar_publicacion = True
    args.publicar_bigquery = True
    monkeypatch.setattr(inc, "read_inventory", Mock(return_value=({}, {}, [])))
    with pytest.raises(ValueError, match="No hay"):
        inc.execute(tmp_path, [date(2001, 1, 1)], args)


def test_ruta_de_lote_no_puede_salir_de_batches(tmp_path):
    with pytest.raises(ValueError):
        inc._folder(tmp_path, "../fuera")


def test_dos_ejecuciones_del_mismo_rango_toman_segmentos_distintos(tmp_path, api, args, monkeypatch):
    snap = listing(["1-1-LE26", "2-1-LE26", "3-1-LE26", "4-1-LE26"])
    done = {}
    args.limite = 2
    monkeypatch.setattr(inc, "read_inventory", lambda root: ({"2001-01-01": snap}, done, []))
    def finish(root, state, options):
        manifest = inc._read(inc._folder(root, state["lote"]) / "captura.json")
        done.setdefault(state["fecha"], set()).update(manifest["seleccionados"])
        state["fase"] = "completo"
        inc._atomic_json(root / inc.STATE_REL, state)
    monkeypatch.setattr(inc, "finish", finish)
    assert inc.execute(tmp_path, [date(2001, 1, 1)], args) == 0
    assert inc.execute(tmp_path, [date(2001, 1, 1)], args) == 0
    assert [call.args[0] for call in api.get_licitacion_por_codigo.call_args_list] == list(snap.codes)
    api.get_licitaciones_por_fecha.assert_not_called()


def test_execute_recupera_solo_el_fallo_del_lote_pendiente(tmp_path, api, args, monkeypatch):
    snap = listing(["1-1-LE26", "2-1-LE26"])
    def partial(code):
        if code == "2-1-LE26":
            raise RuntimeError("fallo simulado")
        return detail(code)
    api.get_licitacion_por_codigo.side_effect = partial
    with pytest.raises(RuntimeError):
        inc.capture_segment(tmp_path, "lote-parcial", "2001-01-01", 2, 0, snap)
    state = {"fase": "captura_pendiente", "lote": "lote-parcial", "fecha": "2001-01-01"}
    inc._atomic_json(tmp_path / inc.STATE_REL, state)
    monkeypatch.setattr(inc, "read_inventory", Mock(return_value=({}, {}, [])))
    finished = Mock()
    monkeypatch.setattr(inc, "finish", finished)
    api.get_licitacion_por_codigo.reset_mock()
    api.get_licitacion_por_codigo.side_effect = detail
    assert inc.execute(tmp_path, [date(2001, 1, 1)], args) == 0
    api.get_licitacion_por_codigo.assert_called_once_with("2-1-LE26")
    finished.assert_called_once()
    new_state = inc._read(tmp_path / inc.STATE_REL)
    assert new_state["lote"] != "lote-parcial"
    batch._load_capture(inc._folder(tmp_path, new_state["lote"]))


def test_reanudar_publicacion_del_lote_no_repite_api_ni_prediccion(tmp_path, api, args, monkeypatch):
    folder = inc.capture_segment(tmp_path, "lote-publicacion", "2001-01-01", 1, 0, listing(["1-1-LE26"]))
    args.publicar_bigquery = True
    args.reintentar_publicacion = True
    inc._atomic_json(tmp_path / inc.STATE_REL,
                     {"fase": "publicacion_en_curso", "lote": folder.name, "destino": inc._target(args)})
    monkeypatch.setattr(inc, "read_inventory", Mock(return_value=({}, {}, [])))
    monkeypatch.setattr(inc, "_prediction", Mock(return_value=tmp_path / "predicciones.csv"))
    predict = Mock(side_effect=AssertionError("No debe predecir"))
    monkeypatch.setattr(batch, "predecir_lote", predict)
    snapshot = tmp_path / "data/processed/history/snapshot"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(inc.history, "build_history", Mock(return_value=snapshot))
    monkeypatch.setattr(inc, "prepare_history", Mock(return_value=object()))
    apply = Mock(return_value={"job_id": "simulado"})
    monkeypatch.setattr(inc, "apply_history", apply)
    api.get_licitacion_por_codigo.reset_mock()
    assert inc.execute(tmp_path, [date(2001, 1, 1)], args) == 0
    api.get_licitacion_por_codigo.assert_not_called()
    predict.assert_not_called()
    apply.assert_called_once()
