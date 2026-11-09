"""Presupuestos, prioridad temporal y cadencia; sin API ni modelos reales."""
from datetime import date
import json
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from src import run_incremental as inc
from src import run_incremental_batch as runner


def snap(day, size=40):
    codes = tuple(f"{day}{i:03d}-1-LE26" for i in range(size))
    raw = json.dumps({"Listado": [{"CodigoExterno": code} for code in codes]}).encode("utf-8")
    return inc.Listing("origen", raw, "2001-01-04T00:00:00+00:00", codes)


@pytest.fixture
def options():
    return SimpleNamespace(desde="2001-01-01", hasta="2001-01-03", limite=10,
                           max_detalles=30, max_segmentos=3, max_consultas=66,
                           max_segundos=600, orden="recientes", ejecutar=True,
                           publicar_bigquery=False, reintentar_publicacion=False,
                           project="proyecto", dataset="mercado_publico", location="us-central1", max_bytes=1024)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = SimpleNamespace(root=tmp_path, done={}, calls=[], operations=[], now=0.0)
    data.days = [date(2001, 1, i) for i in (1, 2, 3)]
    data.listings = {day.isoformat(): snap(day.day) for day in data.days}
    monkeypatch.setattr(inc, "read_inventory", lambda root: (data.listings, data.done, []))
    original_control = inc.batch._RequestControl
    def wait(seconds):
        data.now += seconds
    monkeypatch.setattr(inc.batch, "_RequestControl", lambda: original_control(clock=lambda: data.now, wait=wait))

    def execute(root, days, args, *, control=None, locked=False):
        assert locked and (root / inc.STATE_REL.parent / "ejecucion.lock").is_file()
        _, work = inc.select_work(days, data.listings, data.done, [], args.limite)
        if work is None:
            inc._atomic_json(root / inc.STATE_REL, {"fase": "completo", "lote": None})
            data.operations.append("publicacion_sin_captura")
            return 0
        data.operations.append(work["fecha"])
        lot_id = f"lote-{len(data.operations)}"
        folder = inc._folder(root, lot_id)
        folder.mkdir(parents=True)
        details = []
        for code in work["codigos"]:
            def request(value):
                data.calls.append(value)
                return {}
            control.request(request, code)
            details.append({"codigo": code, "estado": "correcto", "intentos_http": 1})
        (folder / "captura.json").write_text(json.dumps({"detalles": details}), encoding="utf-8")
        data.done.setdefault(work["fecha"], set()).update(work["codigos"])
        inc._atomic_json(root / inc.STATE_REL, {"fase": "completo", "lote": lot_id,
                                              "fecha": work["fecha"], "publicado_bigquery": args.publicar_bigquery})
        return 0
    data.execute = Mock(side_effect=execute)
    monkeypatch.setattr(inc, "execute", data.execute)
    return data


def test_recientes_rota_por_tres_fechas_sin_repetir_la_primera(env, options):
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert env.operations == ["2001-01-03", "2001-01-02", "2001-01-01"]
    assert result["detalles_consultados"] == 30
    assert result["consultas_api"] == 30
    assert len(set(env.calls)) == 30
    assert result["motivo_fin"] == "maximo_segmentos"


def test_backfill_orden_antiguas_atiende_primero_la_mas_antigua(env, options):
    options.orden = "antiguas"
    runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert env.operations == ["2001-01-01", "2001-01-02", "2001-01-03"]


def test_presupuesto_25_ajusta_ultimo_segmento_a_cinco(env, options):
    options.max_detalles = 25
    options.max_segmentos = 5
    result = runner.run(env.root, env.days[:1], options, clock=lambda: env.now)
    assert [row["detalles_consultados"] for row in result["operaciones"]] == [10, 10, 5]
    assert result["detalles_consultados"] == 25
    assert result["motivo_fin"] == "presupuesto_detalles"
    assert len(set(env.calls)) == 25


def test_reserva_de_reintentos_no_supera_presupuesto_http(env, options):
    options.max_consultas = 20
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert result["consultas_api"] == 10
    assert result["motivo_fin"] == "presupuesto_consultas_api"
    assert env.execute.call_count == 1


def test_tiempo_no_interrumpe_segmento_pero_evitar_iniciar_otro(env, options):
    options.max_segundos = 1
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert result["detalles_consultados"] == 10
    assert result["duracion_segundos"] > options.max_segundos
    assert result["motivo_fin"] == "tiempo_para_iniciar_segmentos_agotado"
    assert env.execute.call_count == 1


def test_rango_agotado_no_hace_api_y_no_gira_indefinidamente(env, options):
    env.done.update({day: set(value.codes) for day, value in env.listings.items()})
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert result["operaciones"] == []
    assert result["consultas_api"] == 0
    env.execute.assert_not_called()


def test_rango_agotado_puede_publicar_una_sola_vez(env, options):
    env.done.update({day: set(value.codes) for day, value in env.listings.items()})
    options.publicar_bigquery = True
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert result["consultas_api"] == 0
    assert env.operations == ["publicacion_sin_captura"]
    env.execute.assert_called_once()


def test_fallo_en_segundo_segmento_corta_corrida_y_guarda_diagnostico(env, options):
    original = env.execute.side_effect
    def fail(root, days, args, **kwargs):
        if len(env.operations) == 1:
            raise RuntimeError("ticket=secreto-prueba")
        return original(root, days, args, **kwargs)
    env.execute.side_effect = fail
    with pytest.raises(RuntimeError):
        runner.run(env.root, env.days, options, clock=lambda: env.now)
    reports = list((env.root / "data/processed/incremental/ejecuciones").glob("*.json"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["estado"] == "fallido"
    assert report["consultas_api"] == 10
    assert report["detalles_consultados"] == 10
    assert len(report["operaciones"]) == 1
    assert "secreto" not in text
    assert not (env.root / inc.STATE_REL.parent / "ejecucion.lock").exists()


def test_plan_no_escribe_ni_ejecuta_aunque_se_pida_bigquery(env, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", env.root)
    before = sorted(str(path) for path in env.root.rglob("*"))
    run = Mock(side_effect=AssertionError("No debe ejecutar"))
    monkeypatch.setattr(runner, "run", run)
    assert runner.main(["--desde", "2001-01-01", "--hasta", "2001-01-03", "--publicar-bigquery"]) == 0
    run.assert_not_called()
    assert sorted(str(path) for path in env.root.rglob("*")) == before


def test_publicacion_pendiente_impide_comenzar_capturas(env, options):
    inc._atomic_json(env.root / inc.STATE_REL,
                     {"fase": "publicacion_en_curso", "lote": "pendiente", "destino": inc._target(options)})
    options.publicar_bigquery = True
    with pytest.raises(RuntimeError, match="sin confirmar"):
        runner.run(env.root, env.days, options, clock=lambda: env.now)
    env.execute.assert_not_called()
    assert env.calls == []


def test_reintento_explicito_solo_publica_y_termina_sin_otro_segmento(env, options):
    options.publicar_bigquery = True
    options.reintentar_publicacion = True
    inc._atomic_json(env.root / inc.STATE_REL,
                     {"fase": "publicacion_en_curso", "lote": "pendiente", "destino": inc._target(options)})
    def resume(root, days, args, **kwargs):
        assert args.reintentar_publicacion
        inc._atomic_json(root / inc.STATE_REL, {"fase": "completo", "lote": "pendiente",
                                              "publicado_bigquery": True, "publicacion": {"job_id": "simulado"}})
    env.execute.side_effect = resume
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert result["motivo_fin"] == "publicacion_reanudada"
    assert result["consultas_api"] == 0
    env.execute.assert_called_once()


def test_recuperacion_no_comienza_si_no_cabe_en_presupuesto(env, options):
    folder = inc._folder(env.root, "pendiente")
    folder.mkdir(parents=True)
    (folder / "captura.json").write_text(json.dumps({"estado": "incompleto",
        "plan": {"fecha_iso": "2001-01-01"}, "detalles": [{"estado": "fallido"}] * 4}), encoding="utf-8")
    inc._atomic_json(env.root / inc.STATE_REL, {"fase": "captura_pendiente", "lote": "pendiente"})
    options.max_detalles = 3
    result = runner.run(env.root, env.days, options, clock=lambda: env.now)
    assert result["motivo_fin"] == "presupuesto_detalles"
    env.execute.assert_not_called()


@pytest.mark.parametrize("flag,value", [("--max-detalles", "0"), ("--max-detalles", "101"),
                                        ("--max-segmentos", "21"), ("--max-consultas", "0"),
                                        ("--max-segundos", "0"), ("--limite", "11")])
def test_presupuestos_invalidos_se_rechazan_sin_escribir(env, monkeypatch, flag, value):
    monkeypatch.setattr(runner, "ROOT", env.root)
    with pytest.raises(SystemExit):
        runner.main(["--desde", "2001-01-01", "--hasta", "2001-01-03", flag, value])
    assert not (env.root / inc.STATE_REL.parent).exists()


def test_contador_y_cadencia_compartidos_no_contaminan_manifiestos(tmp_path, monkeypatch):
    api = ModuleType("src.api.mercado_publico_client")
    api.get_licitaciones_por_fecha = Mock(side_effect=AssertionError("Listado no esperado"))
    api.get_licitacion_por_codigo = Mock(side_effect=lambda code: {"Listado": [{"CodigoExterno": code,
                                                                           "Items": {"Listado": []}}]})
    monkeypatch.setitem(sys.modules, "src.api.mercado_publico_client", api)
    now = [0.0]
    def wait(seconds):
        now[0] += seconds
    control = inc.batch._RequestControl(clock=lambda: now[0], wait=wait)
    snapshot = snap(1, 2)
    first = inc.capture_segment(tmp_path, "lote-primero", "2001-01-01", 1, 0, snapshot, control=control)
    second = inc.capture_segment(tmp_path, "lote-segundo", "2001-01-01", 1, 1, snapshot, control=control)
    a, b = inc._read(first / "captura.json"), inc._read(second / "captura.json")
    assert control.attempts == 2
    assert a["consultas_api"] == b["consultas_api"] == 1
    assert len(a["control_api"]["eventos"]) == len(b["control_api"]["eventos"]) == 1
    assert now[0] >= 3
    inc.batch._load_capture(first)
    inc.batch._load_capture(second)
