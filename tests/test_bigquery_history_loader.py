"""Pruebas locales: el doble de cliente no ejecuta el dialecto SQL de BigQuery."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.warehouse import load_history as loader


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    builder = root / "src/reporting/build_batch_history.py"
    builder.parent.mkdir(parents=True)
    builder.write_text("# constructor de prueba\n", encoding="utf-8")
    source = root / "data/raw/example.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"Nombre": "Análisis ciudadano"}', encoding="utf-8")
    folder = root / "data/processed/history/ejemplo"
    folder.mkdir(parents=True)
    schemas = json.loads((loader.ROOT / "config/bigquery/history_schemas.json").read_text(encoding="utf-8"))
    rows = {}
    default = {"STRING": "ejemplo", "INT64": "1", "FLOAT64": "0.0", "BOOL": "False",
               "DATE": "2026-10-16", "DATETIME": "2026-10-16T12:30:00", "TIMESTAMP": "2026-10-16T15:30:00+00:00"}
    for table, fields in schemas.items():
        rows[table] = [{f["name"]: default[f["type"]] for f in fields}]
    for table in rows:
        for row in rows[table]:
            row.update({key: value for key, value in {"LoteId": "lote_1", "CodigoExterno": "101-1-LE26", "CapturaId": "captura_1",
                       "PrediccionId": "prediccion_1", "AfinidadPredicha": "0"}.items() if key in row})
    rows["predicciones"][0].update(ProbAfinidad0="1.0", AfinidadEsperada="0.0")
    rows["revisiones"][0].update(AfinidadHumana="3", BasesRevisadas="False", IncorporadoEntrenamiento="False")
    rows["vista_actual"] = [{"CodigoExterno": "101-1-LE26"}]

    def write():
        for table, records in rows.items():
            fields = [f["name"] for f in schemas[table]] if table in schemas else ["CodigoExterno"]
            with (folder / (table + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
        manifest = {"schema_version": 1, "constructor_sha256": sha(builder.read_bytes()),
                    "entradas_sha256": {"data/raw/example.json": sha(source.read_bytes())},
                    "salidas_sha256": {t + ".csv": sha((folder / (t + ".csv")).read_bytes()) for t in rows},
                    "conteos": {t: len(r) for t, r in rows.items()}}
        identity = ["history", 1, manifest["constructor_sha256"], manifest["entradas_sha256"]]
        manifest["huella_entradas"] = sha(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode())
        (folder / "resumen.json").write_text(json.dumps(manifest), encoding="utf-8")
    write()
    return SimpleNamespace(root=root, folder=folder, source=source, builder=builder, rows=rows, write=write)


def test_plan_preserves_zero_unicode_review_and_source_bytes(snapshot):
    before = {str(p): p.read_bytes() for p in snapshot.root.rglob("*") if p.is_file()}
    plan = loader.prepare_history(snapshot.folder, snapshot.root)
    prediction = json.loads(plan.payloads["predicciones"])
    review = json.loads(plan.payloads["revisiones"])
    assert prediction["AfinidadPredicha"] == 0
    assert review["AfinidadHumana"] == 3
    assert review["BasesRevisadas"] is False
    assert plan.counts == dict.fromkeys(loader.KEYS, 1)
    assert plan.summary()["escrituras_cloud"] is False
    assert before == {str(p): p.read_bytes() for p in snapshot.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("target", ["source", "builder", "csv"])
def test_tampering_stops_before_upload(snapshot, target):
    path = snapshot.folder / "capturas.csv" if target == "csv" else getattr(snapshot, target)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        loader.prepare_history(snapshot.folder, snapshot.root)


@pytest.mark.parametrize("table,field,value", [
    ("lote_captura", "LoteId", "missing"),
    ("predicciones", "CapturaId", "missing"),
    ("revisiones", "PrediccionId", "missing"),
    ("revisiones", "AfinidadPredicha", "2"),
    ("revisiones", "AfinidadHumana", "4"),
    ("predicciones", "ProbAfinidad0", "nan"),
    ("predicciones", "ProbAfinidad0", "0.1"),
    ("predicciones", "AfinidadEsperada", "1.0"),
    ("predicciones", "AfinidadPredicha", ""),
    ("capturas", "FechaCapturaUTC", "2026-10-16T15:30:00"),
    ("capturas", "FechaCierre", "2026-10-16T15:30:00+00:00"),
    ("revisiones", "BasesRevisadas", "false"),
    ("lotes", "ConsultasAPI", str(2**63)),
])
def test_invalid_typed_data_and_links_are_rejected(snapshot, table, field, value):
    snapshot.rows[table][0][field] = value
    snapshot.write()
    with pytest.raises(ValueError):
        loader.prepare_history(snapshot.folder, snapshot.root)


def test_duplicate_identity_rejected(snapshot):
    snapshot.rows["capturas"].append(dict(snapshot.rows["capturas"][0]))
    snapshot.write()
    with pytest.raises(ValueError, match="duplicada"):
        loader.prepare_history(snapshot.folder, snapshot.root)


def test_missing_predictions_and_reviews_are_valid(snapshot):
    snapshot.rows["predicciones"] = []
    snapshot.rows["revisiones"] = []
    snapshot.write()
    plan = loader.prepare_history(snapshot.folder, snapshot.root)
    assert plan.counts["predicciones"] == plan.counts["revisiones"] == 0
    assert plan.current_count == 1


def test_manifest_cannot_escape_root(snapshot):
    path = snapshot.folder / "resumen.json"
    manifest = json.loads(path.read_text())
    manifest["entradas_sha256"] = {"../escape.json": "0" * 64}
    identity = ["history", 1, manifest["constructor_sha256"], manifest["entradas_sha256"]]
    manifest["huella_entradas"] = sha(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ruta"):
        loader.prepare_history(snapshot.folder, snapshot.root)


def test_transaction_uses_insert_only_and_keeps_guards(snapshot):
    plan = loader.prepare_history(snapshot.folder, snapshot.root)
    sql = loader.transaction_sql(plan, "project-demo", "mercado_publico", "a" * 32)
    assert sql.count("MERGE ") == 5
    assert sql.index("BEGIN TRANSACTION") < sql.index("MERGE ") < sql.index("COMMIT TRANSACTION")
    assert "WHEN MATCHED" not in sql and "UPDATE " not in sql and "DELETE " not in sql
    assert "Contenido distinto para identidad existente en revisiones" in sql
    assert "Revision sin prediccion coincidente" in sql
    assert "IS DISTINCT FROM" in sql
    assert "{capturas}" not in sql and "__carga_" + "a" * 32 in sql
    assert "0 AS Insertadas" not in sql  # Se calculan conteos en el servidor.
    with pytest.raises(ValueError):
        loader.transaction_sql(plan, "project-demo`", "mercado_publico", "a" * 32)


class Field:
    @classmethod
    def from_api_repr(cls, value):
        return SimpleNamespace(name=value["name"], field_type=value["type"], mode=value["mode"])


class Table:
    def __init__(self, name, schema):
        self.name, self.schema, self.table_type = name, schema, "TABLE"


SDK = SimpleNamespace(SchemaField=Field, Table=Table, LoadJobConfig=lambda **kw: SimpleNamespace(**kw),
    QueryJobConfig=lambda **kw: SimpleNamespace(**kw), SourceFormat=SimpleNamespace(NEWLINE_DELIMITED_JSON="JSON"),
    WriteDisposition=SimpleNamespace(WRITE_EMPTY="WRITE_EMPTY"), CreateDisposition=SimpleNamespace(CREATE_NEVER="CREATE_NEVER"))


class FakeClient:
    """Comprueba el orden del protocolo; no simula la ejecución SQL."""
    def __init__(self, plan):
        self.plan, self.created, self.loaded, self.deleted, self.queries = plan, [], [], [], []
        self.region, self.bad_schema, self.fail_load, self.fail_query = "us-central1", False, False, False

    def get_dataset(self, name):
        return SimpleNamespace(location=self.region)

    def get_table(self, name):
        table = name.rsplit(".", 1)[-1]
        fields = self.plan.schemas[table][:-1] if self.bad_schema else self.plan.schemas[table]
        return Table(name, [Field.from_api_repr(f) for f in fields])

    def create_table(self, table):
        self.created.append(table)

    def load_table_from_file(self, stream, name, **kwargs):
        if self.fail_load:
            raise RuntimeError("Fallo de transporte en carga auxiliar")
        data = stream.read()
        self.loaded.append((name, data, kwargs))
        return SimpleNamespace(result=lambda: None, output_rows=len(data.splitlines()))

    def query(self, sql, **kwargs):
        self.queries.append((sql, kwargs))
        if self.fail_query:
            raise RuntimeError("Resultado de transacción desconocido")
        return SimpleNamespace(result=lambda: [{"Tabla": "lotes", "Antes": 1, "Despues": 1, "Insertadas": 0}])

    def delete_table(self, name, **kwargs):
        self.deleted.append(name)


def test_upload_protocol_expiration_and_cleanup(snapshot):
    plan = loader.prepare_history(snapshot.folder, snapshot.root)
    client = FakeClient(plan)
    report = loader.apply_history(plan, "project-demo", "mercado_publico", client=client, sdk=SDK)
    assert len(client.created) == len(client.loaded) == len(client.deleted) == 5
    assert all((t.expires - datetime.now(timezone.utc)).total_seconds() > 23 * 3600 for t in client.created)
    assert all(item[2]["job_config"].write_disposition == "WRITE_EMPTY" for item in client.loaded)
    assert client.queries[0][1]["job_retry"] is None
    assert client.queries[0][1]["job_config"].maximum_bytes_billed == 1_073_741_824
    assert report["tablas"][0]["Insertadas"] == 0


@pytest.mark.parametrize("failure", ["region", "schema", "load", "query", "changed_source"])
def test_failure_does_not_clean_unknown_jobs_or_publish_bad_data(snapshot, failure):
    plan = loader.prepare_history(snapshot.folder, snapshot.root)
    client = FakeClient(plan)
    if failure == "region":
        client.region = "US"
    elif failure == "schema":
        client.bad_schema = True
    elif failure == "load":
        client.fail_load = True
    elif failure == "query":
        client.fail_query = True
    else:
        snapshot.source.write_bytes(b"changed")
    with pytest.raises((RuntimeError, ValueError)):
        loader.apply_history(plan, "project-demo", "mercado_publico", client=client, sdk=SDK)
    assert not client.deleted
    if failure != "query":
        assert not client.queries
    if failure in {"region", "schema", "changed_source"}:
        assert not client.created and not client.loaded


def test_empty_optional_tables_create_schema_without_empty_upload(snapshot):
    snapshot.rows["revisiones"] = []
    snapshot.write()
    plan = loader.prepare_history(snapshot.folder, snapshot.root)
    client = FakeClient(plan)
    loader.apply_history(plan, "project-demo", "mercado_publico", client=client, sdk=SDK)
    assert len(client.created) == 5 and len(client.loaded) == 4


def test_cli_plan_does_not_invoke_cloud(snapshot, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("El plan intentó conectarse a Google")
    monkeypatch.setattr(loader, "apply_history", unexpected)
    assert loader.main(["--historial", str(snapshot.folder), "--root", str(snapshot.root)]) == 0
