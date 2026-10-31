"""Valida un historial local y, con --aplicar, lo incorpora a BigQuery.

No consulta Mercado Público, no carga modelos y no modifica fuentes locales.
La planificación funciona sin instalar bibliotecas de Google ni autenticarse.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
KEYS = {
    "lotes": ("LoteId",), "capturas": ("CapturaId",),
    "lote_captura": ("LoteId", "CodigoExterno"),
    "predicciones": ("PrediccionId",), "revisiones": ("RevisionHistorialId",),
}
FILES = {name + ".csv" for name in KEYS} | {"vista_actual.csv"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _checked_file(base, relative, expected):
    path = (base / relative).resolve()
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_relative_to(base):
        raise ValueError("Una ruta del resumen sale de su carpeta de origen.")
    data = path.read_bytes()
    if _sha(data) != expected:
        raise ValueError(f"La huella no coincide: {relative}")
    return data


def _convert(value, kind):
    if value == "":
        return None
    if kind in {"INT64", "INTEGER"}:
        if not re.fullmatch(r"-?[0-9]+", value):
            raise ValueError("Entero inválido.")
        number = int(value)
        if not -(2**63) <= number < 2**63:
            raise ValueError("Entero fuera del rango INT64.")
        return number
    if kind in {"FLOAT64", "FLOAT"}:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Número no finito.")
        return number
    if kind in {"BOOL", "BOOLEAN"}:
        if value not in {"True", "False"}:
            raise ValueError("Booleano inválido.")
        return value == "True"
    if kind == "DATE":
        return date.fromisoformat(value).isoformat()
    if kind in {"TIMESTAMP", "DATETIME"}:
        parsed = datetime.fromisoformat(value)
        if kind == "TIMESTAMP":
            if parsed.tzinfo is None:
                raise ValueError("Un instante UTC necesita zona horaria.")
            return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
        if parsed.tzinfo is not None:
            raise ValueError("Una fecha civil no debe tener zona horaria.")
        return parsed.isoformat(sep=" ", timespec="microseconds")
    if kind != "STRING":
        raise ValueError(f"Tipo no admitido: {kind}")
    return value


def _validate_rows(rows):
    indexes = {}
    for table, keys in KEYS.items():
        index = {}
        for row in rows[table]:
            key = tuple(row[field] for field in keys)
            if key in index:
                raise ValueError(f"Identidad duplicada en {table}.")
            index[key] = row
        indexes[table] = index
    for row in rows["lote_captura"]:
        capture = indexes["capturas"].get((row["CapturaId"],))
        if (row["LoteId"],) not in indexes["lotes"] or not capture or capture["CodigoExterno"] != row["CodigoExterno"]:
            raise ValueError("Relación lote-captura inválida.")
    for row in rows["predicciones"]:
        capture = indexes["capturas"].get((row["CapturaId"],))
        link = indexes["lote_captura"].get((row["LoteId"], row["CodigoExterno"]))
        if not capture or capture["CodigoExterno"] != row["CodigoExterno"] or not link or link["CapturaId"] != row["CapturaId"]:
            raise ValueError("Predicción sin captura y lote coincidentes.")
        probs = [row[f"ProbAfinidad{i}"] for i in range(4)]
        expected = sum(i * prob for i, prob in enumerate(probs))
        if (row["AfinidadPredicha"] not in range(4) or not 0 <= row["AfinidadEsperada"] <= 3
                or any(not 0 <= p <= 1 for p in probs) or abs(sum(probs) - 1) > 1e-5
                or abs(row["AfinidadEsperada"] - expected) > 1e-5):
            raise ValueError("Clase, probabilidades o afinidad esperada inválidas.")
    for row in rows["revisiones"]:
        prediction = indexes["predicciones"].get((row["PrediccionId"],))
        if not prediction or any(row[k] != prediction[k] for k in ("CapturaId", "CodigoExterno", "LoteId", "AfinidadPredicha")):
            raise ValueError("Revisión sin predicción coincidente.")
        if row["AfinidadHumana"] is not None and row["AfinidadHumana"] not in range(4):
            raise ValueError("Afinidad humana inválida.")


@dataclass(frozen=True)
class HistoryPlan:
    fingerprint: str
    schemas: dict
    payloads: dict
    counts: dict
    current_count: int
    checked_hashes: dict

    def summary(self):
        return {"historial": self.fingerprint, "filas_a_comparar": self.counts,
                "licitaciones_en_vista_local": self.current_count,
                "archivos_verificados": len(self.checked_hashes), "escrituras_cloud": False}


def prepare_history(history, root=ROOT):
    """Verifica fuentes y CSV; congela los bytes a enviar, sin conexión cloud."""
    root, history = Path(root).resolve(), Path(history).resolve()
    schemas = json.loads((ROOT / "config/bigquery/history_schemas.json").read_text(encoding="utf-8"))
    manifest_bytes = (history / "resumen.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema_version"] != 1 or set(manifest["salidas_sha256"]) != FILES:
        raise ValueError("Versión o tablas del historial no admitidas.")
    if not manifest["entradas_sha256"]:
        raise ValueError("El historial no declara fuentes verificables.")
    builder = root / "src/reporting/build_batch_history.py"
    checked = {str(history / "resumen.json"): _sha(manifest_bytes)}
    if _sha(builder.read_bytes()) != manifest["constructor_sha256"]:
        raise ValueError("Cambió el constructor del historial. Reconstruye el historial.")
    checked[str(builder)] = manifest["constructor_sha256"]
    identity = ["history", 1, manifest["constructor_sha256"], manifest["entradas_sha256"]]
    fingerprint = _sha(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if fingerprint != manifest["huella_entradas"]:
        raise ValueError("La identidad del historial no coincide con sus fuentes.")
    for name, digest in manifest["entradas_sha256"].items():
        _checked_file(root, name, digest)
        checked[str((root / name).resolve())] = digest
    contents = {}
    for name, digest in manifest["salidas_sha256"].items():
        contents[name] = _checked_file(history, name, digest)
        checked[str(history / name)] = digest
    rows, payloads = {}, {}
    for table in KEYS:
        reader = csv.DictReader(io.StringIO(contents[table + ".csv"].decode("utf-8-sig"), newline=""))
        schema = schemas[table]
        if reader.fieldnames != [field["name"] for field in schema]:
            raise ValueError(f"Columnas inesperadas en {table}.")
        rows[table] = []
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Fila incompleta en {table}.")
            record = {f["name"]: _convert(raw[f["name"]], f["type"]) for f in schema}
            if any(record[f["name"]] is None for f in schema if f["mode"] == "REQUIRED"):
                raise ValueError(f"Campo obligatorio vacío en {table}.")
            rows[table].append(record)
        if len(rows[table]) != manifest["conteos"][table]:
            raise ValueError(f"Conteo distinto al resumen en {table}.")
        payloads[table] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows[table]).encode("utf-8")
    _validate_rows(rows)
    current = list(csv.DictReader(io.StringIO(contents["vista_actual.csv"].decode("utf-8-sig"), newline="")))
    if len(current) != manifest["conteos"]["vista_actual"] or len({r["CodigoExterno"] for r in current}) != len(current):
        raise ValueError("Conteo o códigos duplicados en la vista local.")
    return HistoryPlan(fingerprint, schemas, payloads, {k: len(v) for k, v in rows.items()}, len(current), checked)


def check_unchanged(plan):
    for path, digest in plan.checked_hashes.items():
        if _sha(Path(path).read_bytes()) != digest:
            raise ValueError("Una fuente cambió después de preparar la carga. Reconstruye el plan.")


def _prefix(project, dataset):
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ValueError("ID de proyecto inválido.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", dataset):
        raise ValueError("ID de dataset inválido.")
    return f"{project}.{dataset}"


def transaction_sql(plan, project, dataset, run_id):
    prefix = _prefix(project, dataset)
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Identificador de ejecución inválido.")
    target = {t: f"`{prefix}.{t}`" for t in KEYS}
    staging = {t: f"`{prefix}.__carga_{run_id}_{t}`" for t in KEYS}
    checks = (ROOT / "sql/bigquery/validar_historial.sql").read_text(encoding="utf-8")
    sql = [f"DECLARE antes_{t} INT64;\nDECLARE despues_{t} INT64;" for t in KEYS]
    sql += ["BEGIN TRANSACTION;", checks.format(**staging)]
    for table, keys in KEYS.items():
        fields = [f["name"] for f in plan.schemas[table]]
        match = " AND ".join(f"T.{k}=S.{k}" for k in keys)
        difference = " OR ".join(f"T.{f} IS DISTINCT FROM S.{f}" for f in fields)
        sql += [f"SET antes_{table} = (SELECT COUNT(*) FROM {target[table]});",
                f"ASSERT NOT EXISTS (SELECT 1 FROM {target[table]} T JOIN {staging[table]} S ON {match} WHERE {difference}) AS 'Contenido distinto para identidad existente en {table}';",
                f"MERGE {target[table]} T USING {staging[table]} S ON {match}\nWHEN NOT MATCHED THEN INSERT ({', '.join(fields)}) VALUES ({', '.join('S.' + f for f in fields)});"]
    sql.append(checks.format(**target))
    sql += [f"SET despues_{t} = (SELECT COUNT(*) FROM {target[t]});" for t in KEYS]
    sql += ["COMMIT TRANSACTION;", "\nUNION ALL\n".join(
        f"SELECT '{t}' AS Tabla, antes_{t} AS Antes, despues_{t} AS Despues, despues_{t}-antes_{t} AS Insertadas" for t in KEYS) + ";"]
    return "\n\n".join(sql)


def apply_history(plan, project, dataset, location="us-central1", *, max_bytes=1_073_741_824, client=None, sdk=None):
    """Sube auxiliares y confirma cinco MERGE juntos. Sólo un escritor a la vez."""
    prefix = _prefix(project, dataset)
    if max_bytes <= 0:
        raise ValueError("El límite de bytes debe ser positivo.")
    check_unchanged(plan)
    if sdk is None:
        try:
            from google.cloud import bigquery as sdk
        except ImportError as exc:
            raise RuntimeError("Instala requirements-bigquery.txt en la .venv del proyecto.") from exc
    own_client = client is None
    client = client or sdk.Client(project=project, location=location)
    run_id = uuid.uuid4().hex
    job_id = "historial_merge_" + run_id
    print("Trabajo BigQuery:", f"{project}:{location}.{job_id}", flush=True)
    stages = []
    try:
        remote = client.get_dataset(prefix)
        if remote.location.lower() != location.lower():
            raise ValueError("La región del dataset no coincide con la configuración.")
        aliases = {"INTEGER": "INT64", "FLOAT": "FLOAT64", "BOOLEAN": "BOOL"}
        # Comprobar todas las tablas antes de crear auxiliares o subir datos.
        for table in KEYS:
            existing = client.get_table(f"{prefix}.{table}")
            actual = [(f.name, aliases.get(f.field_type, f.field_type), f.mode) for f in existing.schema]
            expected = [(f["name"], f["type"], f["mode"]) for f in plan.schemas[table]]
            if existing.table_type != "TABLE" or actual != expected:
                raise ValueError(f"El esquema de {table} no coincide con el contrato.")
        query = transaction_sql(plan, project, dataset, run_id)
        for table in KEYS:
            stage_id = f"{prefix}.__carga_{run_id}_{table}"
            schema = [sdk.SchemaField.from_api_repr(f) for f in plan.schemas[table]]
            stage = sdk.Table(stage_id, schema=schema)
            stage.expires = datetime.now(timezone.utc) + timedelta(hours=24)
            client.create_table(stage)
            stages.append(stage_id)
            if plan.counts[table]:
                config = sdk.LoadJobConfig(schema=schema, source_format=sdk.SourceFormat.NEWLINE_DELIMITED_JSON,
                                           write_disposition=sdk.WriteDisposition.WRITE_EMPTY,
                                           create_disposition=sdk.CreateDisposition.CREATE_NEVER)
                job = client.load_table_from_file(io.BytesIO(plan.payloads[table]), stage_id,
                    job_config=config, location=location, job_id=f"historial_{run_id}_{table}")
                job.result()
                if job.output_rows != plan.counts[table]:
                    raise RuntimeError(f"No se cargaron todas las filas auxiliares de {table}.")
        # Una fuente cambiada detiene la confirmación; auxiliares expiran solos.
        check_unchanged(plan)
        config = sdk.QueryJobConfig(use_legacy_sql=False, maximum_bytes_billed=max_bytes)
        job = client.query(query, location=location, job_config=config, job_id=job_id, job_retry=None)
        result = [dict(row) for row in job.result()]
        print("Carga confirmada; las cinco tablas se incorporaron en una transacción.", flush=True)
        warnings = []
        for stage_id in stages:
            try:
                client.delete_table(stage_id, not_found_ok=True)
            except Exception:
                warnings.append(stage_id)
        return {"historial": plan.fingerprint, "job_id": job_id, "ubicacion": location,
                "tablas": result, "auxiliares_pendientes_de_expirar": warnings}
    except Exception:
        print("No se confirmó el resultado al cliente. Revisa el trabajo indicado en BigQuery antes de repetir. "
              "Las auxiliares creadas caducan en 24 horas.", file=sys.stderr, flush=True)
        raise
    finally:
        if own_client:
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historial", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--project", default="inteligencia-mercado-publico")
    parser.add_argument("--dataset", default="mercado_publico")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--max-bytes", type=int, default=1_073_741_824)
    parser.add_argument("--aplicar", action="store_true", help="Enviar datos y ejecutar MERGE en Google Cloud.")
    args = parser.parse_args(argv)
    try:
        _prefix(args.project, args.dataset)
        plan = prepare_history(args.historial, args.root)
        print(json.dumps(plan.summary(), ensure_ascii=False, indent=2), flush=True)
        if args.aplicar:
            report = apply_history(plan, args.project, args.dataset, args.location, max_bytes=args.max_bytes)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("Plan validado. No se contactó Google Cloud. Usa --aplicar para cargarlo.")
    except Exception as exc:
        print(f"Carga detenida: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
