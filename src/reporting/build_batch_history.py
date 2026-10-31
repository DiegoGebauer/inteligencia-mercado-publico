"""Construye un historial local verificable sin API, modelos ni entrenamiento.

Las entradas se leen sin modificarlas. Cada combinación de entradas y versión
de este constructor produce una carpeta inmutable; repetirla reutiliza esa
carpeta después de comprobar sus huellas.
"""
import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
CONFIRMED = {
    "revision_asistida_confirmada_por_usuario",
    "revision_humana_confirmada",
    "confirmada_por_usuario_sobre_texto_disponible",
}
CAPTURE_FIELDS = [
    "CapturaId", "CodigoExterno", "FechaCapturaUTC", "HashFuenteSHA256", "ArchivoOrigen",
    "Nombre", "Organismo", "Region", "FechaPublicacion", "FechaCierre", "EstadoCapturado",
]
PREDICTION_FIELDS = [
    "PrediccionId", "CapturaId", "CodigoExterno", "LoteId", "IntentoId", "FechaPrediccionUTC",
    "AfinidadPredicha", "AfinidadEsperada", "ProbAfinidad0", "ProbAfinidad1", "ProbAfinidad2", "ProbAfinidad3",
    "ModeloURI", "ModeloRunId", "HashModeloOriginalSHA256", "Encoder", "RevisionEncoderLocal",
    "VersionPerfil", "VersionCriterio", "HashTextoSHA256", "ArchivoPredicciones", "HashPrediccionesSHA256",
]
REVIEW_FIELDS = [
    "RevisionHistorialId", "RevisionId", "PrediccionId", "CapturaId", "CodigoExterno", "LoteId",
    "AfinidadPredicha", "AfinidadHumana", "EstadoRevision", "FechaRevisionLocal", "ZonaHoraria",
    "Motivo", "ConfirmacionUsuario", "BasesRevisadas", "AlcanceEvidencia", "IncorporadoEntrenamiento",
    "VersionCriterio", "ArchivoRevision", "HashRevisionSHA256",
]
TABLE_FIELDS = {
    "lotes.csv": ["LoteId", "EstadoLote", "FechaConsultada", "InicioUTC", "FinUTC", "ConsultasAPI", "DetallesCorrectos", "DetallesFallidos", "LoteOrigen", "ArchivoRegistro"],
    "capturas.csv": CAPTURE_FIELDS,
    "lote_captura.csv": ["LoteId", "CodigoExterno", "CapturaId", "ArchivoCaptura", "ReutilizadoDeLote"],
    "predicciones.csv": PREDICTION_FIELDS,
    "revisiones.csv": REVIEW_FIELDS,
    "vista_actual.csv": [
        "CodigoExterno", "Nombre", "Organismo", "Region", "FechaPublicacion", "FechaCierre", "EstadoCapturado",
        "CapturaId", "FechaCapturaUTC", "CapturasHistoricas", "PrediccionId", "FechaPrediccionUTC",
        "AfinidadPredicha", "AfinidadEsperada", "ModeloURI", "AfinidadHumana", "EstadoRevision",
        "FechaRevisionLocal", "MotivoRevision", "DifierePrediccionRevision", "RevisionesHistoricas",
    ],
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _identity(kind, *values):
    return _sha(json.dumps([kind, *values], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _utc(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Una fecha de captura o predicción no tiene zona horaria.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _code(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", value):
        raise ValueError("Hay un código de licitación inválido.")
    return value


def _class(value):
    if str(value) not in {"0", "1", "2", "3"}:
        raise ValueError("La afinidad debe ser un entero de 0 a 3.")
    return int(value)


class _Sources:
    def __init__(self, root):
        self.root = root
        self.contents = {}

    def read(self, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (self.root / path).resolve().is_relative_to(self.root):
            raise ValueError("Una ruta de entrada sale del proyecto.")
        name = path.as_posix()
        if name not in self.contents:
            self.contents[name] = (self.root / path).read_bytes()
        return self.contents[name]

    def sha(self, relative):
        return _sha(self.read(relative))

    def json(self, relative):
        return json.loads(self.read(relative).decode("utf-8-sig"))

    def csv(self, relative):
        reader = csv.DictReader(io.StringIO(self.read(relative).decode("utf-8-sig"), newline=""))
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"CSV sin encabezado válido: {relative}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"CSV con filas incompletas: {relative}")
        return rows

    def hashes(self):
        return {name: _sha(data) for name, data in sorted(self.contents.items())}

    def verify_unchanged(self):
        for name, data in self.contents.items():
            if not (self.root / name).is_file() or (self.root / name).read_bytes() != data:
                raise RuntimeError(f"Una entrada cambió durante la construcción: {name}")


def _inventory(root):
    return {
        "capturas": sorted(path.relative_to(root).as_posix() for path in (root / "data/raw/batches").glob("*/captura.json")),
        "predicciones": sorted(path.relative_to(root).as_posix() for path in (root / "data/processed/batches").glob("*/*/registro_ejecucion.json")),
        "revisiones": sorted(path.relative_to(root).as_posix() for path in (root / "data/processed/reviews").glob("*/*.csv")),
    }


def _captures(sources, names):
    lots, captures, references, by_lot = [], {}, [], {}
    manifests = {}
    for name in names:
        folder = Path(name).parent
        manifest = sources.json(name)
        lot_id = folder.name
        if manifest["lote_id"] != lot_id or manifest["estado"] not in {"completo", "incompleto", "vacio", "fallido"}:
            raise ValueError(f"Registro de captura inválido: {name}")
        manifests[lot_id] = (name, manifest)
        entries = manifest["detalles"]
        codes = [_code(entry["codigo"]) for entry in entries]
        if len(set(codes)) != len(codes):
            raise ValueError(f"Un lote repite códigos: {lot_id}")
        selected = manifest["seleccionados"]
        if len(set(selected)) != len(selected) or not set(codes).issubset(selected):
            raise ValueError(f"La selección y los detalles no coinciden: {lot_id}")
        if manifest["estado"] == "completo" and (not entries or codes != selected or any(entry["estado"] != "correcto" for entry in entries)):
            raise ValueError(f"El lote completo no contiene todos los detalles: {lot_id}")
        if manifest.get("listado"):
            listed = manifest["listado"]
            if listed["archivo"] != "listado.json" or sources.sha(folder / "listado.json") != listed["sha256"]:
                raise ValueError(f"Cambió el listado de {lot_id}.")
        lots.append({
            "LoteId": lot_id, "EstadoLote": manifest["estado"], "FechaConsultada": manifest["plan"]["fecha_iso"],
            "InicioUTC": _utc(manifest["inicio_utc"]), "FinUTC": _utc(manifest["fin_utc"]),
            "ConsultasAPI": manifest["consultas_api"], "DetallesCorrectos": sum(entry["estado"] == "correcto" for entry in entries),
            "DetallesFallidos": sum(entry["estado"] != "correcto" for entry in entries),
            "LoteOrigen": manifest.get("origen", {}).get("lote_id", ""), "ArchivoRegistro": name,
        })
        for entry in entries:
            if entry["estado"] != "correcto":
                continue
            code = entry["codigo"]
            expected = f"details/{code}.json"
            if entry["archivo"] != expected:
                raise ValueError("Un detalle tiene una ruta inesperada.")
            path = (folder / expected).as_posix()
            if sources.sha(path) != entry["sha256"]:
                raise ValueError(f"Cambió la captura de {code} en {lot_id}.")
            payload = sources.json(path)
            if not isinstance(payload.get("Listado"), list) or len(payload["Listado"]) != 1:
                raise ValueError(f"El detalle no contiene una licitación: {path}")
            row = payload["Listado"][0]
            if row.get("CodigoExterno") != code:
                raise ValueError("La identidad del detalle no coincide con su registro.")
            captured_at = _utc(entry["captura_utc"])
            capture_id = _identity("capture", code, captured_at, entry["sha256"])
            dates = row.get("Fechas") or {}
            buyer = row.get("Comprador") or {}
            captures.setdefault(capture_id, {
                "CapturaId": capture_id, "CodigoExterno": code, "FechaCapturaUTC": captured_at,
                "HashFuenteSHA256": entry["sha256"], "ArchivoOrigen": path, "Nombre": row.get("Nombre", ""),
                "Organismo": buyer.get("NombreOrganismo", ""), "Region": buyer.get("RegionUnidad", ""),
                "FechaPublicacion": dates.get("FechaPublicacion", ""), "FechaCierre": dates.get("FechaCierre", ""),
                "EstadoCapturado": row.get("Estado", ""),
            })
            by_lot[(lot_id, code)] = capture_id
            references.append({"LoteId": lot_id, "CodigoExterno": code, "CapturaId": capture_id,
                               "ArchivoCaptura": path, "ReutilizadoDeLote": entry.get("reutilizado_de_lote", "")})
    # Una recuperación enlaza con el manifiesto que realmente utilizó.
    for lot_id, (_, manifest) in manifests.items():
        origin = manifest.get("origen")
        if origin:
            if origin["lote_id"] not in manifests or sources.sha(manifests[origin["lote_id"]][0]) != origin["captura_sha256"]:
                raise ValueError(f"El origen de la recuperación cambió: {lot_id}")
    return lots, captures, references, by_lot, manifests


def _predictions(sources, names, captures, by_lot, manifests):
    predictions, index = [], {}
    incomplete = 0
    for name in names:
        path = Path(name)
        lot_id, attempt = path.parent.parent.name, path.parent.name
        log = sources.json(name)
        if log["estado"] != "completo":
            incomplete += 1
            continue
        if log["lote_id"] != lot_id or lot_id not in manifests:
            raise ValueError("Hay predicciones sin un lote válido.")
        manifest_path, capture_log = manifests[lot_id]
        if capture_log["estado"] != "completo" or sources.sha(manifest_path) != log["hash_captura_sha256"]:
            raise ValueError("La predicción no corresponde a la captura completa guardada.")
        csv_path = (path.parent / "predicciones.csv").as_posix()
        if sources.sha(csv_path) != log["predicciones_sha256"]:
            raise ValueError(f"Cambió el CSV de predicciones: {csv_path}")
        rows = sources.csv(csv_path)
        codes = [row["CodigoExterno"] for row in rows]
        if len(rows) != log["filas"] or len(set(codes)) != len(codes) or set(codes) != set(capture_log["seleccionados"]):
            raise ValueError("Las filas de predicción no coinciden con el lote.")
        for row in rows:
            code = _code(row["CodigoExterno"])
            capture_id = by_lot[(lot_id, code)]
            capture = captures[capture_id]
            expected_source = f"data/raw/batches/{lot_id}/details/{code}.json"
            if (row["LoteId"] != lot_id or row["OrigenArchivo"] != expected_source
                    or row["HashFuenteSHA256"] != capture["HashFuenteSHA256"]
                    or _utc(row["FechaCapturaUTC"]) != capture["FechaCapturaUTC"]
                    or _sha(row["TextoLicitacion"].encode("utf-8")) != row["HashTextoSHA256"]):
                raise ValueError("La predicción perdió la trazabilidad a su texto o captura.")
            for key in ["ModeloURI", "ModeloRunId", "HashModeloOriginalSHA256", "Encoder", "RevisionEncoderLocal", "VersionPerfil", "VersionCriterio"]:
                if not row[key] or row[key] != log["identidad_modelo"][key]:
                    raise ValueError(f"La identidad de la predicción difiere en {key}.")
            probabilities = [float(row[f"ProbAfinidad{k}"]) for k in range(4)]
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
                raise ValueError("Hay probabilidades inválidas.")
            expected = float(row["AfinidadEsperada"])
            if not math.isfinite(expected) or not math.isclose(expected, sum(k * p for k, p in enumerate(probabilities)), abs_tol=1e-6):
                raise ValueError("La afinidad esperada no coincide con las probabilidades.")
            prediction = {key: row.get(key, "") for key in PREDICTION_FIELDS}
            prediction.update(
                PrediccionId=_identity("prediction", lot_id, attempt, code), CapturaId=capture_id,
                AfinidadPredicha=_class(row["AfinidadPredicha"]), IntentoId=attempt,
                FechaPrediccionUTC=_utc(row["FechaPrediccionUTC"]), ArchivoPredicciones=csv_path,
                HashPrediccionesSHA256=log["predicciones_sha256"],
            )
            predictions.append(prediction)
            index[(csv_path, code)] = (prediction, row)
    return predictions, index, incomplete


def _reviews(sources, names, prediction_index):
    reviews = {}
    for name in names:
        for row in sources.csv(name):
            key = (row["OrigenPredicciones"], row["CodigoExterno"])
            if key not in prediction_index:
                raise ValueError("Una revisión no corresponde a una predicción guardada.")
            prediction, original = prediction_index[key]
            for field in ["LoteId", "ModeloURI", "VersionCriterio", "HashModeloOriginalSHA256", "HashFuenteSHA256", "HashTextoSHA256", "OrigenArchivo"]:
                if row[field] != original[field]:
                    raise ValueError(f"La revisión difiere de la predicción en {field}.")
            if row["HashPrediccionesSHA256"] != prediction["HashPrediccionesSHA256"] or _class(row["AfinidadPredicha"]) != prediction["AfinidadPredicha"]:
                raise ValueError("La revisión cambió la referencia a la predicción original.")
            if row.get("HashCriterioSHA256"):
                criterion = Path(row["OrigenCriterio"])
                if criterion.parts[:2] != ("data", "labeled") or criterion.name != "criterio.md" or sources.sha(criterion) != row["HashCriterioSHA256"]:
                    raise ValueError("La revisión no conserva la referencia al criterio.")
            human = row["AfinidadHumana"]
            if human != "":
                human = _class(human)
                if row["EstadoRevision"] not in CONFIRMED or not row.get("ConfirmacionUsuario", "").strip():
                    raise ValueError("Una afinidad humana carece de confirmación explícita.")
            date.fromisoformat(row["FechaRevisionLocal"])
            if not row["RevisionId"]:
                raise ValueError("La revisión no tiene identidad.")
            review_id = _identity("review", prediction["PrediccionId"], row["RevisionId"])
            record = {field: row.get(field, "") for field in REVIEW_FIELDS}
            record.update(RevisionHistorialId=review_id, PrediccionId=prediction["PrediccionId"], CapturaId=prediction["CapturaId"],
                          AfinidadPredicha=prediction["AfinidadPredicha"], AfinidadHumana=human,
                          ArchivoRevision=name, HashRevisionSHA256=sources.sha(name))
            if review_id in reviews:
                previous = reviews[review_id]
                comparable = [field for field in REVIEW_FIELDS if field not in {"ArchivoRevision", "HashRevisionSHA256"}]
                if any(previous[field] != record[field] for field in comparable):
                    raise ValueError("Dos archivos asignan valores distintos a la misma revisión.")
            else:
                reviews[review_id] = record
    return list(reviews.values())


def _current(captures, predictions, reviews):
    codes = sorted({row["CodigoExterno"] for row in captures.values()})
    result = []
    for code in codes:
        history = [row for row in captures.values() if row["CodigoExterno"] == code]
        capture = max(history, key=lambda row: (row["FechaCapturaUTC"], row["CapturaId"]))
        available = [row for row in predictions if row["CapturaId"] == capture["CapturaId"]]
        prediction = max(available, key=lambda row: (row["FechaPrediccionUTC"], row["PrediccionId"])) if available else None
        linked = [row for row in reviews if prediction and row["PrediccionId"] == prediction["PrediccionId"] and row["AfinidadHumana"] != ""]
        review = max(linked, key=lambda row: (row["FechaRevisionLocal"], row["RevisionId"])) if linked else None
        row = {field: capture.get(field, "") for field in TABLE_FIELDS["vista_actual.csv"]}
        row.update(CapturasHistoricas=len(history), RevisionesHistoricas=sum(item["CodigoExterno"] == code and item["AfinidadHumana"] != "" for item in reviews),
                   EstadoRevision="pendiente_revision_humana" if prediction else "sin_prediccion_en_ultima_captura")
        if prediction:
            row.update({field: prediction[field] for field in ["PrediccionId", "FechaPrediccionUTC", "AfinidadPredicha", "AfinidadEsperada", "ModeloURI"]})
        if review:
            row.update(AfinidadHumana=review["AfinidadHumana"], EstadoRevision=review["EstadoRevision"],
                       FechaRevisionLocal=review["FechaRevisionLocal"], MotivoRevision=review["Motivo"],
                       DifierePrediccionRevision=review["AfinidadHumana"] != prediction["AfinidadPredicha"])
        result.append(row)
    return result


def _verify_snapshot(folder, fingerprint, input_hashes, builder_hash):
    manifest_path = folder / "resumen.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Hay una salida incompleta o una ejecución en curso: {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["huella_entradas"] != fingerprint or manifest["entradas_sha256"] != input_hashes or manifest["constructor_sha256"] != builder_hash:
        raise RuntimeError("La carpeta existente no coincide con las entradas y constructor actuales.")
    if set(manifest["salidas_sha256"]) != set(TABLE_FIELDS):
        raise RuntimeError("El historial existente no contiene las tablas esperadas.")
    for name in TABLE_FIELDS:
        path = folder / name
        if not path.is_file() or _sha(path.read_bytes()) != manifest["salidas_sha256"][name]:
            raise RuntimeError(f"Cambió una salida del historial: {name}")
    return manifest


def build_history(root: Path = ROOT, output_base: Path | None = None) -> Path:
    """Lee lotes y devuelve una carpeta histórica verificada. No modifica fuentes."""
    root = Path(root).resolve()
    default_output = root / "data/processed/history"
    output_base = Path(output_base).resolve() if output_base is not None else default_output
    if output_base.is_relative_to(root) and not output_base.is_relative_to(default_output):
        raise ValueError("Dentro del proyecto, la salida debe quedar en data/processed/history.")
    inventory = _inventory(root)
    sources = _Sources(root)
    lots, captures, references, by_lot, manifests = _captures(sources, inventory["capturas"])
    predictions, prediction_index, incomplete = _predictions(sources, inventory["predicciones"], captures, by_lot, manifests)
    reviews = _reviews(sources, inventory["revisiones"], prediction_index)
    current = _current(captures, predictions, reviews)
    tables = {
        "lotes.csv": lots, "capturas.csv": sorted(captures.values(), key=lambda row: (row["CodigoExterno"], row["FechaCapturaUTC"], row["CapturaId"])),
        "lote_captura.csv": references, "predicciones.csv": predictions, "revisiones.csv": reviews, "vista_actual.csv": current,
    }
    sources.verify_unchanged()
    if _inventory(root) != inventory:
        raise RuntimeError("Aparecieron o desaparecieron entradas durante la construcción; repite la ejecución.")
    inputs = sources.hashes()
    builder_hash = _sha(Path(__file__).read_bytes())
    fingerprint = _identity("history", SCHEMA_VERSION, builder_hash, inputs)
    folder = output_base / fingerprint[:20]
    if folder.exists():
        _verify_snapshot(folder, fingerprint, inputs, builder_hash)
        return folder
    folder.mkdir(parents=True, exist_ok=False)
    for name, fields in TABLE_FIELDS.items():
        with (folder / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(tables[name])
    summary = {
        "schema_version": SCHEMA_VERSION, "creacion_utc": datetime.now(timezone.utc).isoformat(),
        "huella_entradas": fingerprint, "constructor_sha256": builder_hash,
        "entradas_sha256": inputs, "salidas_sha256": {name: _sha((folder / name).read_bytes()) for name in TABLE_FIELDS},
        "conteos": {name.removesuffix(".csv"): len(rows) for name, rows in tables.items()},
        "capturas_reutilizadas_sin_duplicar": len(references) - len(captures),
        "intentos_prediccion_incompletos": incomplete,
        "licitaciones_sin_prediccion_actual": sum(not row["PrediccionId"] for row in current),
        "revisiones_humanas_confirmadas": sum(row["AfinidadHumana"] != "" for row in reviews),
        "consultas_api": 0, "modelos_cargados": 0,
        "nota": "Índice derivado. No modifica capturas, predicciones, etiquetas ni modelos. Las revisiones sólo se aplican a la predicción y captura que referencian. No calcula métricas de calidad.",
    }
    with (folder / "resumen.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    _verify_snapshot(folder, fingerprint, inputs, builder_hash)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    folder = build_history(args.root, args.output_dir)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    print("Historial verificado:", folder)
    print(json.dumps(summary["conteos"], ensure_ascii=False, indent=2))
    print("Vista de seguimiento:", folder / "vista_actual.csv")


if __name__ == "__main__":
    main()
