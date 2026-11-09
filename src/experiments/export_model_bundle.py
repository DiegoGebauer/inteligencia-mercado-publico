"""Exporta el candidato acordado y E5 como archivo portátil; sin API ni inferencia.

Sin --crear sólo valida fuentes y muestra el plan. No incluye .env, credenciales,
MLflow SQLite ni datos de entrenamiento. La equivalencia numérica se comprueba
por separado con verify_model_bundle.
"""
import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "mercado_publico_afinidad_embeddings"
MODEL_SHA = "371f8222a53f0fa355a3b19826174daa8772bc42c6b87312b95ad4856918185c"
ENCODER = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
REFERENCE_LOT = "20261016T030234991480Z_5b2b8bbd"
ENCODER_FILES = (
    "config.json", "model.safetensors", "modules.json", "README.md",
    "sentence_bert_config.json", "sentencepiece.bpe.model", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json", "1_Pooling/config.json",
)
CODE_FILES = (
    "src/nlp/rank_tenders.py", "src/nlp/train_affinity.py",
    "src/nlp/predict_affinity_pilot.py", "src/nlp/build_tender_texts.py",
    "src/processing/normalize_tenders.py",
)
DIRECTORIES = ("encoder/1_Pooling", "encoder/2_Normalize")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def default_encoder():
    return (Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots" / REVISION)


@dataclass(frozen=True)
class Payload:
    name: str
    size: int
    sha256: str
    source: Path | None = None
    data: bytes | None = None


def payload_file(name, source):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Falta un archivo requerido: {source}")
    return Payload(name, source.stat().st_size, sha_file(source), source=source)


def payload_bytes(name, data):
    return Payload(name, len(data), hashlib.sha256(data).hexdigest(), data=data)


def expected_reference_hash(log, relative):
    hashes = log["originales_verificados_sha256"]
    if relative in hashes:
        return hashes[relative]
    matches = [value for key, value in hashes.items() if key.endswith("/baseline/" + relative)]
    if len(matches) != 1:
        raise ValueError(f"No hay una huella única de referencia para {relative}.")
    return matches[0]


def build_plan(root=ROOT, encoder_dir=None):
    root = Path(root).resolve()
    encoder_dir = Path(encoder_dir or default_encoder()).resolve()
    receipt = read_json(root / "data/processed/affinity_mlflow_baseline_registration.json")
    if (receipt["registered_model"] != MODEL_NAME or str(receipt["model_version"]) != "1"
            or receipt["model_sha256"] != MODEL_SHA):
        raise ValueError("El comprobante no corresponde al candidato v1 acordado.")
    classifier = payload_file("classifier.joblib", root / "models/affinity_aligned_299.joblib")
    if classifier.sha256 != MODEL_SHA:
        raise ValueError("El clasificador cambió respecto de la referencia acordada.")
    reference_base = root / "data/processed/batches" / REFERENCE_LOT
    completed = [(path, read_json(path)) for path in reference_base.glob("*/registro_ejecucion.json")]
    completed = [(path, log) for path, log in completed if log.get("estado") == "completo"]
    if len(completed) != 1:
        raise ValueError("Se necesita una única predicción histórica completa del lote de referencia.")
    log_path, log = completed[0]
    identity = log["identidad_modelo"]
    if (identity["HashModeloOriginalSHA256"] != MODEL_SHA or identity["ModeloRunId"] != receipt["run_id"]
            or identity["ModeloURI"] != receipt["model_uri"] or identity["Encoder"] != ENCODER
            or identity["RevisionEncoderLocal"] != REVISION):
        raise ValueError("La referencia no utiliza el mismo modelo y revisión del encoder.")
    csv_path = log_path.parent / "predicciones.csv"
    if sha_file(csv_path) != log["predicciones_sha256"]:
        raise ValueError("El CSV de referencia no conserva su huella.")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5 or log["filas"] != 5 or len({row["CodigoExterno"] for row in rows}) != 5:
        raise ValueError("La referencia debe contener cinco licitaciones únicas.")
    cases = []
    for row in rows:
        text = row["TextoLicitacion"]
        if not text.strip() or hashlib.sha256(text.encode("utf-8")).hexdigest() != row["HashTextoSHA256"]:
            raise ValueError("El texto de referencia perdió su integridad.")
        probabilities = [float(row[f"ProbAfinidad{i}"]) for i in range(4)]
        if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)
                or row["AfinidadPredicha"] not in {"0", "1", "2", "3"}):
            raise ValueError("La referencia contiene resultados inválidos.")
        cases.append({"codigo": row["CodigoExterno"], "texto": text,
                      "clase": int(row["AfinidadPredicha"]), "probabilidades": probabilities})
    payloads = [classifier, payload_bytes("regression_cases.json", canonical({"cases": cases}))]
    for relative, target in [("config/company_profile.yaml", "company_profile.yaml"),
                             ("data/labeled/temporal_aligned/criterio.md", "labeling_policy.md")]:
        item = payload_file(target, root / relative)
        if item.sha256 != expected_reference_hash(log, relative):
            raise ValueError(f"Cambió una configuración de referencia: {relative}")
        payloads.append(item)
    code_hashes = {}
    for relative in CODE_FILES:
        digest = sha_file(root / relative)
        if digest != expected_reference_hash(log, relative):
            raise ValueError(f"Cambió el código de inferencia de referencia: {relative}")
        code_hashes[relative] = digest
    config = read_json(encoder_dir / "config.json")
    if config.get("hidden_size") != 384 or config.get("max_position_embeddings") != 512:
        raise ValueError("El encoder no cumple la dimensión y límite acordados.")
    modules = read_json(encoder_dir / "modules.json")
    if [(m["path"], m["type"]) for m in modules] != [
        ("", "sentence_transformers.models.Transformer"),
        ("1_Pooling", "sentence_transformers.models.Pooling"),
        ("2_Normalize", "sentence_transformers.models.Normalize"),
    ]:
        raise ValueError("La estructura de E5 no coincide con la revisión acordada.")
    for relative in ENCODER_FILES:
        payloads.append(payload_file("encoder/" + relative, encoder_dir / relative))
    packages = log["versiones"]
    required = {"numpy", "scikit-learn", "pandas", "sentence-transformers", "transformers", "torch"}
    if not required.issubset(packages) or any(not packages[key] for key in required):
        raise ValueError("Faltan versiones del entorno de referencia.")
    identity_data = {
        "schema_version": 1,
        "model": {"name": MODEL_NAME, "version": "1", "uri": receipt["model_uri"],
                  "run_id": receipt["run_id"], "sha256": MODEL_SHA},
        "encoder": {"name": ENCODER, "revision": REVISION, "dimension": 384, "max_tokens": 512},
        "representation": {"prefix": "passage: ", "chunk_tokens": 400, "overlap_tokens": 40,
                           "pooling": "mean_normalized_fragments", "target": "Afinidad", "classes": [0, 1, 2, 3]},
        "profile_version": identity["VersionPerfil"], "labeling_policy_version": identity["VersionCriterio"],
        "reference": {"lot": REFERENCE_LOT, "csv_sha256": log["predicciones_sha256"],
                      "log_sha256": sha_file(log_path), "cases": 5,
                      "purpose": "regression_only_not_quality_evaluation"},
        "reference_environment": {"python": log["python"], "packages": packages},
        "inference_code_sha256": code_hashes,
        "directories": list(DIRECTORIES),
        "files": {item.name: {"sha256": item.sha256, "bytes": item.size} for item in payloads},
    }
    fingerprint = hashlib.sha256(canonical(identity_data)).hexdigest()
    manifest = dict(identity_data, bundle_id=fingerprint)
    name = f"modelo-v1-{fingerprint[:16]}.tar.gz"
    return {"manifest": manifest, "payloads": payloads,
            "archive": root / "data/processed/deployment" / name,
            "bytes": sum(item.size for item in payloads)}


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or ":" in name or path.as_posix() != name):
        raise ValueError("Ruta no admitida en el paquete.")


def verify_archive(path, expected_id=None):
    """Valida miembros y bytes sin extraer ni deserializar el clasificador."""
    with tarfile.open(path, "r:gz") as archive:
        members = []
        for member in archive:
            if len(members) >= 64:
                raise ValueError("El paquete contiene demasiados miembros.")
            if member.isdir():
                member.name = member.name.rstrip("/")
            _safe_name(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError("El paquete contiene enlaces u otro tipo no admitido.")
            members.append(member)
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members) or "manifest.json" not in by_name:
            raise ValueError("Miembros duplicados o manifiesto ausente.")
        member = by_name["manifest.json"]
        if not member.isfile() or member.size > 1024 * 1024:
            raise ValueError("Manifiesto inválido.")
        manifest = json.loads(archive.extractfile(member).read().decode("utf-8"))
        identity = {key: value for key, value in manifest.items() if key != "bundle_id"}
        fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
        if (manifest.get("schema_version") != 1 or manifest.get("bundle_id") != fingerprint
                or (expected_id is not None and expected_id != fingerprint)
                or manifest["model"]["name"] != MODEL_NAME or manifest["model"]["version"] != "1"
                or manifest["model"]["sha256"] != MODEL_SHA
                or manifest["encoder"]["name"] != ENCODER or manifest["encoder"]["revision"] != REVISION
                or manifest["encoder"]["dimension"] != 384 or manifest["encoder"]["max_tokens"] != 512):
            raise ValueError("La identidad del paquete no coincide.")
        expected_files = {"classifier.joblib", "regression_cases.json", "company_profile.yaml", "labeling_policy.md"}
        expected_files.update("encoder/" + name for name in ENCODER_FILES)
        if set(manifest["files"]) != expected_files or manifest["directories"] != list(DIRECTORIES):
            raise ValueError("El manifiesto incluye archivos no previstos.")
        if set(by_name) != expected_files | {"manifest.json"} | set(DIRECTORIES):
            raise ValueError("Faltan o sobran miembros en el paquete.")
        if any(not by_name[name].isdir() for name in DIRECTORIES):
            raise ValueError("Directorio de módulo inválido.")
        for name, spec in manifest["files"].items():
            member = by_name[name]
            if not member.isfile() or member.size != spec["bytes"] or not 0 <= member.size <= 1024**3:
                raise ValueError(f"Tamaño de archivo inválido: {name}")
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"La huella de un archivo no coincide: {name}")
        if manifest["files"]["classifier.joblib"]["sha256"] != MODEL_SHA:
            raise ValueError("El clasificador no es el aprobado.")
    return manifest


def create_archive(plan):
    target = plan["archive"]
    if target.exists():
        verify_archive(target, plan["manifest"]["bundle_id"])
        return target
    existing = next(parent for parent in [target.parent, *target.parents] if parent.exists())
    if shutil.disk_usage(existing).free < 2 * plan["bytes"] + 64 * 1024**2:
        raise OSError("Falta espacio local para preparar el paquete.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=3) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for name in DIRECTORIES:
                        info = tarfile.TarInfo(name)
                        info.type, info.mode = tarfile.DIRTYPE, 0o755
                        archive.addfile(info)
                    manifest_bytes = canonical(plan["manifest"])
                    items = [payload_bytes("manifest.json", manifest_bytes), *plan["payloads"]]
                    for item in sorted(items, key=lambda value: value.name):
                        info = tarfile.TarInfo(item.name)
                        info.size, info.mode = item.size, 0o644
                        stream = item.source.open("rb") if item.source else io.BytesIO(item.data)
                        with stream:
                            archive.addfile(info, stream)
        verify_archive(temporary, plan["manifest"]["bundle_id"])
        # Apertura exclusiva: tampoco requiere enlaces duros en el disco D:.
        try:
            destination = target.open("xb")
        except FileExistsError:
            verify_archive(target, plan["manifest"]["bundle_id"])
        else:
            try:
                with destination, temporary.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                verify_archive(target, plan["manifest"]["bundle_id"])
            except BaseException:
                # Sólo se retira el archivo que acaba de crear esta operación.
                target.unlink(missing_ok=True)
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, default=None)
    parser.add_argument("--crear", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(ROOT, args.encoder_dir)
    print(json.dumps({"modelo": plan["manifest"]["model"], "encoder": plan["manifest"]["encoder"],
                      "archivos": len(plan["payloads"]), "mib_sin_comprimir": round(plan["bytes"] / 1024**2, 2),
                      "casos_referencia": 5, "bundle_id": plan["manifest"]["bundle_id"],
                      "archivo_destino": str(plan["archive"])}, ensure_ascii=False, indent=2))
    if not args.crear:
        print("Sólo plan: no se crearon archivos, no se cargó el modelo ni se contactó cloud.")
        return 0
    path = create_archive(plan)
    print("Paquete creado y verificado:", path)
    print("SHA256 del archivo:", sha_file(path))
    print("MiB comprimidos:", round(path.stat().st_size / 1024**2, 2))
    print("La equivalencia numérica se comprueba con src.experiments.verify_model_bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
