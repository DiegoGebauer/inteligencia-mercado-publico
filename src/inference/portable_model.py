"""Usa el paquete aprobado desde un archivo local, sin registro MLflow ni cloud.

El descriptor de GCS identifica la procedencia. La descarga es una etapa aparte;
este módulo sólo abre archivos locales y nunca sigue una URI gs://.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from uuid import uuid4

from src.experiments import export_model_bundle as bundle


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_VARIABLE = "MP_MODEL_BUNDLE"
DESCRIPTOR = "config/cloud/model_bundle.json"
PACKAGES = ("scikit-learn", "numpy", "pandas", "sentence-transformers", "transformers", "torch")
RUNTIME_FILES = (*bundle.CODE_FILES, "src/ingestion/recent_batch.py", "src/ingestion/capture_store.py",
                 "src/inference/portable_model.py", "src/experiments/export_model_bundle.py",
                 "config/company_profile.yaml", DESCRIPTOR)
REPRESENTATION = {"prefix": "passage: ", "chunk_tokens": 400, "overlap_tokens": 40,
                  "pooling": "mean_normalized_fragments", "target": "Afinidad", "classes": [0, 1, 2, 3]}


def configured_archive(root=ROOT):
    value = os.environ.get(ENVIRONMENT_VARIABLE)
    if value is None:
        return None
    if not value.strip() or "://" in value:
        raise ValueError("MP_MODEL_BUNDLE debe indicar un archivo local, no una URI ni un valor vacío.")
    path = Path(value)
    return (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def source_hashes(root):
    return {name: bundle.sha_file(Path(root) / name) for name in RUNTIME_FILES}


def inspect_bundle(path, root=ROOT):
    """Valida el archivo y el contrato sin escribir, cargar joblib o importar ML."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    descriptor_bytes = (root / DESCRIPTOR).read_bytes()
    descriptor = json.loads(descriptor_bytes.decode("utf-8-sig"))
    if descriptor.get("schema_version") != 1:
        raise ValueError("Versión de referencia del modelo no admitida.")
    pinned = descriptor["bundle"]
    if (not re.fullmatch(r"[a-f0-9]{64}", str(pinned.get("id", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(pinned.get("sha256", "")))
            or not re.fullmatch(r"[1-9][0-9]*", str(pinned.get("generation", "")))
            or not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+/[^?#\s]+", str(pinned.get("gcs_uri", "")))
            or type(pinned.get("bytes")) is not int or not 0 < pinned["bytes"] <= 1024**3
            or pinned.get("format") != "tar.gz"):
        raise ValueError("La referencia GCS necesita generación, tamaño y SHA256 válidos.")
    if path.stat().st_size != pinned["bytes"] or bundle.sha_file(path) != pinned["sha256"]:
        raise ValueError("El archivo local difiere del tamaño o SHA256 fijado en la referencia.")
    manifest = bundle.verify_archive(path, expected_id=pinned["id"])
    if manifest["model"] != descriptor["model"] or manifest["encoder"] != descriptor["encoder"]:
        raise ValueError("El modelo o encoder no coincide con la referencia versionada.")
    if manifest.get("representation") != REPRESENTATION:
        raise ValueError("El paquete utiliza otra representación de los textos.")
    if set(manifest["inference_code_sha256"]) != set(bundle.CODE_FILES):
        raise ValueError("El paquete no fija los cinco módulos de inferencia esperados.")
    before = source_hashes(root)
    if before[DESCRIPTOR] != hashlib.sha256(descriptor_bytes).hexdigest():
        raise RuntimeError("La referencia del paquete cambió mientras se validaba.")
    for name, expected in manifest["inference_code_sha256"].items():
        if before[name] != expected:
            raise ValueError(f"Cambió el código de inferencia de referencia: {name}")
    if before["config/company_profile.yaml"] != manifest["files"]["company_profile.yaml"]["sha256"]:
        raise ValueError("El perfil de empresa difiere del paquete aprobado.")
    return {"archive": path, "descriptor": descriptor, "manifest": manifest, "sources": before}


def check_environment(manifest):
    expected = manifest["reference_environment"]["packages"]
    for name in PACKAGES:
        if version(name) != expected[name]:
            raise RuntimeError(f"La versión de {name} debe coincidir con la referencia: {expected[name]}.")


@contextmanager
def offline_environment():
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class Runtime:
    classifier: object
    encoder: object
    saved: dict
    manifest: dict
    lineage: dict
    sources: dict

    def close(self):
        self.classifier = self.encoder = self.saved = None
        gc.collect()


def _load_local(directory, manifest):
    import joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer

    saved = joblib.load(directory / "classifier.joblib")
    if (saved.get("embedding_model") != manifest["encoder"]["name"] or saved.get("target") != "Afinidad"
            or saved.get("pooling") != "mean_normalized_fragments"
            or saved.get("profile_version") != manifest["profile_version"]
            or saved.get("labeling_policy_version") != manifest["labeling_policy_version"]):
        raise ValueError("Los metadatos del clasificador no coinciden con el paquete.")
    classifier = saved["classifier"]
    np.testing.assert_array_equal(classifier.classes_, [0, 1, 2, 3])
    if (classifier.n_features_in_ != 384 or classifier.get_params()["logisticregression__C"] != 0.1
            or [type(step).__name__ for _, step in classifier.steps] != ["StandardScaler", "LogisticRegression"]):
        raise ValueError("El clasificador no cumple el contrato del candidato aprobado.")
    encoder = SentenceTransformer(str(directory / "encoder"), device="cpu", local_files_only=True)
    if encoder.get_embedding_dimension() != 384 or not 0 < encoder.max_seq_length <= 512:
        raise ValueError("El encoder no cumple las dimensiones o límite de tokens.")
    return classifier, encoder, saved


@contextmanager
def loaded_model(path, root=ROOT):
    root = Path(root).resolve()
    plan = inspect_bundle(path, root)
    manifest = plan["manifest"]
    check_environment(manifest)
    base = root / "data/processed/deployment"
    parent = next(p for p in (base, *base.parents) if p.exists())
    required = plan["archive"].stat().st_size + sum(item["bytes"] for item in manifest["files"].values()) + 64 * 1024**2
    if shutil.disk_usage(parent).free < required:
        raise OSError("Falta espacio para preparar temporalmente el modelo portable.")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-modelo-", dir=base) as temporary:
        directory = Path(temporary).resolve()
        if directory.parent != base.resolve():
            raise RuntimeError("La carpeta temporal no está en la ubicación esperada.")
        private = directory / "paquete.tar.gz"
        with plan["archive"].open("rb") as source, private.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if bundle.sha_file(private) != plan["descriptor"]["bundle"]["sha256"]:
            raise ValueError("El archivo cambió al preparar el modelo.")
        bundle.verify_archive(private, expected_id=manifest["bundle_id"])
        with tarfile.open(private, "r:gz") as archive:
            for name in manifest["files"]:
                target = directory.joinpath(*PurePosixPath(name).parts)
                if not target.resolve().is_relative_to(directory):
                    raise ValueError("Un miembro sale de la carpeta temporal.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            for name in manifest["directories"]:
                directory.joinpath(*PurePosixPath(name).parts).mkdir(parents=True, exist_ok=True)
        for name, expected in manifest["files"].items():
            if bundle.sha_file(directory / name) != expected["sha256"]:
                raise ValueError("Un archivo extraído perdió su integridad.")
        if source_hashes(root) != plan["sources"]:
            raise RuntimeError("Cambió el código o la configuración antes de cargar el modelo.")
        runtime = None
        with offline_environment():
            try:
                classifier, encoder, saved = _load_local(directory, manifest)
                runtime = Runtime(classifier, encoder, saved, manifest, dict(plan["descriptor"]["bundle"]), plan["sources"])
                # Evita conservar referencias adicionales que impidan liberar los pesos en Windows.
                del classifier, encoder, saved
                yield runtime
            finally:
                if runtime is not None:
                    runtime.close()


def _predict_frame(frame, runtime):
    import numpy as np
    from src.nlp.rank_tenders import split_text
    from src.nlp.train_affinity import encode_tenders
    from src.nlp.predict_affinity_pilot import build_predictions

    for text in frame["TextoLicitacion"]:
        chunks = split_text(text, runtime.encoder.tokenizer)
        if not chunks or any(len(runtime.encoder.tokenizer.encode("passage: " + chunk, add_special_tokens=True,
                                                                  truncation=False)) > min(512, runtime.encoder.max_seq_length)
                             for chunk in chunks):
            raise ValueError("Hay fragmentos que exceden el límite del encoder.")
    embeddings = encode_tenders(frame["TextoLicitacion"].tolist(), runtime.encoder)
    if (embeddings.shape != (len(frame), 384) or embeddings.dtype != np.float32
            or not np.isfinite(embeddings).all()
            or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)):
        raise ValueError("La matriz de embeddings no cumple el contrato.")
    return build_predictions(frame, runtime.classifier, embeddings)


def predict_batch(folder, *, root=ROOT, archive):
    """Nueva inferencia de un lote completo; no captura ni publica datos."""
    from src.ingestion import recent_batch as batch
    from src.ingestion.capture_store import guardar_captura

    root, folder = Path(root).resolve(), Path(folder).resolve()
    if folder.parent != (root / "data/raw/batches").resolve():
        raise ValueError("El lote debe estar directamente en data/raw/batches.")
    capture, records, sources = batch._load_capture(folder)
    output_base = root / "data/processed/batches" / folder.name
    for previous in output_base.glob("*/registro_ejecucion.json"):
        if batch._read(previous).get("estado") == "completo":
            raise FileExistsError(f"El lote ya tiene una predicción completa: {previous.parent}")
    capture_hash = bundle.sha_file(folder / "captura.json")
    with loaded_model(archive, root) as runtime:
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
        output = output_base / attempt
        output.mkdir(parents=True, exist_ok=False)
        log = {"schema_version": 1, "lote_id": folder.name, "inicio_utc": _utc(), "estado": "fallido",
               "hash_captura_sha256": capture_hash, "consultas_api_en_prediccion": 0,
               "origen_carga_modelo": "paquete_portable", "paquete_modelo": runtime.lineage}
        try:
            frame = batch._prepare(records)
            result = _predict_frame(frame, runtime)
            result["LoteId"] = folder.name
            result["OrigenArchivo"] = result["CodigoExterno"].map(lambda code: (folder / sources[code]["archivo"]).relative_to(root).as_posix())
            result["HashFuenteSHA256"] = result["CodigoExterno"].map(lambda code: sources[code]["sha256"])
            result["FechaCapturaUTC"] = result["CodigoExterno"].map(lambda code: sources[code]["captura_utc"])
            result["HashTextoSHA256"] = result["TextoLicitacion"].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
            manifest = runtime.manifest
            identity = {"ModeloAlias": "version_fija_portable", "ModeloVersion": manifest["model"]["version"],
                        "ModeloURI": manifest["model"]["uri"], "ModeloRunId": manifest["model"]["run_id"],
                        "HashModeloOriginalSHA256": manifest["model"]["sha256"], "Encoder": manifest["encoder"]["name"],
                        "RevisionEncoderLocal": manifest["encoder"]["revision"], "VersionPerfil": manifest["profile_version"],
                        "VersionCriterio": manifest["labeling_policy_version"], "FechaPrediccionUTC": _utc(),
                        "EstadoRevision": "prediccion_pendiente_de_revision_humana"}
            for name, value in identity.items():
                result[name] = value
            if source_hashes(root) != runtime.sources or bundle.sha_file(archive) != runtime.lineage["sha256"]:
                raise RuntimeError("Cambió un archivo de código, configuración o modelo durante la inferencia.")
            batch._load_capture(folder)
            if bundle.sha_file(folder / "captura.json") != capture_hash:
                raise RuntimeError("El registro de captura cambió durante la predicción.")
            csv_path = output / "predicciones.csv"
            with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
                result.to_csv(stream, index=False)
            log.update(estado="completo", filas=len(result), identidad_modelo=identity,
                       captura=folder.relative_to(root).as_posix(), predicciones_sha256=bundle.sha_file(csv_path),
                       distribucion_predicha={str(key): int(count) for key, count in result["AfinidadPredicha"].value_counts().sort_index().items()},
                       originales_verificados_sha256=runtime.sources,
                       versiones={name: version(name) for name in (*PACKAGES, "joblib")}, python=sys.version.split()[0],
                       seleccion=capture["plan"], nota="Inferencia CPU desde paquete aprobado; sin MLflow ni consultas cloud. Las probabilidades no estiman adjudicación. No es una evaluación de calidad.")
        except Exception as error:
            log["tipo_error"] = type(error).__name__
            raise
        finally:
            log["fin_utc"] = _utc()
            guardar_captura(log, output / "registro_ejecucion.json")
    print("Predicción portable guardada:", csv_path)
    print("Registro:", output / "registro_ejecucion.json")
    return csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    plan = inspect_bundle(args.archivo, args.root)
    print(json.dumps({"modelo": plan["manifest"]["model"], "paquete": plan["descriptor"]["bundle"],
                      "modo": "portable_local", "mlflow_requerido": False}, ensure_ascii=False, indent=2))
    print("Sólo validación: no se escribieron archivos, no se cargó el modelo ni se contactó la API o Google Cloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
