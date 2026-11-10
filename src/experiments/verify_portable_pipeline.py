"""Ejercita el pipeline portable contra el lote histórico en una carpeta aislada.

No consulta Mercado Público, MLflow o BigQuery. No modifica el historial real.
Carga E5 y el clasificador en CPU; requiere el paquete ya disponible localmente.
"""
import argparse
import json
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

from src.experiments import export_model_bundle as bundle
from src.inference import portable_model as portable
from src.ingestion import recent_batch as batch
from src.reporting import build_batch_history as history


ROOT = Path(__file__).resolve().parents[2]


def reference_capture_inputs(root, lot_id):
    """Lee el lote y todos sus orígenes; conserva los bytes y valida sus enlaces."""
    root = Path(root).resolve()
    base = (root / "data/raw/batches").resolve()
    sources = history._Sources(root)
    names, visited, visiting = [], set(), set()

    def visit(identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Identificador de lote de referencia inválido.")
        if identifier in visiting:
            raise ValueError("La cadena de recuperación contiene un ciclo.")
        if identifier in visited:
            return
        if len(visited) + len(visiting) >= 100:
            raise ValueError("La cadena de recuperación supera el límite de la prueba.")
        folder = (base / identifier).resolve()
        if folder.parent != base:
            raise ValueError("Un origen de recuperación sale de data/raw/batches.")
        name = f"data/raw/batches/{identifier}/captura.json"
        manifest = sources.json(name)
        if manifest.get("lote_id") != identifier:
            raise ValueError("El manifiesto no coincide con su lote de referencia.")
        visiting.add(identifier)
        origin = manifest.get("origen")
        if origin:
            if not isinstance(origin, dict):
                raise ValueError("Referencia de recuperación inválida.")
            parent = origin.get("lote_id")
            expected = origin.get("captura_sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError("Falta una huella válida del origen de recuperación.")
            visit(parent)
            if sources.sha(f"data/raw/batches/{parent}/captura.json") != expected:
                raise ValueError("El manifiesto de origen difiere de la huella guardada.")
        visiting.remove(identifier)
        visited.add(identifier)
        names.append(name)

    visit(lot_id)
    # Lee también listados y detalles correctos; exige conservar todos los orígenes.
    history._captures(sources, names)
    sources.verify_unchanged()
    return dict(sources.contents)


def copy_reference_captures(root, stage, lot_id):
    """Copia sólo las capturas necesarias a una carpeta temporal nueva."""
    root, stage = Path(root).resolve(), Path(stage).resolve()
    inputs = reference_capture_inputs(root, lot_id)
    for name, data in sorted(inputs.items()):
        target = (stage / name).resolve()
        if not target.is_relative_to(stage):
            raise ValueError("Una captura sale de la carpeta temporal.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    names = sorted(name for name in inputs if name.endswith("/captura.json"))
    copied = history._Sources(stage)
    history._captures(copied, names)
    copied.verify_unchanged()
    for name, data in inputs.items():
        if copied.read(name) != data or (root / name).read_bytes() != data:
            raise RuntimeError("Cambió una captura durante la preparación de la prueba.")
    return {root / name: history._sha(data) for name, data in inputs.items()}


def verify_pipeline(path, root=ROOT):
    import numpy as np
    import pandas as pd

    root, path = Path(root).resolve(), Path(path).resolve()
    plan = portable.inspect_bundle(path, root)
    lot_id = bundle.REFERENCE_LOT
    records = []
    for log_path in (root / "data/processed/batches" / lot_id).glob("*/registro_ejecucion.json"):
        log = bundle.read_json(log_path)
        if log.get("estado") == "completo":
            records.append((log_path, log))
    if len(records) != 1:
        raise ValueError("La referencia debe tener una única predicción completa.")
    log_path, original = records[0]
    csv_path = log_path.parent / "predicciones.csv"
    if (bundle.sha_file(csv_path) != original["predicciones_sha256"]
            or original["predicciones_sha256"] != plan["manifest"]["reference"]["csv_sha256"]
            or bundle.sha_file(log_path) != plan["manifest"]["reference"]["log_sha256"]):
        raise ValueError("La referencia histórica no coincide con la del paquete.")
    capture_folder = root / "data/raw/batches" / lot_id
    batch._load_capture(capture_folder)
    protected = {csv_path: bundle.sha_file(csv_path), log_path: bundle.sha_file(log_path)}
    base = root / "data/processed/deployment"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verificar-pipeline-", dir=base) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != base.resolve():
            raise RuntimeError("La carpeta de prueba debe quedar dentro de deployment.")
        target_capture = stage / "data/raw/batches" / lot_id
        protected.update(copy_reference_captures(root, stage, lot_id))
        for name in portable.RUNTIME_FILES:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        cache = stage / "cache_hf_vacia"
        cache.mkdir()
        environment = {portable.ENVIRONMENT_VARIABLE: str(path), "HF_HOME": str(cache),
                       "HUGGINGFACE_HUB_CACHE": str(cache / "hub"), "HF_HUB_CACHE": str(cache / "hub")}
        print("1. Ejecutando la ruta real de predicción en un proyecto temporal...")
        with patch.dict("os.environ", environment), \
                patch.object(batch, "_load_candidate", side_effect=AssertionError("No usar catálogo MLflow")), \
                patch.object(batch, "_originals", side_effect=AssertionError("No abrir originales MLflow")):
            current_path = batch.predecir_lote(target_capture, raiz=stage)
        current_log = bundle.read_json(current_path.parent / "registro_ejecucion.json")
        if (current_log["origen_carga_modelo"] != "paquete_portable"
                or current_log["paquete_modelo"] != plan["descriptor"]["bundle"]
                or "mlflow" in current_log["versiones"]
                or (stage / "mlflow_local").exists() or (stage / "models").exists()):
            raise ValueError("La inferencia no conserva el contrato portable.")
        current = pd.read_csv(current_path, dtype={"CodigoExterno": "string"}).set_index("CodigoExterno").sort_index()
        expected = pd.read_csv(csv_path, dtype={"CodigoExterno": "string"}).set_index("CodigoExterno").sort_index()
        print("2. Comparando textos, clases, probabilidades y trazabilidad...")
        if len(current) != 5 or not current.index.equals(expected.index):
            raise ValueError("No se reprodujeron los cinco códigos de referencia.")
        np.testing.assert_array_equal(current["AfinidadPredicha"].to_numpy(), expected["AfinidadPredicha"].to_numpy())
        columns = [f"ProbAfinidad{i}" for i in range(4)] + ["AfinidadEsperada"]
        np.testing.assert_allclose(current[columns].to_numpy(dtype=float), expected[columns].to_numpy(dtype=float),
                                   rtol=1e-5, atol=1e-6)
        for column in ("HashTextoSHA256", "HashFuenteSHA256", "TextoLicitacion", "ModeloURI", "ModeloRunId",
                       "HashModeloOriginalSHA256", "RevisionEncoderLocal", "VersionPerfil", "VersionCriterio"):
            np.testing.assert_array_equal(current[column].to_numpy(), expected[column].to_numpy())
        inventory = history._inventory(stage)
        sources = history._Sources(stage)
        _, captures, _, by_lot, manifests = history._captures(sources, inventory["capturas"])
        predictions, _, incomplete = history._predictions(sources, inventory["predicciones"], captures, by_lot, manifests)
        if len(predictions) != 5 or incomplete:
            raise ValueError("El historial no acepta las cinco predicciones portables.")
        sources.verify_unchanged()
    for file, digest in protected.items():
        if bundle.sha_file(file) != digest:
            raise RuntimeError("Cambió un archivo histórico protegido durante la comprobación.")
    print("OK: cinco clases y probabilidades reproducidas dentro de la tolerancia.")
    print("OK: el historial acepta las predicciones y conserva su trazabilidad.")
    print("OK: pipeline ejecutado desde el paquete, sin usar el catálogo MLflow ni una caché E5 previa.")
    print("OK: originales conservados; sin consultas API ni publicaciones BigQuery.")
    return {"casos": 5, "paquete": plan["manifest"]["bundle_id"], "modo": "portable_local"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_pipeline(args.archivo, ROOT), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
