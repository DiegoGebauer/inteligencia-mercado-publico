"""Compara el modelo empaquetado contra cinco resultados históricos, sin cloud."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from src.experiments import export_model_bundle as bundle


ROOT = Path(__file__).resolve().parents[2]


def verify_inference(path, root=ROOT):
    manifest = bundle.verify_archive(path)
    for package in ("numpy", "scikit-learn", "pandas", "sentence-transformers", "transformers", "torch"):
        expected = manifest["reference_environment"]["packages"][package]
        if version(package) != expected:
            raise RuntimeError(f"La versión de {package} difiere de la referencia ({expected}).")
    for relative, expected in manifest["inference_code_sha256"].items():
        if bundle.sha_file(root / relative) != expected:
            raise ValueError(f"Cambió el código de inferencia: {relative}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from src.nlp.rank_tenders import split_text
    from src.nlp.train_affinity import encode_tenders

    temporary_base = (root / "data/processed/deployment").resolve()
    required = sum(item["bytes"] for item in manifest["files"].values()) + 64 * 1024**2
    existing = next(parent for parent in [temporary_base, *temporary_base.parents] if parent.exists())
    if shutil.disk_usage(existing).free < required:
        raise OSError("Falta espacio para extraer temporalmente el modelo.")
    temporary_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verificar-modelo-", dir=temporary_base) as directory:
        destination = Path(directory).resolve()
        if destination.parent != temporary_base:
            raise RuntimeError("La carpeta temporal no está dentro de la ubicación esperada.")
        # Extraer sólo los miembros verificados; no usa extractall ni interpreta enlaces.
        with tarfile.open(path, "r:gz") as archive:
            for name in ["manifest.json", *manifest["files"]]:
                target = destination.joinpath(*Path(name).parts).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("El archivo sale de la carpeta temporal.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            for name in manifest["directories"]:
                destination.joinpath(*Path(name).parts).mkdir(parents=True, exist_ok=True)
        # Verificar de nuevo los archivos extraídos antes de abrir el joblib aprobado.
        for name, spec in manifest["files"].items():
            if bundle.sha_file(destination / name) != spec["sha256"]:
                raise ValueError("Cambió un archivo después de extraerlo.")
        saved = joblib.load(destination / "classifier.joblib")
        if (saved.get("embedding_model") != bundle.ENCODER or saved.get("target") != "Afinidad"
                or saved.get("pooling") != "mean_normalized_fragments"
                or saved.get("profile_version") != manifest["profile_version"]
                or saved.get("labeling_policy_version") != manifest["labeling_policy_version"]):
            raise ValueError("Los metadatos del clasificador no coinciden con el paquete.")
        classifier = saved["classifier"]
        np.testing.assert_array_equal(classifier.classes_, [0, 1, 2, 3])
        if classifier.n_features_in_ != 384 or classifier.get_params()["logisticregression__C"] != 0.1:
            raise ValueError("El clasificador no cumple el contrato del candidato.")
        encoder = SentenceTransformer(str(destination / "encoder"), device="cpu", local_files_only=True)
        if encoder.get_embedding_dimension() != 384:
            raise ValueError("El encoder no produce 384 dimensiones.")
        cases = bundle.read_json(destination / "regression_cases.json")["cases"]
        if len(cases) != 5 or len({row["codigo"] for row in cases}) != 5:
            raise ValueError("La referencia no contiene cinco casos únicos.")
        lengths = [len(encoder.tokenizer.encode("passage: " + chunk, add_special_tokens=True, truncation=False))
                   for row in cases for chunk in split_text(row["texto"], encoder.tokenizer)]
        if not lengths or max(lengths) > min(encoder.max_seq_length, 512):
            raise ValueError("Los fragmentos exceden el límite del encoder.")
        embeddings = encode_tenders([row["texto"] for row in cases], encoder)
        if embeddings.shape != (5, 384) or embeddings.dtype != np.float32:
            raise ValueError("La representación no cumple el contrato esperado.")
        np.testing.assert_array_equal(classifier.predict(embeddings), [row["clase"] for row in cases])
        np.testing.assert_allclose(classifier.predict_proba(embeddings),
                                   [row["probabilidades"] for row in cases], rtol=1e-5, atol=1e-6)
        print("OK: las cinco clases coinciden con la referencia histórica.")
        print("OK: las probabilidades coinciden dentro de la tolerancia numérica.")
        print("Fragmento más largo:", max(lengths), "tokens; límite:", encoder.max_seq_length)
    print("OK: modelo y encoder cargados desde el paquete; sin usar MLflow ni consultas cloud.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_inference(args.archivo, ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
