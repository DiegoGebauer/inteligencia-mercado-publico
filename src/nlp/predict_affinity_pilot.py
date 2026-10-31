"""Aplica el clasificador guardado a un piloto histórico; no lo reentrena."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.nlp.train_affinity import encode_tenders


ROOT = Path(__file__).resolve().parents[2]
MODEL_RELATIVE = Path("models/affinity_temporal_v1.joblib")
SOURCE_RELATIVE = Path("data/processed/monthly_texts_2026_combined.csv")
OUTPUT_RELATIVE = Path("data/processed/affinity_pilot_predictions.csv")
MANIFEST_RELATIVE = Path("data/processed/affinity_pilot_manifest.json")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return " ".join(text.split())


def tender_family(code):
    parts = str(code).split("-")
    return "-".join(parts[:2]) + "-" + parts[-1][-2:]


def prepare_pool(root, saved, profile):
    source = pd.read_csv(root / SOURCE_RELATIVE, dtype={"CodigoExterno": "string"})
    required = {"CodigoExterno", "FechaPublicacion", "Nombre", "TextoLicitacion"}
    if not required.issubset(source.columns):
        raise ValueError(f"Faltan columnas: {required - set(source.columns)}")
    if source["CodigoExterno"].isna().any() or not source["CodigoExterno"].is_unique:
        raise ValueError("El catálogo contiene códigos vacíos o repetidos.")
    dates = pd.to_datetime(source["FechaPublicacion"], errors="raise")
    if dates.isna().any():
        raise ValueError("Hay fechas de publicación vacías.")
    # Agosto y septiembre se reservan para evaluación.
    pool = source.loc[dates.ge("2026-01-01") & dates.lt("2026-08-01")].copy()
    if pool["TextoLicitacion"].isna().any() or pool["TextoLicitacion"].str.strip().eq("").any():
        raise ValueError("Hay licitaciones sin texto en el período del piloto.")

    known_codes = set(saved["train_codes"]) | set(saved["validation_codes"])
    known_texts = set()
    # Excluir etiquetas antiguas y actuales, incluidas las copias de control.
    paths = sorted((root / "data/labeled").rglob("*.csv"))
    legacy_texts = root / "data/processed/tender_texts.csv"
    if legacy_texts.exists():
        paths.append(legacy_texts)
    for path in paths:
        headers = pd.read_csv(path, nrows=0).columns
        columns = [name for name in ["CodigoExterno", "TextoLicitacion"] if name in headers]
        if not columns:
            continue
        frame = pd.read_csv(path, usecols=columns, dtype=str)
        if "CodigoExterno" in frame:
            known_codes.update(frame["CodigoExterno"].dropna())
        if "TextoLicitacion" in frame:
            known_texts.update(frame["TextoLicitacion"].dropna().map(normalize))
    known_families = {tender_family(code) for code in known_codes}
    pool["_family"] = pool["CodigoExterno"].map(tender_family)
    pool["_text"] = pool["TextoLicitacion"].map(normalize)
    pool = pool.loc[
        ~pool["CodigoExterno"].isin(known_codes)
        & ~pool["_family"].isin(known_families)
        & ~pool["_text"].isin(known_texts)
    ].copy()
    pool = pool.sort_values(["FechaPublicacion", "CodigoExterno"])
    pool = pool.drop_duplicates("_family").drop_duplicates("_text").copy()

    topics = {
        normalize(keyword)
        for service in profile["relevant_services"].values()
        for keyword in service["keywords"]
        if str(keyword).strip()
    }
    topics.update(["bases de datos", "gobernanza de datos"])
    if not topics:
        raise ValueError("El perfil no tiene temas para seleccionar candidatos.")
    pattern = r"\b(?:" + "|".join(re.escape(topic) for topic in sorted(topics)) + r")\b"
    pool["_signal"] = pool["_text"].str.contains(pattern, regex=True)
    return pool


def select_sample(pool, size, seed):
    if size <= 0 or size > len(pool):
        raise ValueError(f"Tamaño inválido: {size}. Hay {len(pool)} casos elegibles.")
    # Enriquecer el piloto para poder observar oportunidades y confusiones.
    # Las palabras seleccionan candidatos; no les asignan afinidad.
    signals = pool.loc[pool["_signal"]]
    rest = pool.loc[~pool["_signal"]]
    signal_count = min(round(size * 0.30), len(signals))
    signal_count = max(signal_count, size - len(rest))
    chosen_signals = signals.sample(n=signal_count, random_state=seed).copy()
    chosen_rest = rest.sample(n=size - signal_count, random_state=seed + 1).copy()
    chosen_signals["GrupoPiloto"] = "senales_del_perfil"
    chosen_rest["GrupoPiloto"] = "resto"
    return pd.concat([chosen_signals, chosen_rest], ignore_index=True).sort_values(
        ["FechaPublicacion", "CodigoExterno"]
    ).reset_index(drop=True)


def order_predictions(frame):
    result = frame.drop(columns=["Posicion"], errors="ignore").copy()
    required = {"CodigoExterno", "AfinidadPredicha", "ProbAfinidad0", "ProbAfinidad1", "ProbAfinidad2", "ProbAfinidad3"}
    if not required.issubset(result.columns):
        raise ValueError(f"Faltan columnas para ordenar: {required - set(result.columns)}")
    if result.empty or result["CodigoExterno"].isna().any() or not result["CodigoExterno"].is_unique:
        raise ValueError("El piloto tiene códigos vacíos o repetidos.")
    if not result["AfinidadPredicha"].isin([0, 1, 2, 3]).all():
        raise ValueError("El piloto tiene afinidades predichas inválidas.")
    probabilities = result[[f"ProbAfinidad{k}" for k in [0, 1, 2, 3]]].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("El piloto tiene probabilidades inválidas.")
    # Valorar los cuatro niveles evita tratar 2 y 3 como equivalentes al ordenar.
    result["AfinidadEsperada"] = probabilities @ np.array([0, 1, 2, 3])
    result = result.sort_values(
        ["AfinidadEsperada", "ProbAfinidad3", "CodigoExterno"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result.insert(0, "Posicion", np.arange(1, len(result) + 1))
    front = ["Posicion", "CodigoExterno", "Nombre", "FechaPublicacion", "AfinidadPredicha", "AfinidadEsperada"]
    return result[front + [column for column in result if column not in front]]


def build_predictions(frame, classifier, embeddings):
    classes = np.asarray(classifier.classes_)
    if not np.array_equal(classes, [0, 1, 2, 3]):
        raise ValueError("El modelo no contiene las cuatro clases esperadas.")
    probabilities = classifier.predict_proba(embeddings)
    if (
        probabilities.shape != (len(frame), 4)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("El modelo devolvió probabilidades inválidas.")
    predictions = classifier.predict(embeddings)
    if not np.isin(predictions, classes).all():
        raise ValueError("El modelo devolvió afinidades inválidas.")
    result = frame.drop(columns=[c for c in frame if c.startswith("_")]).copy()
    result["AfinidadPredicha"] = predictions
    for index, affinity in enumerate(classes):
        result[f"ProbAfinidad{affinity}"] = probabilities[:, index]
    result["ScoreRelevancia"] = probabilities[:, 2] + probabilities[:, 3]
    ordered_probabilities = np.sort(probabilities, axis=1)
    result["MargenPrediccion"] = ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
    return order_predictions(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza los resultados del piloto anterior.")
    parser.add_argument("--rerank", action="store_true", help="Reordena el piloto existente usando las probabilidades guardadas.")
    args = parser.parse_args()
    output = ROOT / OUTPUT_RELATIVE
    manifest_path = ROOT / MANIFEST_RELATIVE
    if args.rerank:
        original = pd.read_csv(output, dtype={"CodigoExterno": "string"})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(original) != manifest["size"]:
            raise ValueError("El tamaño del piloto no coincide con su registro.")
        if file_hash(ROOT / MODEL_RELATIVE) != manifest["model_sha256"]:
            raise ValueError("El modelo cambió desde la predicción del piloto.")
        result = order_predictions(original)
        manifest["previous_ranking_score"] = manifest["ranking_score"]
        manifest["ranking_score"] = "ProbAfinidad1 + 2 * ProbAfinidad2 + 3 * ProbAfinidad3"
        manifest["ranking_revision"] = "afinidad_esperada_v2"
        result.to_csv(output, index=False, encoding="utf-8-sig")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Piloto reordenado con las probabilidades guardadas; afinidades predichas conservadas.")
        print(result[["Posicion", "AfinidadPredicha", "AfinidadEsperada", "Nombre"]].head(20).to_string(index=False))
        print(f"Archivo actualizado: {output}")
        return
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError("Ya existe un piloto. Para reemplazarlo, usa --overwrite.")
    model_path = ROOT / MODEL_RELATIVE
    saved = joblib.load(model_path)
    if saved.get("target") != "Afinidad" or saved.get("pooling") != "mean_normalized_fragments":
        raise ValueError("El modelo guardado no usa la representación esperada.")
    with (ROOT / "config/company_profile.yaml").open(encoding="utf-8") as file:
        profile = yaml.safe_load(file)
    if str(profile["profile_version"]) != saved["profile_version"]:
        raise ValueError("La versión del perfil no coincide con la del modelo.")
    for filename, expected in saved["dataset_hashes"].items():
        if file_hash(ROOT / "data/labeled/temporal_v1" / filename) != expected:
            raise ValueError(f"Cambió {filename}. Cierra las etiquetas y reentrena antes del piloto.")

    pool = prepare_pool(ROOT, saved, profile)
    sample = select_sample(pool, args.size, args.seed)
    print(f"Licitaciones elegibles de enero a julio: {len(pool):,}")
    print(f"Piloto: {len(sample):,}")
    print("Selección:", sample["GrupoPiloto"].value_counts().to_dict())
    print("Aplicando el modelo guardado. Las afinidades resultantes son predicciones.")
    # El entrenamiento ya descargó este encoder; se reutiliza su copia local.
    encoder = SentenceTransformer(saved["embedding_model"], device="cpu", local_files_only=True)
    embeddings = encode_tenders(sample["TextoLicitacion"].tolist(), encoder)
    result = build_predictions(sample, saved["classifier"], embeddings)
    manifest = {
        "purpose": "exploracion_para_revision_y_ampliacion_del_train",
        "size": len(result), "seed": args.seed,
        "period_start": "2026-01-01", "period_end_exclusive": "2026-08-01",
        "eligible_cases": len(pool),
        "eligible_by_group": pool["_signal"].value_counts().rename(index={True: "senales_del_perfil", False: "resto"}).to_dict(),
        "sample_by_group": sample["GrupoPiloto"].value_counts().to_dict(),
        "sample_by_month": sample["FechaPublicacion"].str[:7].value_counts().sort_index().to_dict(),
        "model_sha256": file_hash(model_path),
        "source_sha256": file_hash(ROOT / SOURCE_RELATIVE),
        "profile_sha256": file_hash(ROOT / "config/company_profile.yaml"),
        "embedding_model": saved["embedding_model"],
        "dataset_version": saved["dataset_version"],
        "label_status": "predicciones_sin_validacion",
        "ranking_score": "ProbAfinidad1 + 2 * ProbAfinidad2 + 3 * ProbAfinidad3",
        "ranking_revision": "afinidad_esperada_v2",
        "sampling_note": "Muestra enriquecida con señales del perfil y resto aleatorio; no estima proporciones del universo.",
        "duplicate_exclusions": "Códigos ya etiquetados, familias de readjudicación y textos normalizados idénticos. No garantiza excluir todos los contenidos similares.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Distribución de predicciones:")
    print(result["AfinidadPredicha"].value_counts().sort_index().to_string())
    print("\nPrimeras 20:")
    print(result[["Posicion", "AfinidadPredicha", "AfinidadEsperada", "Nombre"]].head(20).to_string(index=False))
    print(f"\nArchivo generado: {output}")
    print(f"Registro del piloto: {manifest_path}")


if __name__ == "__main__":
    main()
