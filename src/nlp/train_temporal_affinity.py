"""Entrenamiento temporal con encoder fijo y evaluación test independiente."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.nlp.train_affinity import encode_tenders
from src.nlp.rank_tenders import MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "labeled" / "temporal_v1"
MODEL = ROOT / "models" / "affinity_temporal_v1.joblib"
REPORTS = ROOT / "data" / "processed"
CLASSES = [0, 1, 2, 3]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sets():
    frames = {}
    for split, filename in [
        ("train", "train.csv"),
        ("val", "validation.csv"),
        ("test", "test.csv"),
    ]:
        frame = pd.read_csv(DATA / filename, dtype={"CodigoExterno": "string"})
        if frame.empty or frame["CodigoExterno"].isna().any() or not frame["CodigoExterno"].is_unique:
            raise ValueError(f"Códigos inválidos en {split}.")
        if frame["TextoLicitacion"].isna().any() or frame["TextoLicitacion"].str.strip().eq("").any():
            raise ValueError(f"Textos vacíos en {split}.")
        if not frame["Conjunto"].eq(split).all():
            raise ValueError(f"Conjunto incorrecto en {split}.")
        affinity = pd.to_numeric(frame["Afinidad"], errors="raise")
        if not affinity.dropna().isin(CLASSES).all():
            raise ValueError(f"Afinidad fuera de 0–3 en {split}.")
        frame["Afinidad"] = affinity
        dates = pd.to_datetime(frame["FechaPublicacion"], errors="raise")
        if dates.isna().any():
            raise ValueError(f"Fecha vacía en {split}.")
        correct = (
            dates.ge("2026-01-01") & dates.lt("2026-08-01")
            if split == "train"
            else dates.dt.strftime("%Y-%m").eq("2026-08" if split == "val" else "2026-09")
        )
        if not correct.all():
            raise ValueError(f"Hay publicaciones fuera del período de {split}.")
        if split == "train" and affinity.isna().any():
            raise ValueError("Train contiene afinidades sin determinar.")
        frames[split] = frame
    all_rows = pd.concat(frames.values(), ignore_index=True)
    if not all_rows["CodigoExterno"].is_unique:
        raise ValueError("Hay códigos compartidos entre conjuntos.")
    normalized = all_rows["TextoLicitacion"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    if all_rows.assign(_text=normalized).groupby("_text")["Conjunto"].nunique().gt(1).any():
        raise ValueError("Hay textos idénticos entre conjuntos.")
    return frames


def known_rows(frame):
    selected = frame.loc[frame["Afinidad"].notna()].copy()
    if selected.empty:
        raise ValueError("No hay etiquetas conocidas para evaluar.")
    return selected


def scores(frame, predictions):
    y = frame["Afinidad"].astype(int).to_numpy()
    weights = pd.to_numeric(frame["PesoMuestra"], errors="raise").to_numpy()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Pesos de evaluación inválidos.")
    return {
        "F1MacroPonderado": f1_score(y, predictions, labels=CLASSES, average="macro", sample_weight=weights, zero_division=0),
        "ExactitudPonderada": accuracy_score(y, predictions, sample_weight=weights),
        "MAEPonderado": mean_absolute_error(y, predictions, sample_weight=weights),
    }


def save_predictions(frame, predictions, filename):
    result = frame.copy()
    result["AfinidadPredicha"] = predictions
    result["ErrorAbsoluto"] = np.abs(result["Afinidad"] - predictions)
    REPORTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS / filename, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true", help="Evalúa el modelo guardado en test; no reentrena.")
    args = parser.parse_args()
    frames = load_sets()
    print("Los resultados comparan contra etiquetas mayoritariamente asistidas; no son una evaluación experta completa.")
    if args.evaluate_test:
        saved = joblib.load(MODEL)
        for filename, expected in saved["dataset_hashes"].items():
            if file_hash(DATA / filename) != expected:
                raise ValueError(f"Cambió {filename} desde el entrenamiento. Revisa las etiquetas y vuelve a entrenar antes de evaluar test.")
        test = known_rows(frames["test"])
        encoder = SentenceTransformer(saved["embedding_model"], device="cpu")
        X_test = encode_tenders(test["TextoLicitacion"].tolist(), encoder)
        predictions = saved["classifier"].predict(X_test)
        print("Test:", scores(test, predictions))
        save_predictions(test, predictions, "affinity_temporal_test_predictions.csv")
        return

    train = frames["train"]
    validation = known_rows(frames["val"])
    y_train = train["Afinidad"].astype(int).to_numpy()
    if set(y_train) != set(CLASSES):
        raise ValueError("Train debe incluir las cuatro clases.")
    print(f"Train: {len(train)}; validación etiquetada: {len(validation)}; pendientes: {len(frames['val']) - len(validation)}")
    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    X_train = encode_tenders(train["TextoLicitacion"].tolist(), encoder)
    X_val = encode_tenders(validation["TextoLicitacion"].tolist(), encoder)
    candidates = []
    for c in [0.1, 1.0, 10.0]:
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42))
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_val)
        result = scores(validation, predictions)
        print(f"C={c}: {result}")
        candidates.append((result["F1MacroPonderado"], -result["MAEPonderado"], -c, classifier, predictions, result))
    best = max(candidates, key=lambda item: item[:3])
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print("Referencia mayoritaria:", scores(validation, baseline.predict(X_val)))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "classifier": best[3], "embedding_model": MODEL_NAME,
        "pooling": "mean_normalized_fragments", "target": "Afinidad",
        "dataset_version": "temporal_v1", "profile_version": "0.1.0",
        "chosen_C": -best[2], "validation_scores": best[5],
        "train_codes": train["CodigoExterno"].tolist(),
        "validation_codes": validation["CodigoExterno"].tolist(),
        "dataset_hashes": {name: file_hash(DATA / name) for name in ["train.csv", "validation.csv", "test.csv"]},
    }, MODEL)
    save_predictions(validation, best[4], "affinity_temporal_validation_predictions.csv")
    print(f"Modelo guardado: {MODEL}")
    print("Test queda reservado para la evaluación final con --evaluate-test.")


if __name__ == "__main__":
    main()
