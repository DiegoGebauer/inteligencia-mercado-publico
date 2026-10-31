from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.nlp.rank_tenders import MODEL_NAME, split_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42
CLASSES = [0, 1, 2, 3]


def encode_tenders(texts: list[str], encoder) -> np.ndarray:
    passages = []
    owners = []

    for index, text in enumerate(texts):
        chunks = split_text(text, encoder.tokenizer)

        if not chunks:
            raise ValueError(f"Texto sin fragmentos en la fila {index}.")

        for chunk in chunks:
            passages.append("passage: " + chunk)
            owners.append(index)

    print(f"Fragmentos a representar: {len(passages)}")

    fragment_embeddings = encoder.encode(
        passages,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Produce un único vector por licitación.
    embeddings = np.zeros(
        (len(texts), fragment_embeddings.shape[1]),
        dtype=np.float32,
    )
    np.add.at(embeddings, np.asarray(owners), fragment_embeddings)

    counts = np.bincount(owners, minlength=len(texts))
    embeddings /= counts[:, None]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)

    if not np.isfinite(embeddings).all():
        raise ValueError("Hay embeddings con valores inválidos.")

    return embeddings


def main():
    dataset = pd.read_csv(
        LABELED_DIR / "affinity_dataset.csv",
        dtype={"CodigoExterno": "string"},
    )

    if (
        dataset.empty
        or dataset["CodigoExterno"].isna().any()
        or not dataset["CodigoExterno"].is_unique
        or dataset["TextoLicitacion"].isna().any()
        or dataset["TextoLicitacion"].str.strip().eq("").any()
    ):
        raise ValueError("El dataset tiene códigos o textos inválidos.")

    if dataset["TextoLicitacion"].duplicated().any():
        raise ValueError(
            "Hay textos idénticos: debemos agruparlos antes de dividir."
        )

    affinity = pd.to_numeric(dataset["Afinidad"], errors="raise")
    if not affinity.isin(CLASSES).all():
        raise ValueError("Hay afinidades vacías o fuera de 0–3.")

    y = affinity.astype("int64").to_numpy()

    if set(y) != set(CLASSES) or pd.Series(y).value_counts().min() < 2:
        raise ValueError("Faltan ejemplos para dividir las cuatro clases.")

    # La separación es por licitación, nunca por fragmento.
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    print(f"Entrenamiento: {len(train_indices)} licitaciones")
    print(f"Validación provisional: {len(val_indices)} licitaciones")
    print(f"\nCargando encoder fijo: {MODEL_NAME}")

    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    X = encode_tenders(
        dataset["TextoLicitacion"].tolist(),
        encoder,
    )

    # El escalado aprende únicamente del conjunto de entrenamiento.
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
        ),
    )

    print("\nEntrenando el clasificador...")
    classifier.fit(X[train_indices], y[train_indices])

    predictions = classifier.predict(X[val_indices])

    # Referencia simple: predecir siempre la clase mayoritaria.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X[train_indices], y[train_indices])
    baseline_predictions = baseline.predict(X[val_indices])

    print("\nF1 macro — referencia mayoritaria:")
    print(f"{f1_score(
        y[val_indices], baseline_predictions,
        labels=CLASSES, average='macro', zero_division=0
    ):.3f}")

    print("\nF1 macro — clasificador entrenado:")
    print(f"{f1_score(
        y[val_indices], predictions,
        labels=CLASSES, average='macro', zero_division=0
    ):.3f}")

    print("\nError absoluto medio de afinidad:")
    print(f"{mean_absolute_error(y[val_indices], predictions):.3f}")

    print("\nResultados por clase:")
    print(classification_report(
        y[val_indices],
        predictions,
        labels=CLASSES,
        zero_division=0,
    ))

    print("Matriz de confusión — filas reales, columnas predichas:")
    print(confusion_matrix(
        y[val_indices], predictions, labels=CLASSES
    ))

    validation = dataset.iloc[val_indices].copy()
    validation["AfinidadPredicha"] = predictions
    validation["ErrorAbsoluto"] = np.abs(
        validation["Afinidad"] - predictions
    )

    validation_path = (
        PROCESSED_DIR / "affinity_validation_predictions.csv"
    )
    validation.to_csv(
        validation_path, index=False, encoding="utf-8"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "affinity_classifier.joblib"

    joblib.dump(
        {
            "classifier": classifier,
            "embedding_model": MODEL_NAME,
            "pooling": "mean_normalized_fragments",
            "target": "Afinidad",
            "seed": SEED,
            "train_codes": (
                dataset.iloc[train_indices]["CodigoExterno"].tolist()
            ),
            "validation_codes": (
                dataset.iloc[val_indices]["CodigoExterno"].tolist()
            ),
        },
        model_path,
    )

    print(f"\nModelo guardado: {model_path}")
    print(f"Predicciones de validación: {validation_path}")


if __name__ == "__main__":
    main()