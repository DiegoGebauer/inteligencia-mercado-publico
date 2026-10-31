from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_affinity_dataset() -> pd.DataFrame:
    labels = pd.read_csv(
        LABELED_DIR / "tender_labels.csv",
        dtype={"CodigoExterno": "string"},
    )
    reviewed = pd.read_csv(
        LABELED_DIR / "tender_labels_reviewed.csv",
        dtype={"CodigoExterno": "string"},
    )
    texts = pd.read_csv(
        PROCESSED_DIR / "tender_texts.csv",
        dtype={"CodigoExterno": "string"},
    )

    for name, dataframe in [
        ("etiquetas", labels),
        ("revisión", reviewed),
        ("textos", texts),
    ]:
        codes = dataframe["CodigoExterno"]
        if codes.isna().any() or not codes.is_unique:
            raise ValueError(f"Hay códigos vacíos o duplicados en {name}.")

    labels = labels.set_index("CodigoExterno")
    reviewed = reviewed.set_index("CodigoExterno")

    if not reviewed.index.isin(labels.index).all():
        raise ValueError("Hay revisiones sin licitación original.")

    if set(texts["CodigoExterno"]) != set(labels.index):
        raise ValueError("Los textos y las etiquetas cubren distintos códigos.")

    labels["OrigenEtiqueta"] = "asistida"

    # La revisión reemplaza también los valores dejados sin determinar.
    labels.loc[reviewed.index, "Afinidad"] = reviewed["Afinidad"]
    labels.loc[reviewed.index, "OrigenEtiqueta"] = reviewed["OrigenEtiqueta"]

    dataset = texts.merge(
        labels.reset_index()[
            ["CodigoExterno", "Afinidad", "OrigenEtiqueta"]
        ],
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    if (
        dataset["TextoLicitacion"].isna().any()
        or dataset["TextoLicitacion"].str.strip().eq("").any()
    ):
        raise ValueError("Hay licitaciones sin texto.")

    dataset["Afinidad"] = pd.to_numeric(
        dataset["Afinidad"], errors="raise"
    )

    if not dataset["Afinidad"].dropna().isin([0, 1, 2, 3]).all():
        raise ValueError("Afinidad debe ser 0, 1, 2 o 3.")

    # Una afinidad desconocida no se convierte en una etiqueta 0.
    dataset = dataset.loc[dataset["Afinidad"].notna()].copy()
    dataset["Afinidad"] = dataset["Afinidad"].astype("int64")

    return dataset.reset_index(drop=True)


if __name__ == "__main__":
    dataset = build_affinity_dataset()

    output_path = LABELED_DIR / "affinity_dataset.csv"
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Licitaciones preparadas: {len(dataset)}")
    print("\nDistribución de afinidad:")
    print(dataset["Afinidad"].value_counts().sort_index().to_string())
    print("\nOrigen de las etiquetas:")
    print(dataset["OrigenEtiqueta"].value_counts().to_string())
    print(f"\nArchivo generado: {output_path}")