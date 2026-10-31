from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_monthly_affinity_dataset() -> pd.DataFrame:
    batch_paths = sorted(
        LABELED_DIR.glob("training_review_batch_*.csv")
    )

    if not batch_paths:
        raise ValueError("No hay lotes de revisión disponibles.")

    label_columns = [
        "CodigoExterno",
        "Afinidad",
        "EncajeServicio",
        "Especializacion",
        "Motivo",
        "OrigenEtiqueta",
        "EstadoRevision",
    ]

    labels = pd.concat(
        [
            pd.read_csv(
                path,
                dtype={"CodigoExterno": "string"},
                usecols=label_columns,
            )
            for path in batch_paths
        ],
        ignore_index=True,
    )

    texts = pd.read_csv(
        PROCESSED_DIR / "monthly_texts_2026_combined.csv",
        dtype={"CodigoExterno": "string"},
    )

    for name, dataframe in [
        ("etiquetas", labels),
        ("textos mensuales", texts),
    ]:
        codes = dataframe["CodigoExterno"]

        if (
            codes.isna().any()
            or codes.str.strip().eq("").any()
            or not codes.is_unique
        ):
            raise ValueError(
                f"Hay códigos vacíos o duplicados en {name}."
            )

    if not labels["CodigoExterno"].isin(
        texts["CodigoExterno"]
    ).all():
        raise ValueError(
            "Hay etiquetas sin texto en el catálogo mensual."
        )

    labels["Afinidad"] = pd.to_numeric(
        labels["Afinidad"], errors="raise"
    )

    if not labels["Afinidad"].dropna().isin(
        [0, 1, 2, 3]
    ).all():
        raise ValueError("Afinidad debe ser 0, 1, 2 o 3.")

    # Los casos sin determinar quedan fuera del entrenamiento.
    labels = labels.loc[labels["Afinidad"].notna()].copy()
    labels["Afinidad"] = labels["Afinidad"].astype("int64")

    dataset = labels.merge(
        texts,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    if (
        dataset["TextoLicitacion"].isna().any()
        or dataset["TextoLicitacion"].str.strip().eq("").any()
    ):
        raise ValueError("Hay licitaciones sin texto.")

    dates = pd.to_datetime(
        dataset["FechaPublicacion"], errors="raise"
    )

    if dates.isna().any():
        raise ValueError("Hay licitaciones sin fecha de publicación.")

    if (
        dataset["PeriodoPublicacion"].isna().any()
        or not dates.dt.strftime("%Y-%m").eq(
            dataset["PeriodoPublicacion"]
        ).all()
    ):
        raise ValueError("El período no coincide con la publicación.")

    return dataset.reset_index(drop=True)


if __name__ == "__main__":
    dataset = build_monthly_affinity_dataset()

    output_path = LABELED_DIR / "monthly_affinity_dataset.csv"
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Licitaciones preparadas: {len(dataset)}")

    print("\nDistribución de afinidad:")
    print(dataset["Afinidad"].value_counts().sort_index().to_string())

    print("\nDistribución por mes:")
    print(
        dataset["PeriodoPublicacion"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOrigen de las etiquetas:")
    print(dataset["OrigenEtiqueta"].value_counts().to_string())

    print(f"\nArchivo generado: {output_path}")