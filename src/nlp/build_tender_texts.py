from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def join_texts(values) -> str:
    parts = [
        " ".join(str(value).split())
        for value in values
        if pd.notna(value)
    ]
    return " | ".join(dict.fromkeys(part for part in parts if part))


def build_tender_texts(
    tenders: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    items = items.copy()

    items["TextoItem"] = items.apply(
        lambda row: join_texts([
            row["NombreProducto"],
            row["Descripcion"],
        ]),
        axis=1,
    )

    item_texts = (
        items.groupby("CodigoExterno", sort=False)["TextoItem"]
        .agg(join_texts)
        .reset_index(name="TextoItems")
    )

    result = tenders.merge(
        item_texts,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    result["TextoLicitacion"] = result.apply(
        lambda row: join_texts([
            row["Nombre"],
            row["Descripcion"],
            row["TextoItems"],
        ]),
        axis=1,
    )

    if result["TextoLicitacion"].eq("").any():
        raise ValueError("Hay licitaciones sin texto para el modelo.")

    return result[["CodigoExterno", "TextoLicitacion"]]


if __name__ == "__main__":
    tenders = pd.read_csv(
        PROCESSED_DIR / "tenders.csv",
        dtype={"CodigoExterno": "string"},
        usecols=["CodigoExterno", "Nombre", "Descripcion"],
    )

    items = pd.read_csv(
        PROCESSED_DIR / "tender_items.csv",
        dtype={"CodigoExterno": "string"},
        usecols=["CodigoExterno", "NombreProducto", "Descripcion"],
    )

    texts = build_tender_texts(tenders, items)
    output_path = PROCESSED_DIR / "tender_texts.csv"
    texts.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Licitaciones preparadas: {len(texts)}")
    print(f"Archivo generado: {output_path}")
    print("\nEjemplo:")
    print(texts.iloc[0]["TextoLicitacion"][:600])