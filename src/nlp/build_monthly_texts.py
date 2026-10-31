import argparse
from pathlib import Path

import pandas as pd

from src.nlp.build_tender_texts import build_tender_texts, join_texts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Recuperamos signos de Windows-1252 leídos como controles con Latin1.
TEXT_REPLACEMENTS = {
    code: bytes([code]).decode("cp1252", errors="ignore") or " "
    for code in range(0x80, 0xA0)
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).translate(TEXT_REPLACEMENTS)
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--month", type=int, choices=range(1, 13), required=True
    )
    args = parser.parse_args()

    period = f"{args.year}_{args.month:02d}"

    catalog_path = PROCESSED_DIR / f"monthly_catalog_{period}.csv"
    items_path = PROCESSED_DIR / f"monthly_items_{period}.csv"
    output_path = PROCESSED_DIR / f"monthly_texts_{period}.csv"

    for input_path in [catalog_path, items_path]:
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")

    if output_path.exists():
        raise FileExistsError(f"El archivo ya existe: {output_path}")

    catalog = pd.read_csv(catalog_path, dtype="string")
    items = pd.read_csv(
        items_path,
        dtype="string",
        usecols=[
            "CodigoExterno",
            "CodigoItem",
            "NombreProducto",
            "NombreItem",
            "Descripcion",
        ],
    )

    if catalog["CodigoExterno"].duplicated().any():
        raise ValueError("Hay códigos repetidos en el catálogo.")

    if items.duplicated(["CodigoExterno", "CodigoItem"]).any():
        raise ValueError("Hay ítems repetidos.")

    if set(catalog["CodigoExterno"]) != set(items["CodigoExterno"]):
        raise ValueError("Los códigos del catálogo y los ítems no coinciden.")

    for column in ["Nombre", "Descripcion"]:
        catalog[column] = catalog[column].map(clean_text)

    for column in ["NombreProducto", "NombreItem", "Descripcion"]:
        items[column] = items[column].map(clean_text)

    # Incluimos el nombre específico de la línea en el texto del producto.
    # Esta adaptación ocurre en memoria.
    items["NombreProducto"] = items.apply(
        lambda row: join_texts([
            row["NombreProducto"],
            row["NombreItem"],
        ]),
        axis=1,
    )

    texts = build_tender_texts(catalog, items)

    # Conservamos la fecha de publicación y los metadatos del catálogo.
    result = catalog.merge(
        texts,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    if result["TextoLicitacion"].isna().any():
        raise ValueError("Hay licitaciones sin texto.")

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Licitaciones con texto: {len(result):,}")
    print(f"Archivo generado: {output_path}")
    print("\nEjemplo:")
    print(result.iloc[0]["TextoLicitacion"][:800])


if __name__ == "__main__":
    main()