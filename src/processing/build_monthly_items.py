import argparse
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONTHLY_DIR = PROJECT_ROOT / "data" / "raw" / "monthly"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COLUMN_MAPPING = {
    "CodigoExterno": "CodigoExterno",
    "Codigoitem": "CodigoItem",
    "Correlativo": "Correlativo",
    "CodigoProductoONU": "CodigoProducto",
    "Nombre producto genrico": "NombreProducto",
    "Nombre linea Adquisicion": "NombreItem",
    "Descripcion linea Adquisicion": "Descripcion",
    "UnidadMedida": "UnidadMedida",
    "Cantidad": "Cantidad",
}


def build_monthly_items(year: int, month: int) -> pd.DataFrame:
    zip_path = MONTHLY_DIR / f"{year}-{month}.zip"

    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró: {zip_path}")

    parts = []

    with zipfile.ZipFile(zip_path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                "Se esperaba exactamente un CSV dentro del ZIP."
            )

        with archive.open(csv_files[0]) as file:
            for chunk in pd.read_csv(
                file,
                sep=";",
                encoding="latin1",
                dtype="string",
                usecols=list(COLUMN_MAPPING),
                chunksize=50_000,
            ):
                parts.append(chunk.drop_duplicates())

    items = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates()
        .rename(columns=COLUMN_MAPPING)
    )

    key_columns = ["CodigoExterno", "CodigoItem"]

    for column in key_columns:
        if (
            items[column].isna().any()
            or items[column].str.strip().eq("").any()
        ):
            raise ValueError(f"Hay ítems sin {column}.")

    # Al quitar las ofertas, cada ítem debe tener datos consistentes.
    if items.duplicated(key_columns).any():
        raise ValueError(
            "Hay datos contradictorios para un mismo ítem."
        )

    items = items.sort_values(key_columns).reset_index(drop=True)

    # Posición interna; conservamos también el correlativo original.
    items["ItemPosicion"] = (
        items.groupby("CodigoExterno", sort=False).cumcount() + 1
    )

    items["OrigenDatos"] = "chilecompra_descarga_mensual"
    items["ArchivoOrigen"] = zip_path.name

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--month", type=int, choices=range(1, 13), required=True
    )
    args = parser.parse_args()

    catalog_path = PROCESSED_DIR / (
        f"monthly_catalog_{args.year}_{args.month:02d}.csv"
    )
    output_path = PROCESSED_DIR / (
        f"monthly_items_{args.year}_{args.month:02d}.csv"
    )

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Primero debes generar el catálogo: {catalog_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            f"El archivo de ítems ya existe: {output_path}"
        )

    catalog = pd.read_csv(
        catalog_path,
        dtype="string",
        usecols=["CodigoExterno"],
    )

    items = build_monthly_items(args.year, args.month)

    catalog_codes = set(catalog["CodigoExterno"])
    item_codes = set(items["CodigoExterno"])

    if item_codes - catalog_codes:
        raise ValueError("Hay ítems sin licitación en el catálogo.")

    if catalog_codes - item_codes:
        raise ValueError("Hay licitaciones del catálogo sin ítems.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    items.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Ítems distintos: {len(items):,}")
    print(f"Licitaciones con ítems: {len(item_codes):,}")
    print("Validación de códigos: correcta")
    print(f"\nArchivo generado: {output_path}")


if __name__ == "__main__":
    main()