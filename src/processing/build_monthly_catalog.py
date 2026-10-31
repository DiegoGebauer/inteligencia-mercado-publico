import argparse
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONTHLY_DIR = PROJECT_ROOT / "data" / "raw" / "monthly"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COLUMNS = [
    "CodigoExterno",
    "Nombre",
    "Descripcion",
    "FechaPublicacion",
    "Estado",
    "Tipo",
    "CodigoTipo",
]


def build_monthly_catalog(year: int, month: int) -> pd.DataFrame:
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
            # Leemos por bloques porque el CSV descomprimido es grande.
            for chunk in pd.read_csv(
                file,
                sep=";",
                encoding="latin1",
                dtype="string",
                usecols=COLUMNS,
                chunksize=50_000,
            ):
                parts.append(chunk.drop_duplicates())

    catalog = pd.concat(parts, ignore_index=True).drop_duplicates()

    if (
        catalog["CodigoExterno"].isna().any()
        or catalog["CodigoExterno"].str.strip().eq("").any()
    ):
        raise ValueError("Hay licitaciones sin código.")

    # Si un código conserva varias filas, sus datos son diferentes.
    if catalog["CodigoExterno"].duplicated().any():
        raise ValueError(
            "Hay datos contradictorios para un mismo código."
        )

    publication_dates = pd.to_datetime(
        catalog["FechaPublicacion"],
        format="%Y-%m-%d",
        errors="raise",
    )

    belongs_to_month = (
        publication_dates.dt.year.eq(year)
        & publication_dates.dt.month.eq(month)
    )

    if not belongs_to_month.all():
        raise ValueError(
            "El archivo contiene fechas fuera del mes solicitado."
        )

    catalog["FechaPublicacion"] = publication_dates.dt.strftime(
        "%Y-%m-%d"
    )
    catalog["OrigenDatos"] = "chilecompra_descarga_mensual"
    catalog["ArchivoOrigen"] = zip_path.name

    return catalog.sort_values(
        ["FechaPublicacion", "CodigoExterno"]
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, choices=range(1, 13), required=True)
    args = parser.parse_args()

    output_path = PROCESSED_DIR / (
        f"monthly_catalog_{args.year}_{args.month:02d}.csv"
    )

    if output_path.exists():
        raise FileExistsError(
            f"El catálogo ya existe: {output_path}"
        )

    catalog = build_monthly_catalog(args.year, args.month)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Licitaciones distintas: {len(catalog):,}")
    print("\nDistribución por tipo:")
    print(catalog["Tipo"].value_counts().to_string())
    print(f"\nArchivo generado: {output_path}")


if __name__ == "__main__":
    main()