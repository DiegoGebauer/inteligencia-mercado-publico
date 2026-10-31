import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        choices=range(1, 13),
        required=True,
    )
    args = parser.parse_args()

    months = sorted(set(args.months))
    output_path = PROCESSED_DIR / (
        f"monthly_texts_{args.year}_combined.csv"
    )

    if output_path.exists():
        raise FileExistsError(f"El archivo ya existe: {output_path}")

    parts = []

    for month in months:
        input_path = PROCESSED_DIR / (
            f"monthly_texts_{args.year}_{month:02d}.csv"
        )

        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")

        frame = pd.read_csv(input_path, dtype="string")

        publication_dates = pd.to_datetime(
            frame["FechaPublicacion"],
            format="%Y-%m-%d",
            errors="raise",
        )

        belongs_to_month = (
            publication_dates.dt.year.eq(args.year)
            & publication_dates.dt.month.eq(month)
        )

        if not belongs_to_month.all():
            raise ValueError(
                f"Hay fechas fuera del periodo en {input_path.name}."
            )

        frame["PeriodoPublicacion"] = (
            publication_dates.dt.strftime("%Y-%m")
        )
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)

    if (
        combined["CodigoExterno"].isna().any()
        or combined["CodigoExterno"].str.strip().eq("").any()
    ):
        raise ValueError("Hay licitaciones sin código.")

    if combined["CodigoExterno"].duplicated().any():
        raise ValueError("Hay códigos repetidos entre los archivos.")

    if (
        combined["TextoLicitacion"]
        .fillna("")
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError("Hay licitaciones sin texto.")

    combined = combined.sort_values(
        ["FechaPublicacion", "CodigoExterno"]
    ).reset_index(drop=True)

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Licitaciones reunidas: {len(combined):,}")
    print("\nDistribución por mes:")
    print(
        combined["PeriodoPublicacion"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print(f"\nArchivo generado: {output_path}")


if __name__ == "__main__":
    main()