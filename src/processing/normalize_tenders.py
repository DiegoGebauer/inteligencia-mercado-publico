import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAILS_DIR = PROJECT_ROOT / "data" / "raw" / "details"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_tender_records(details_dir: Path) -> list[dict]:
    records = []

    for json_path in sorted(details_dir.glob("*.json")):
        with json_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        records.extend(payload["Listado"])

    if not records:
        raise RuntimeError(
            f"No se encontraron registros de licitaciones en {details_dir}"
        )

    return records

def build_tender_rows(
    tender_records: list[dict],
) -> list[dict]:
    tender_rows = []

    for tender in tender_records:
        tender_row = tender.copy()
        tender_row.pop("Items", None)
        tender_rows.append(tender_row)

    return tender_rows


def build_tender_item_rows(
    tender_records: list[dict],
) -> list[dict]:
    item_rows = []

    for tender in tender_records:
        tender_code = tender["CodigoExterno"]
        items = tender["Items"]["Listado"]

        for position, item in enumerate(items, start=1):
            item_row = item.copy()
            item_row["CodigoExterno"] = tender_code
            # La API puede repetir Correlativo; conservamos su valor original.
            item_row["ItemPosicion"] = position
            item_rows.append(item_row)

    return item_rows


def build_tenders_dataframe(
    tender_rows: list[dict],
) -> pd.DataFrame:
    return pd.json_normalize(tender_rows, sep="_")


def build_tender_items_dataframe(
    tender_item_rows: list[dict],
) -> pd.DataFrame:
    return pd.json_normalize(tender_item_rows, sep="_")


def convert_tender_date_columns(
    tenders_df: pd.DataFrame,
) -> pd.DataFrame:
    normalized_df = tenders_df.copy()

    date_columns = [
        column
        for column in normalized_df.columns
        if column.startswith("Fechas_Fecha")
        and column != "Fechas_FechasUsuario"
    ]

    for column in date_columns:
        normalized_df[column] = pd.to_datetime(
            normalized_df[column],
            format="mixed",
            errors="coerce",
        )

    return normalized_df


def convert_tender_numeric_columns(
    tenders_df: pd.DataFrame,
) -> pd.DataFrame:
    normalized_df = tenders_df.copy()

    normalized_df["DiasCierreLicitacion"] = pd.to_numeric(
        normalized_df["DiasCierreLicitacion"],
        errors="coerce",
    ).astype("Int64")

    normalized_df["MontoEstimado"] = pd.to_numeric(
        normalized_df["MontoEstimado"],
        errors="coerce",
    )

    return normalized_df


def convert_tender_item_identifier_columns(
    tender_items_df: pd.DataFrame,
) -> pd.DataFrame:
    normalized_df = tender_items_df.copy()

    identifier_columns = [
        "CodigoExterno",
        "CodigoProducto",
        "CodigoCategoria",
    ]

    for column in identifier_columns:
        normalized_df[column] = normalized_df[column].astype(
            "string"
        )

    return normalized_df


def save_processed_data(
    tenders_df: pd.DataFrame,
    tender_items_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tenders_path = output_dir / "tenders.csv"
    tender_items_path = output_dir / "tender_items.csv"

    tenders_df.to_csv(
        tenders_path,
        index=False,
        encoding="utf-8",
    )

    tender_items_df.to_csv(
        tender_items_path,
        index=False,
        encoding="utf-8",
    )

    return tenders_path, tender_items_path


def validate_normalized_data(
    tenders_df: pd.DataFrame,
    tender_items_df: pd.DataFrame,
) -> None:
    duplicated_tenders = tenders_df[
        "CodigoExterno"
    ].duplicated().sum()

    duplicated_items = tender_items_df.duplicated(
        subset=["CodigoExterno", "ItemPosicion"]
    ).sum()

    missing_tender_ids = tenders_df[
        "CodigoExterno"
    ].isna().sum()

    missing_item_tender_ids = tender_items_df[
        "CodigoExterno"
    ].isna().sum()

    missing_item_positions = tender_items_df[
        "ItemPosicion"
    ].isna().sum()

    tender_ids = set(
        tenders_df["CodigoExterno"]
        .astype("string")
        .dropna()
    )

    item_tender_ids = set(
        tender_items_df["CodigoExterno"]
        .astype("string")
        .dropna()
    )

    orphan_tender_ids = item_tender_ids - tender_ids

    errors = []

    if duplicated_tenders:
        errors.append(
            f"{duplicated_tenders} licitaciones duplicadas"
        )

    if duplicated_items:
        errors.append(
            f"{duplicated_items} ítems duplicados"
        )

    if missing_tender_ids:
        errors.append(
            f"{missing_tender_ids} licitaciones sin código"
        )

    if missing_item_tender_ids:
        errors.append(
            f"{missing_item_tender_ids} ítems sin código"
        )

    if missing_item_positions:
        errors.append(
            f"{missing_item_positions} ítems sin posición interna"
        )

    if orphan_tender_ids:
        errors.append(
            f"{len(orphan_tender_ids)} relaciones huérfanas"
        )

    if errors:
        raise ValueError(
            "Validación fallida: " + "; ".join(errors)
        )

    print("\nValidación de integridad:")
    print("- Licitaciones duplicadas: 0")
    print("- Ítems duplicados: 0")
    print("- Relaciones huérfanas: 0")
    print("- Resultado: correcto")


if __name__ == "__main__":
    tender_records = load_tender_records(DETAILS_DIR)

    tender_rows = build_tender_rows(tender_records)
    tender_item_rows = build_tender_item_rows(tender_records)

    tenders_df = build_tenders_dataframe(tender_rows)
    tender_items_df = build_tender_items_dataframe(
        tender_item_rows
    )

    print(f"Licitaciones cargadas: {len(tender_records)}")
    print(f"Filas para tenders: {len(tender_rows)}")
    print(
        f"¿Items quedó en tenders?: "
        f"{'Items' in tender_rows[0]}"
    )
    print(f"Filas para tender_items: {len(tender_item_rows)}")

    print(
        f"Primera relación licitación-ítem: "
        f"{tender_item_rows[0]['CodigoExterno']} -> "
        f"{tender_item_rows[0]['NombreProducto']}"
    )

    print(f"Filas del DataFrame tenders: {len(tenders_df)}")
    print(
        "Columna de comprador creada:",
        "Comprador_NombreOrganismo" in tenders_df.columns,
    )
    print(
        "Columna de fecha creada:",
        "Fechas_FechaPublicacion" in tenders_df.columns,
    )

    print(
        f"Filas del DataFrame tender_items: "
        f"{len(tender_items_df)}"
    )
    print(
        "Columna de relación creada:",
        "CodigoExterno" in tender_items_df.columns,
    )

    print(
        tender_items_df[
            [
                "CodigoExterno",
                "Correlativo",
                "NombreProducto",
            ]
        ].head(10).to_string(index=False)
    )

    tender_columns_to_inspect = [
        "CodigoExterno",
        "CodigoEstado",
        "DiasCierreLicitacion",
        "MontoEstimado",
        "Fechas_FechaPublicacion",
        "Fechas_FechaCierre",
    ]

    item_columns_to_inspect = [
        "CodigoExterno",
        "CodigoProducto",
        "CodigoCategoria",
        "Cantidad",
    ]

    print("\nTipos principales de tenders antes de convertir:")
    print(tenders_df[tender_columns_to_inspect].dtypes)

    print("\nValores principales de tenders:")
    print(
        tenders_df[
            tender_columns_to_inspect
        ].to_string(index=False)
    )

    print("\nTipos principales de tender_items:")
    print(tender_items_df[item_columns_to_inspect].dtypes)

    tender_items_df = convert_tender_item_identifier_columns(
        tender_items_df
    )

    print("\nTipos de identificadores después de convertir:")
    print(
        tender_items_df[
            [
                "CodigoExterno",
                "CodigoProducto",
                "CodigoCategoria",
            ]
        ].dtypes
    )

    tenders_df = convert_tender_date_columns(tenders_df)

    print("\nTipos después de convertir fechas:")
    print(
        tenders_df[
            [
                "Fechas_FechaPublicacion",
                "Fechas_FechaCierre",
            ]
        ].dtypes
    )

    tenders_df = convert_tender_numeric_columns(tenders_df)

    print("\nTipo después de convertir días de cierre:")
    print(tenders_df["DiasCierreLicitacion"].dtype)

    print("\nValores de días de cierre:")
    print(
        tenders_df[
            [
                "CodigoExterno",
                "DiasCierreLicitacion",
            ]
        ].to_string(index=False)
    )

    validate_normalized_data(
        tenders_df,
        tender_items_df,
    )

    tenders_path, tender_items_path = save_processed_data(
        tenders_df,
        tender_items_df,
        PROCESSED_DIR,
    )

    print("\nArchivos generados:")
    print(f"- {tenders_path}")
    print(f"- {tender_items_path}")