import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from src.api.mercado_publico_client import get_licitaciones_por_fecha


# Raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_daily_tenders(
    fecha: str,
    sobrescribir: bool = False,
) -> Path:
    """
    Consulta las licitaciones publicadas en una fecha
    y guarda la respuesta original en data/raw.

    Si el archivo ya existe, la extracción se omite,
    salvo que sobrescribir sea True.
    """
    output_dir = PROJECT_ROOT / "data" / "raw" / "licitaciones"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"licitaciones_{fecha}.json"

    if output_path.exists() and not sobrescribir:
        print(f"La fecha {fecha} ya fue extraída.")
        print(f"Archivo existente: {output_path}")
        print("Extracción omitida.")

        return output_path

    datos = get_licitaciones_por_fecha(fecha)

    with output_path.open("w", encoding="utf-8") as archivo:
        json.dump(
            datos,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    cantidad = datos.get("Cantidad", 0)

    print(f"Fecha extraída: {fecha}")
    print(f"Licitaciones obtenidas: {cantidad}")
    print(f"Archivo guardado en: {output_path}")

    return output_path


def parse_arguments() -> argparse.Namespace:
    """
    Lee los argumentos entregados desde la terminal.
    Si no se indica una fecha, utiliza el día anterior.
    """
    fecha_ayer = (date.today() - timedelta(days=1)).strftime("%d%m%Y")

    parser = argparse.ArgumentParser(
        description="Extrae licitaciones diarias desde Mercado Público."
    )

    parser.add_argument(
        "--fecha",
        default=fecha_ayer,
        help=(
            "Fecha que se consultará en formato ddmmaaaa. "
            "Si se omite, se utiliza el día anterior."
        ),
    )

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Reemplaza el archivo si la fecha ya había sido extraída.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    argumentos = parse_arguments()

    extract_daily_tenders(
        fecha=argumentos.fecha,
        sobrescribir=argumentos.sobrescribir,
    )