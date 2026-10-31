import argparse
import zipfile
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONTHLY_DIR = PROJECT_ROOT / "data" / "raw" / "monthly"

BASE_URL = "https://transparenciachc.blob.core.windows.net/lic-da"


def download_monthly_file(year: int, month: int) -> None:
    filename = f"{year}-{month}.zip"
    output_path = MONTHLY_DIR / filename
    temporary_path = MONTHLY_DIR / f"{filename}.part"

    if output_path.exists():
        if not zipfile.is_zipfile(output_path):
            raise ValueError(
                f"El archivo existente no es un ZIP válido: {output_path}"
            )

        print(f"Ya disponible: {filename}")
        return

    url = f"{BASE_URL}/{filename}"
    print(f"Descargando: {filename}", flush=True)

    try:
        # Guardamos primero como .part para distinguir descargas incompletas.
        with requests.get(
            url,
            stream=True,
            timeout=(20, 180),
        ) as response:
            response.raise_for_status()

            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        if not zipfile.is_zipfile(temporary_path):
            raise ValueError(
                f"La descarga de {filename} no contiene un ZIP válido."
            )

        with zipfile.ZipFile(temporary_path) as archive:
            csv_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if len(csv_files) != 1:
                raise ValueError(
                    f"{filename} debe contener exactamente un CSV."
                )

        temporary_path.rename(output_path)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Guardado: {filename} ({size_mb:.1f} MB)")

    except Exception:
        # Eliminamos únicamente la descarga temporal de este archivo.
        temporary_path.unlink(missing_ok=True)
        raise


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

    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    for month in sorted(set(args.months)):
        download_monthly_file(args.year, month)

    print("\nTodos los meses solicitados están disponibles.")


if __name__ == "__main__":
    main()