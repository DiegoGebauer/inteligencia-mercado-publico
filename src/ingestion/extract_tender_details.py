import csv
import json
import random
import time
from pathlib import Path

from src.api.mercado_publico_client import (
    get_licitacion_por_codigo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LISTINGS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "licitaciones"
)

DETAILS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "details"
)

KEYWORD_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "keyword_candidates.csv"
)


DETAILS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_tender_codes(
    fecha: str,
) -> list[str]:
    """
    Lee un archivo diario y devuelve sus códigos.
    """

    input_path = (
        LISTINGS_DIR
        / f"licitaciones_{fecha}.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo diario: "
            f"{input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    codes = [
        tender.get("CodigoExterno")
        for tender in payload.get("Listado", [])
        if tender.get("CodigoExterno")
    ]

    return codes


def load_tender_sample(
    fecha: str,
    per_state: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    Selecciona una muestra reproducible,
    con un máximo por código de estado.
    """

    input_path = (
        LISTINGS_DIR
        / f"licitaciones_{fecha}.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo diario: "
            f"{input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    tenders_by_state = {}

    for tender in payload.get("Listado", []):
        tender_code = tender.get(
            "CodigoExterno"
        )

        state_code = tender.get(
            "CodigoEstado"
        )

        if (
            tender_code
            and state_code is not None
        ):
            tenders_by_state.setdefault(
                state_code,
                [],
            ).append(tender)

    random_generator = random.Random(seed)
    sample = []

    for state_code in sorted(tenders_by_state):
        candidates = tenders_by_state[
            state_code
        ]

        selected = random_generator.sample(
            candidates,
            k=min(
                per_state,
                len(candidates),
            ),
        )

        sample.extend(selected)

    return sample


def load_candidate_codes(
    candidate_path: Path,
) -> list[str]:
    """
    Lee los códigos seleccionados desde
    keyword_candidates.csv.
    """

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de "
            f"candidatos: {candidate_path}"
        )

    with candidate_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if (
            reader.fieldnames is None
            or "CodigoExterno"
            not in reader.fieldnames
        ):
            raise ValueError(
                "El CSV no contiene la columna "
                "'CodigoExterno'."
            )

        codes = []

        for row in reader:
            code = (
                row.get("CodigoExterno")
                or ""
            ).strip()

            if code:
                codes.append(code)

    unique_codes = list(
        dict.fromkeys(codes)
    )

    if not unique_codes:
        raise RuntimeError(
            "El archivo de candidatos no "
            "contiene códigos válidos."
        )

    return unique_codes


def save_tender_detail(
    codigo: str,
    sobrescribir: bool = False,
) -> Path:
    """
    Consulta y guarda la respuesta original
    de una licitación.
    """

    output_path = (
        DETAILS_DIR / f"{codigo}.json"
    )

    if (
        output_path.exists()
        and not sobrescribir
    ):
        print(
            f"Detalle ya existente: {codigo}"
        )

        return output_path

    payload = get_licitacion_por_codigo(
        codigo
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Detalle guardado: {codigo}")

    return output_path


def extract_codes(
    codes: list[str],
    sobrescribir: bool = False,
    delay_seconds: float = 2.0,
) -> None:
    """
    Descarga los detalles de una lista
    explícita de códigos.
    """

    unique_codes = list(
        dict.fromkeys(codes)
    )

    print(
        f"Licitaciones seleccionadas: "
        f"{len(unique_codes)}"
    )

    processed = 0
    skipped = 0
    errors = []

    for code in unique_codes:
        output_path = (
            DETAILS_DIR / f"{code}.json"
        )

        if (
            output_path.exists()
            and not sobrescribir
        ):
            print(
                f"Omitida, ya existe: {code}"
            )

            skipped += 1
            continue

        try:
            save_tender_detail(
                code,
                sobrescribir=sobrescribir,
            )

            processed += 1

        except RuntimeError as error:
            print(
                f"Error en {code}: {error}"
            )

            errors.append(code)

        finally:
            time.sleep(delay_seconds)

    print("\nResumen:")
    print(
        f"- Total seleccionadas: "
        f"{len(unique_codes)}"
    )
    print(
        f"- Nuevas procesadas: {processed}"
    )
    print(
        f"- Omitidas existentes: {skipped}"
    )
    print(
        f"- Errores: {len(errors)}"
    )

    if errors:
        print("\nCódigos con error:")

        for code in errors:
            print(f"- {code}")


def extract_tender_details(
    fecha: str,
    sobrescribir: bool = False,
    per_state: int | None = None,
    seed: int = 42,
    delay_seconds: float = 2.0,
) -> None:
    """
    Extrae detalles desde un listado diario,
    completo o mediante muestra por estado.
    """

    if per_state is None:
        codes = load_tender_codes(fecha)

        print(
            "Modo de extracción: "
            "listado completo"
        )

    else:
        sample = load_tender_sample(
            fecha=fecha,
            per_state=per_state,
            seed=seed,
        )

        codes = [
            tender["CodigoExterno"]
            for tender in sample
        ]

        print(
            "Modo de extracción: "
            f"muestra de hasta "
            f"{per_state} por estado"
        )

    extract_codes(
        codes=codes,
        sobrescribir=sobrescribir,
        delay_seconds=delay_seconds,
    )


if __name__ == "__main__":
    candidate_codes = load_candidate_codes(
        KEYWORD_CANDIDATES_PATH
    )

    print(
        "Modo de extracción: "
        "candidatos seleccionados por keywords"
    )

    extract_codes(
        codes=candidate_codes,
        sobrescribir=False,
        delay_seconds=2.0,
    )