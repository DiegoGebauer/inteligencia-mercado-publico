import json
from pathlib import Path

from src.api.mercado_publico_client import get_licitacion_por_codigo


# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpetas de entrada y salida
LISTINGS_DIR = PROJECT_ROOT / "data" / "raw" / "licitaciones"
DETAILS_DIR = PROJECT_ROOT / "data" / "raw" / "details"

# Crea la carpeta de detalles si todavía no existe
DETAILS_DIR.mkdir(parents=True, exist_ok=True)


def load_tender_codes(fecha: str) -> list[str]:
    """
    Lee el archivo diario de licitaciones y devuelve
    los códigos externos encontrados.
    """

    input_path = LISTINGS_DIR / f"licitaciones_{fecha}.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo diario: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    licitaciones = datos.get("Listado", [])

    codigos = [
        licitacion.get("CodigoExterno")
        for licitacion in licitaciones
        if licitacion.get("CodigoExterno")
    ]

    return codigos

def save_tender_detail(
    codigo: str,
    sobrescribir: bool = False,
) -> Path:
    """
    Consulta el detalle de una licitación y guarda
    la respuesta original en data/raw/details.
    """

    output_path = DETAILS_DIR / f"{codigo}.json"

    if output_path.exists() and not sobrescribir:
        print(f"Detalle ya existente: {codigo}")
        return output_path

    datos = get_licitacion_por_codigo(codigo)

    with output_path.open("w", encoding="utf-8") as archivo:
        json.dump(
            datos,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Detalle guardado: {codigo}")

    return output_path


def extract_tender_details(
    fecha: str,
    sobrescribir: bool = False,
) -> None:
    """
    Obtiene y guarda el detalle de todas las licitaciones
    encontradas en el archivo diario de una fecha.
    """

    codigos = load_tender_codes(fecha)

    print(f"Licitaciones encontradas: {len(codigos)}")

    procesadas = 0
    omitidas = 0
    errores = []

    for codigo in codigos:
        output_path = DETAILS_DIR / f"{codigo}.json"

        if output_path.exists() and not sobrescribir:
            print(f"Omitida, ya existe: {codigo}")
            omitidas += 1
            continue

        try:
            save_tender_detail(
                codigo,
                sobrescribir=sobrescribir,
            )
            procesadas += 1

        except RuntimeError as error:
            print(f"Error en {codigo}: {error}")
            errores.append(codigo)

    print("\nResumen:")
    print(f"- Total encontradas: {len(codigos)}")
    print(f"- Nuevas procesadas: {procesadas}")
    print(f"- Omitidas existentes: {omitidas}")
    print(f"- Errores: {len(errores)}")

    if errores:
        print("\nCódigos con error:")
        for codigo in errores:
            print(f"- {codigo}")


if __name__ == "__main__":
    fecha = "18092026"

    extract_tender_details(fecha)