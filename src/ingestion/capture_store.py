import json
from pathlib import Path


def guardar_captura(payload: dict, destino: Path) -> Path:
    """Guarda JSON UTF-8 sin reemplazar un archivo existente y devuelve su ruta."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    json_texto = json.dumps(payload, ensure_ascii=False, indent=2)
    if destino.exists():
        raise FileExistsError(f'El archivo {destino} ya existe.')
    with destino.open('x', encoding='utf-8') as f:
        f.write(json_texto)
    return destino