import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# Determina la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Endpoint público de licitaciones.
API_URL = (
    "https://api.mercadopublico.cl/"
    "servicios/v1/publico/licitaciones.json"
)

DEFAULT_TIMEOUT = 30


def _retry_after_seconds(value: str | None) -> int | None:
    """Interpreta segundos o una fecha HTTP; no conserva el encabezado crudo."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.isascii() and value.isdigit() and len(value) <= 12:
        return int(value)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        return max(0, math.ceil((date - datetime.now(timezone.utc)).total_seconds()))
    except (TypeError, ValueError, OverflowError):
        return None


class MercadoPublicoAPIError(RuntimeError):
    """Fallo compatible con RuntimeError, con diagnóstico HTTP sin credenciales."""

    def __init__(self, status_code=None, retry_after_seconds=None):
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        status = status_code if status_code is not None else "sin respuesta"
        super().__init__(f"Error al consultar Mercado Público. Estado: {status}")


def _load_ticket() -> str:
    """Carga el ticket desde el archivo local .env."""

    load_dotenv(PROJECT_ROOT / ".env")

    ticket = os.getenv("MERCADO_PUBLICO_TICKET")

    if not ticket:
        raise RuntimeError(
            "No se encontró MERCADO_PUBLICO_TICKET en el archivo .env"
        )

    return ticket


def _request_licitaciones(
    query_params: dict[str, str],
) -> dict[str, Any]:
    """Realiza una consulta segura a la API de licitaciones."""

    parametros = {
        **query_params,
        "ticket": _load_ticket(),
    }

    try:
        respuesta = requests.get(
            API_URL,
            params=parametros,
            timeout=DEFAULT_TIMEOUT,
        )

        respuesta.raise_for_status()

    except requests.RequestException as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else None
        )
        retry_after = _retry_after_seconds(error.response.headers.get("Retry-After")) if error.response is not None else None

        # No mostramos la URL porque contiene el ticket.
        raise MercadoPublicoAPIError(status_code, retry_after) from None

    try:
        return respuesta.json()

    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            "La API respondió, pero el contenido no es un JSON válido."
        ) from None


def get_licitaciones_por_fecha(
    fecha: str,
) -> dict[str, Any]:
    """Obtiene el listado básico de licitaciones de una fecha."""

    if len(fecha) != 8 or not fecha.isdigit():
        raise ValueError(
            "La fecha debe utilizar el formato ddmmaaaa."
        )

    return _request_licitaciones(
        {
            "fecha": fecha,
        }
    )


def get_licitacion_por_codigo(
    codigo: str,
) -> dict[str, Any]:
    """Obtiene el detalle de una licitación por su código."""

    codigo_limpio = codigo.strip()

    if not codigo_limpio:
        raise ValueError(
            "El código de la licitación no puede estar vacío."
        )

    return _request_licitaciones(
        {
            "codigo": codigo_limpio,
        }
    )
