import json
import os
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv


# La raíz del proyecto se encuentra dos niveles sobre este archivo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carga las variables privadas guardadas en .env.
load_dotenv(PROJECT_ROOT / ".env")

ticket = os.getenv("MERCADO_PUBLICO_TICKET")

if not ticket:
    raise RuntimeError(
        "No se encontró MERCADO_PUBLICO_TICKET en el archivo .env"
    )

# Consultaremos las licitaciones publicadas durante el día anterior.
fecha_consulta = (date.today() - timedelta(days=1)).strftime("%d%m%Y")

url = (
    "https://api.mercadopublico.cl/"
    "servicios/v1/publico/licitaciones.json"
)

parametros = {
    "fecha": fecha_consulta,
    "ticket": ticket,
}

# Realiza la consulta a la API.
respuesta = requests.get(url, params=parametros, timeout=30)

print(f"Estado HTTP: {respuesta.status_code}")

# Detiene el programa si la API devuelve un error HTTP.
respuesta.raise_for_status()

# Convierte la respuesta JSON en un diccionario de Python.
datos = respuesta.json()

print(f"Fecha consultada: {fecha_consulta}")
print(f"Cantidad informada por la API: {datos.get('Cantidad', 0)}")

# Obtiene la lista de licitaciones sin generar error si la clave no existe.
licitaciones = datos.get("Listado", [])

# Muestra como máximo cinco licitaciones.
for licitacion in licitaciones[:5]:
    codigo = licitacion.get("CodigoExterno", "Sin código")
    nombre = licitacion.get("Nombre", "Sin nombre")
    print(f"- {codigo}: {nombre}")

# Define la carpeta donde guardaremos la muestra pública.
sample_dir = PROJECT_ROOT / "data" / "sample"
sample_dir.mkdir(parents=True, exist_ok=True)

# Construye el nombre del archivo usando la fecha consultada.
sample_path = sample_dir / f"licitaciones_{fecha_consulta}.json"

# Guarda la respuesta pública como un archivo JSON legible.
with sample_path.open("w", encoding="utf-8") as archivo:
    json.dump(
        datos,
        archivo,
        ensure_ascii=False,
        indent=2,
    )

print(f"Muestra guardada en: {sample_path}")