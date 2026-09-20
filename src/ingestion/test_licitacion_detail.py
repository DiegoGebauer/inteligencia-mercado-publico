import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Determina la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carga las variables privadas desde .env.
load_dotenv(PROJECT_ROOT / ".env")

ticket = os.getenv("MERCADO_PUBLICO_TICKET")

if not ticket:
    raise RuntimeError(
        "No se encontró MERCADO_PUBLICO_TICKET en el archivo .env"
    )

# Licitación relacionada con consultoría, elegida desde la muestra diaria.
codigo_licitacion = "835727-29-LE26"

url = (
    "https://api.mercadopublico.cl/"
    "servicios/v1/publico/licitaciones.json"
)

parametros = {
    "codigo": codigo_licitacion,
    "ticket": ticket,
}

# Consulta el detalle de una licitación específica.
respuesta = requests.get(url, params=parametros, timeout=30)

print(f"Estado HTTP: {respuesta.status_code}")

# Detiene el programa si la API devuelve un error HTTP.
respuesta.raise_for_status()

# Convierte la respuesta JSON en un diccionario de Python.
datos = respuesta.json()

print(f"Cantidad informada por la API: {datos.get('Cantidad', 0)}")

licitaciones = datos.get("Listado", [])

if not licitaciones:
    raise RuntimeError(
        f"No se encontró la licitación {codigo_licitacion}"
    )

# La consulta por código debería devolver una sola licitación.
licitacion = licitaciones[0]

print(f"Código: {licitacion.get('CodigoExterno', 'Sin código')}")
print(f"Nombre: {licitacion.get('Nombre', 'Sin nombre')}")
print(f"Estado: {licitacion.get('CodigoEstado', 'Sin estado')}")

# Muestra los campos generales y sus tipos.
print("\nCampos disponibles en el detalle:")

for campo, valor in licitacion.items():
    tipo = type(valor).__name__
    print(f"- {campo}: {tipo}")

# Campos que contienen estructuras internas.
campos_anidados = [
    "Comprador",
    "Fechas",
    "Items",
    "Adjudicacion",
]

# Examina los nombres y tipos de los campos internos.
for nombre_campo in campos_anidados:
    contenido = licitacion.get(nombre_campo)

    print(f"\nEstructura de {nombre_campo}:")

    if isinstance(contenido, dict):
        for campo, valor in contenido.items():
            tipo = type(valor).__name__
            print(f"- {campo}: {tipo}")
    else:
        tipo = type(contenido).__name__
        print(f"- Sin estructura disponible: {tipo}")

# Obtiene el listado de productos o servicios solicitados.
items = licitacion.get("Items", {})
listado_items = items.get("Listado", [])

print("\nEstructura del primer elemento de Items.Listado:")

# Examina solamente los campos del primer item.
if listado_items:
    primer_item = listado_items[0]

    for campo, valor in primer_item.items():
        tipo = type(valor).__name__
        print(f"- {campo}: {tipo}")
else:
    print("- La licitación no contiene items")

# Guarda el detalle público para examinarlo posteriormente.
sample_dir = PROJECT_ROOT / "data" / "sample"
sample_dir.mkdir(parents=True, exist_ok=True)

sample_path = sample_dir / f"licitacion_{codigo_licitacion}.json"

with sample_path.open("w", encoding="utf-8") as archivo:
    json.dump(
        datos,
        archivo,
        ensure_ascii=False,
        indent=2,
    )

print(f"\nDetalle guardado en: {sample_path}")