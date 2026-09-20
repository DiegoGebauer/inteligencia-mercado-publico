from pathlib import Path

import yaml


# Determina la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

profile_path = PROJECT_ROOT / "config" / "company_profile.yaml"

if not profile_path.exists():
    raise FileNotFoundError(
        f"No se encontró el perfil empresarial: {profile_path}"
    )

# Lee el archivo YAML y lo convierte en un diccionario de Python.
with profile_path.open("r", encoding="utf-8") as archivo:
    profile = yaml.safe_load(archivo)

if not isinstance(profile, dict):
    raise ValueError(
        "El perfil empresarial debe contener un diccionario YAML."
    )

# Secciones mínimas que esperamos encontrar.
required_sections = [
    "profile_version",
    "company",
    "relevant_services",
    "excluded_services",
    "territorial_priority",
]

missing_sections = [
    section
    for section in required_sections
    if section not in profile
]

if missing_sections:
    raise ValueError(
        f"Faltan secciones obligatorias: {missing_sections}"
    )

relevant_services = profile["relevant_services"]
excluded_services = profile["excluded_services"]
territorial_priority = profile["territorial_priority"]

# Cuenta todos los términos positivos del perfil.
positive_keyword_count = sum(
    len(service.get("keywords", []))
    for service in relevant_services.values()
)

# Cuenta los términos excluidos o penalizados.
excluded_keyword_count = len(
    excluded_services.get("keywords", [])
)

priority_regions = territorial_priority.get(
    "priority_regions",
    [],
)

print("Perfil empresarial validado correctamente.")
print(f"Versión: {profile['profile_version']}")
print(f"Empresa: {profile['company'].get('name')}")
print(f"Grupos de servicios: {len(relevant_services)}")
print(f"Términos relevantes: {positive_keyword_count}")
print(f"Términos excluidos: {excluded_keyword_count}")
print(
    "Cobertura:",
    territorial_priority.get("coverage"),
)
print(
    "Regiones prioritarias:",
    ", ".join(priority_regions),
)