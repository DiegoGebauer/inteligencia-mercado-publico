import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LISTINGS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "licitaciones"
)

PROFILE_PATH = (
    PROJECT_ROOT / "config" / "company_profile.yaml"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

KEYWORD_CANDIDATES_PATH = (
    PROCESSED_DIR / "keyword_candidates.csv"
)


def load_daily_tenders(
    listings_dir: Path,
) -> pd.DataFrame:
    records = []

    for json_path in sorted(
        listings_dir.glob("licitaciones_*.json")
    ):
        fecha_texto = json_path.stem.removeprefix(
            "licitaciones_"
        )

        fecha_consulta = pd.to_datetime(
            fecha_texto,
            format="%d%m%Y",
            errors="raise",
        )

        with json_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        for tender in payload.get("Listado", []):
            tender_row = tender.copy()
            tender_row["FechaConsulta"] = fecha_consulta
            records.append(tender_row)

    if not records:
        raise RuntimeError(
            f"No se encontraron licitaciones en "
            f"{listings_dir}"
        )

    return pd.DataFrame(records)


def build_unique_candidate_pool(
    daily_tenders: pd.DataFrame,
) -> pd.DataFrame:
    candidates = daily_tenders.dropna(
        subset=["CodigoExterno"]
    ).copy()

    candidates = candidates.sort_values(
        [
            "FechaConsulta",
            "CodigoExterno",
        ]
    )

    candidates = candidates.drop_duplicates(
        subset=["CodigoExterno"],
        keep="first",
    )

    return candidates.reset_index(drop=True)


def load_profile_keywords(
    profile_path: Path,
) -> tuple[list[str], list[str]]:
    with profile_path.open(encoding="utf-8") as file:
        profile = yaml.safe_load(file)

    relevant_keywords = []

    for service in profile["relevant_services"].values():
        relevant_keywords.extend(
            service.get("keywords", [])
        )

    excluded_keywords = profile[
        "excluded_services"
    ].get(
        "keywords",
        [],
    )

    return relevant_keywords, excluded_keywords


def normalize_search_text(
    value: object,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return text.strip()


def find_keyword_matches(
    text: object,
    keywords: list[str],
) -> list[str]:
    normalized_text = normalize_search_text(text)
    padded_text = f" {normalized_text} "

    matches = []

    for keyword in keywords:
        normalized_keyword = normalize_search_text(
            keyword
        )

        padded_keyword = (
            f" {normalized_keyword} "
        )

        if padded_keyword in padded_text:
            matches.append(keyword)

    return matches


def add_keyword_signals(
    candidates: pd.DataFrame,
    relevant_keywords: list[str],
    excluded_keywords: list[str],
) -> pd.DataFrame:
    candidates = candidates.copy()

    candidates["KeywordsRelevantes"] = (
        candidates["Nombre"].apply(
            lambda name: find_keyword_matches(
                name,
                relevant_keywords,
            )
        )
    )

    candidates["KeywordsExclusion"] = (
        candidates["Nombre"].apply(
            lambda name: find_keyword_matches(
                name,
                excluded_keywords,
            )
        )
    )

    candidates["EsCandidatoKeyword"] = (
        candidates["KeywordsRelevantes"].apply(bool)
    )

    candidates["TieneSeñalExclusion"] = (
        candidates["KeywordsExclusion"].apply(bool)
    )

    return candidates


def save_keyword_candidates(
    candidates: pd.DataFrame,
    output_path: Path,
) -> Path:
    output = candidates.copy()

    output["KeywordsRelevantes"] = output[
        "KeywordsRelevantes"
    ].apply(
        lambda values: " | ".join(values)
    )

    output["KeywordsExclusion"] = output[
        "KeywordsExclusion"
    ].apply(
        lambda values: " | ".join(values)
    )

    output = output[
        [
            "CodigoExterno",
            "Nombre",
            "CodigoEstado",
            "FechaConsulta",
            "FechaCierre",
            "KeywordsRelevantes",
            "KeywordsExclusion",
        ]
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


if __name__ == "__main__":
    daily_tenders = load_daily_tenders(
        LISTINGS_DIR
    )

    missing_codes = daily_tenders[
        "CodigoExterno"
    ].isna().sum()

    duplicated_codes = daily_tenders.duplicated(
        subset=["CodigoExterno"],
        keep="first",
    ).sum()

    candidate_pool = build_unique_candidate_pool(
        daily_tenders
    )

    relevant_keywords, excluded_keywords = (
        load_profile_keywords(PROFILE_PATH)
    )

    candidate_pool = add_keyword_signals(
        candidate_pool,
        relevant_keywords,
        excluded_keywords,
    )

    keyword_candidates = candidate_pool[
        candidate_pool["EsCandidatoKeyword"]
    ].copy()

    exclusion_candidates = candidate_pool[
        candidate_pool["TieneSeñalExclusion"]
    ]

    both_signals = candidate_pool[
        candidate_pool["EsCandidatoKeyword"]
        & candidate_pool["TieneSeñalExclusion"]
    ]

    output_path = save_keyword_candidates(
        keyword_candidates,
        KEYWORD_CANDIDATES_PATH,
    )

    print(
        f"Filas cargadas: {len(daily_tenders)}"
    )
    print(
        f"Códigos ausentes: {missing_codes}"
    )
    print(
        f"Duplicados eliminados: "
        f"{duplicated_codes}"
    )
    print(
        f"Licitaciones únicas: "
        f"{len(candidate_pool)}"
    )
    print(
        f"Candidatos por keywords: "
        f"{len(keyword_candidates)}"
    )
    print(
        f"Con señales de exclusión: "
        f"{len(exclusion_candidates)}"
    )
    print(
        f"Con ambas señales: "
        f"{len(both_signals)}"
    )

    print(
        f"\nCandidatos guardados en: "
        f"{output_path}"
    )

    print("\nCandidatos encontrados:")

    print(
        keyword_candidates[
            [
                "CodigoExterno",
                "Nombre",
                "KeywordsRelevantes",
                "KeywordsExclusion",
            ]
        ].to_string(index=False)
    )