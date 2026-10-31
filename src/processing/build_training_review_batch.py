import argparse
import re
from pathlib import Path

import pandas as pd

from src.processing.build_candidate_pool import (
    find_keyword_matches,
    load_profile_keywords,
    normalize_search_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
PROFILE_PATH = PROJECT_ROOT / "config" / "company_profile.yaml"

CONFUSION_KEYWORDS = [
    "software", "licencia", "licencias", "soporte",
    "capacitación", "capacitaciones", "consultoría",
    "estudio", "estudios", "evaluación", "análisis",
    "inteligencia artificial",
]


def keyword_pattern(keywords: list[str]) -> str:
    terms = [
        re.escape(normalize_search_text(keyword))
        for keyword in keywords
        if normalize_search_text(keyword)
    ]

    if not terms:
        return r"(?!)"

    return r"\b(?:" + "|".join(terms) + r")\b"


def take_balanced_sample(
    frame: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    if len(frame) < size:
        raise ValueError(
            f"Se necesitan {size} casos, pero hay {len(frame)}."
        )

    shuffled = frame.sample(frac=1, random_state=seed).copy()

    # Tomamos casos de distintos meses por turnos.
    shuffled["_TurnoMes"] = (
        shuffled.groupby("PeriodoPublicacion").cumcount()
    )

    return (
        shuffled.sort_values(["_TurnoMes", "PeriodoPublicacion"])
        .head(size)
        .drop(columns="_TurnoMes")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if args.batch < 1:
        parser.error("--batch debe ser un número positivo.")

    output_path = LABELED_DIR / (
        f"training_review_batch_{args.batch:03d}.csv"
    )

    if output_path.exists():
        raise FileExistsError(f"El lote ya existe: {output_path}")

    base = pd.read_csv(
        PROCESSED_DIR / "monthly_texts_2026_combined.csv",
        dtype="string",
    )

    if base["CodigoExterno"].duplicated().any():
        raise ValueError("La base contiene códigos repetidos.")

    # Excluimos casos ya etiquetados o seleccionados en otros lotes.
    known_codes = set()

    for path in sorted(LABELED_DIR.glob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns

        if "CodigoExterno" in columns:
            previous = pd.read_csv(
                path,
                usecols=["CodigoExterno"],
                dtype="string",
            )
            known_codes.update(
                previous["CodigoExterno"].dropna()
            )

    dates = pd.to_datetime(
        base["FechaPublicacion"],
        format="%Y-%m-%d",
        errors="raise",
    )

    pool = base.loc[
        dates.ge("2026-01-01")
        & dates.lt("2026-08-01")
        & ~base["CodigoExterno"].isin(known_codes)
    ].copy()

    pool["PeriodoPublicacion"] = (
        dates.loc[pool.index].dt.strftime("%Y-%m")
    )

    print(f"Candidatos nuevos: {len(pool):,}", flush=True)
    print("Buscando términos del perfil...", flush=True)

    relevant_keywords, excluded_keywords = load_profile_keywords(
        PROFILE_PATH
    )

    search_text = pool["TextoLicitacion"].map(
        normalize_search_text
    )

    relevant = search_text.str.contains(
        keyword_pattern(relevant_keywords),
        regex=True,
    )
    excluded = search_text.str.contains(
        keyword_pattern(excluded_keywords),
        regex=True,
    )
    generic = search_text.str.contains(
        keyword_pattern(CONFUSION_KEYWORDS),
        regex=True,
    )

    groups = [
        ("senales_datos", relevant & ~excluded, 25),
        ("posibles_confusiones", excluded | (generic & ~relevant), 15),
        ("aleatoria", pd.Series(True, index=pool.index), 10),
    ]

    selected_codes = set()
    parts = []
    seed = 42 + args.batch - 1

    for offset, (group_name, mask, size) in enumerate(groups):
        available = pool.loc[
            mask & ~pool["CodigoExterno"].isin(selected_codes)
        ]

        part = take_balanced_sample(
            available,
            size,
            seed + offset,
        ).copy()

        part["GrupoMuestra"] = group_name
        selected_codes.update(part["CodigoExterno"])
        parts.append(part)

    sample = pd.concat(parts, ignore_index=True)

    sample = sample.sort_values(
        ["FechaPublicacion", "CodigoExterno"]
    ).reset_index(drop=True)

    sample.insert(
        0,
        "OrdenRevision",
        range(1, len(sample) + 1),
    )

    sample["KeywordsRelevantes"] = sample["TextoLicitacion"].map(
        lambda text: " | ".join(
            find_keyword_matches(text, relevant_keywords)
        )
    )
    sample["KeywordsExclusion"] = sample["TextoLicitacion"].map(
        lambda text: " | ".join(
            find_keyword_matches(text, excluded_keywords)
        )
    )

    for column in [
        "Afinidad",
        "EncajeServicio",
        "Especializacion",
        "Motivo",
        "OrigenEtiqueta",
    ]:
        sample[column] = ""

    sample["EstadoRevision"] = "pendiente"

    if len(sample) != 50 or sample["CodigoExterno"].duplicated().any():
        raise ValueError("El lote debe contener 50 códigos distintos.")

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nCasos preparados para revisión: {len(sample)}")
    print("\nDistribución por grupo:")
    print(sample["GrupoMuestra"].value_counts().to_string())
    print("\nDistribución por mes:")
    print(
        sample["PeriodoPublicacion"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print(f"\nArchivo generado: {output_path}")


if __name__ == "__main__":
    main()
