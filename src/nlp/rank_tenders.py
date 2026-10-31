from pathlib import Path

import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_NAME = "intfloat/multilingual-e5-small"


def load_profile_queries() -> dict[str, str]:
    profile_path = PROJECT_ROOT / "config" / "company_profile.yaml"

    with profile_path.open(encoding="utf-8") as file:
        profile = yaml.safe_load(file)

    queries = {}

    for name, service in profile["relevant_services"].items():
        topics = ", ".join(service["keywords"])

        queries[name] = (
            f"query: {service['description']} "
            f"Tecnologías y temas específicos: {topics}."
        )

    if not queries:
        raise ValueError("El perfil no contiene servicios relevantes.")

    return queries


def split_text(text: str, tokenizer) -> list[str]:
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    chunks = []
    chunk_size = 400
    overlap = 40

    for start in range(0, len(tokens), chunk_size - overlap):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))

        if end == len(tokens):
            break

    return chunks


if __name__ == "__main__":
    texts = pd.read_csv(
        PROCESSED_DIR / "tender_texts.csv",
        dtype={"CodigoExterno": "string"},
    )

    if texts.empty or texts["TextoLicitacion"].isna().any():
        raise ValueError("Faltan textos para generar el ranking.")

    print(f"Cargando modelo: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    queries = load_profile_queries()
    service_names = list(queries)

    print(f"Especialidades a comparar: {len(service_names)}")

    profile_embeddings = model.encode(
        list(queries.values()),
        normalize_embeddings=True,
    )

    passages = []
    owners = []

    for row in texts.itertuples(index=False):
        chunks = split_text(row.TextoLicitacion, model.tokenizer)

        for chunk in chunks:
            passages.append("passage: " + chunk)
            owners.append(row.CodigoExterno)

    print(f"Fragmentos a comparar: {len(passages)}")

    passage_embeddings = model.encode(
        passages,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Cada fragmento obtiene un puntaje por especialidad.
    similarities = passage_embeddings @ profile_embeddings.T
    score_columns = [f"Score_{name}" for name in service_names]

    fragment_scores = pd.DataFrame(
        similarities,
        columns=score_columns,
    )

    fragment_scores["CodigoExterno"] = owners
    fragment_scores["TextoFragmento"] = passages

    grouped = fragment_scores.groupby("CodigoExterno", sort=False)

    # Promedia los fragmentos para cada especialidad.
    ranking = grouped[score_columns].mean().reset_index()

    ranking["NumeroFragmentos"] = (
        ranking["CodigoExterno"].map(grouped.size())
    )

    # Selecciona la especialidad con mayor similitud media.
    best_columns = ranking[score_columns].idxmax(axis=1)

    ranking["ScoreMedio"] = ranking[score_columns].max(axis=1)

    ranking["EspecialidadMasAfin"] = best_columns.map(
        lambda column: column.removeprefix("Score_")
    )

    preferred_columns = dict(zip(
        ranking["CodigoExterno"],
        best_columns,
    ))

    # Busca el fragmento más afín dentro de la especialidad elegida.
    fragment_scores["ScoreSeleccionado"] = [
        row[preferred_columns[row["CodigoExterno"]]]
        for _, row in fragment_scores.iterrows()
    ]

    best_indices = (
        fragment_scores.groupby("CodigoExterno")["ScoreSeleccionado"]
        .idxmax()
    )

    best_fragments = fragment_scores.loc[
        best_indices,
        ["CodigoExterno", "ScoreSeleccionado", "TextoFragmento"],
    ].rename(columns={
        "ScoreSeleccionado": "ScoreSemantico",
        "TextoFragmento": "FragmentoMasAfin",
    })

    ranking = ranking.merge(
        best_fragments,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    names = pd.read_csv(
        PROCESSED_DIR / "tenders.csv",
        dtype={"CodigoExterno": "string"},
        usecols=["CodigoExterno", "Nombre"],
    )

    ranking = ranking.merge(
        names,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    ranking = ranking.sort_values(
        ["ScoreMedio", "CodigoExterno"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking.insert(0, "Posicion", range(1, len(ranking) + 1))

    output_path = PROCESSED_DIR / "tender_ranking_v3.csv"
    ranking.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nLicitaciones ordenadas: {len(ranking)}")
    print(f"Archivo generado: {output_path}")
    print("\nPrimeras 10:")
    print(
        ranking[
            ["Posicion", "ScoreMedio", "EspecialidadMasAfin", "Nombre"]
        ]
        .head(10)
        .to_string(index=False)
    )