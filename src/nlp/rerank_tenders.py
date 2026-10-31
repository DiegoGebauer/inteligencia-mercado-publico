from pathlib import Path

import pandas as pd
import torch
import yaml
from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TOP_K = 50
MAX_LENGTH = 512


def load_profile_text() -> str:
    profile_path = PROJECT_ROOT / "config" / "company_profile.yaml"

    with profile_path.open(encoding="utf-8") as file:
        profile = yaml.safe_load(file)

    parts = [profile["company"]["description"]]

    for service in profile["relevant_services"].values():
        examples = ", ".join(service["keywords"][:3])
        parts.append(
            f"{service['description']} Ejemplos: {examples}."
        )

    return " ".join(parts)


def split_text(text: str, query: str, tokenizer) -> list[str]:
    query_tokens = tokenizer.encode(
        query,
        add_special_tokens=False,
        truncation=False,
    )

    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    # Reserva espacio para el perfil y los tokens especiales.
    chunk_size = min(
        300,
        MAX_LENGTH - len(query_tokens) - special_tokens - 16,
    )

    if chunk_size < 64:
        raise ValueError("El perfil es demasiado largo para este modelo.")

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    chunks = []
    overlap = 40

    for start in range(0, len(tokens), chunk_size - overlap):
        end = min(start + chunk_size, len(tokens))
        chunk = tokenizer.decode(tokens[start:end])

        pair_tokens = tokenizer.encode(
            query,
            text_pair=chunk,
            truncation=False,
        )

        if len(pair_tokens) > MAX_LENGTH:
            raise ValueError("Un fragmento supera el límite del modelo.")

        chunks.append(chunk)

        if end == len(tokens):
            break

    return chunks


if __name__ == "__main__":
    ranking = pd.read_csv(
        PROCESSED_DIR / "tender_ranking_v3.csv",
        dtype={"CodigoExterno": "string"},
    )

    texts = pd.read_csv(
        PROCESSED_DIR / "tender_texts.csv",
        dtype={"CodigoExterno": "string"},
    )

    candidates = (
        ranking.sort_values("Posicion")
        .head(TOP_K)
        .copy()
        .rename(columns={"Posicion": "PosicionV3"})
    )

    candidates = candidates.merge(
        texts,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    if candidates.empty or candidates["TextoLicitacion"].isna().any():
        raise ValueError("Faltan textos para reordenar los candidatos.")

    print(f"Candidatos seleccionados: {len(candidates)}")
    print(f"Cargando modelo: {MODEL_NAME}")

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
        max_length=MAX_LENGTH,
    )

    query = load_profile_text()
    pairs = []
    owners = []

    for row in candidates.itertuples(index=False):
        chunks = split_text(
            row.TextoLicitacion,
            query,
            model.tokenizer,
        )

        for chunk in chunks:
            pairs.append((query, chunk))
            owners.append(row.CodigoExterno)

    print(f"Pares a evaluar: {len(pairs)}")

    scores = model.predict(
        pairs,
        batch_size=8,
        show_progress_bar=True,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
    ).reshape(-1)

    fragment_scores = pd.DataFrame({
        "CodigoExterno": owners,
        "ScoreReordenado": scores,
        "TextoFragmento": [pair[1] for pair in pairs],
    })

    # Mantiene el promedio de fragmentos como criterio.
    tender_scores = (
        fragment_scores.groupby("CodigoExterno", as_index=False)
        .agg(
            ScoreReordenado=("ScoreReordenado", "mean"),
            FragmentosReordenados=("ScoreReordenado", "size"),
        )
    )

    best_indices = (
        fragment_scores.groupby("CodigoExterno")["ScoreReordenado"]
        .idxmax()
    )

    best_fragments = fragment_scores.loc[
        best_indices,
        ["CodigoExterno", "TextoFragmento"],
    ].rename(columns={
        "TextoFragmento": "FragmentoMasAfinReordenado",
    })

    result = candidates.merge(
        tender_scores,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        best_fragments,
        on="CodigoExterno",
        how="left",
        validate="one_to_one",
    )

    result = result.sort_values(
        ["ScoreReordenado", "CodigoExterno"],
        ascending=[False, True],
    ).reset_index(drop=True)

    result.insert(0, "Posicion", range(1, len(result) + 1))
    result = result.drop(columns=["TextoLicitacion"])

    output_path = PROCESSED_DIR / "tender_reranked_top50.csv"
    result.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nCandidatos reordenados: {len(result)}")
    print(f"Archivo generado: {output_path}")
    print("\nPrimeras 10:")
    print(
        result[
            ["Posicion", "PosicionV3", "ScoreReordenado", "Nombre"]
        ]
        .head(10)
        .to_string(index=False)
    )