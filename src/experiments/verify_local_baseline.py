"""Comprueba la inferencia local contra las predicciones históricas guardadas."""
import hashlib
import json
import os
from pathlib import Path


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from src.ingestion.recent_batch import (
        _load_capture, _prepare, _load_candidate,
    )
    from src.nlp.rank_tenders import split_text
    from src.nlp.train_affinity import encode_tenders
    from src.nlp.predict_affinity_pilot import build_predictions

    raiz = Path(__file__).resolve().parents[2]
    lote_id = "20261016T030234991480Z_5b2b8bbd"
    carpeta_resultados = raiz / "data/processed/batches" / lote_id

    # Encontrar la ejecución histórica completada.
    ejecuciones = []
    for ruta in carpeta_resultados.glob("*/registro_ejecucion.json"):
        registro = json.loads(ruta.read_text(encoding="utf-8"))
        if registro.get("estado") == "completo":
            ejecuciones.append((ruta, registro))

    if len(ejecuciones) != 1:
        raise ValueError("Se esperaba una única ejecución histórica completa.")

    ruta_registro, registro = ejecuciones[0]
    ruta_csv = ruta_registro.parent / "predicciones.csv"
    huella = hashlib.sha256(ruta_csv.read_bytes()).hexdigest()
    if huella != registro["predicciones_sha256"]:
        raise ValueError("El CSV histórico no coincide con su huella registrada.")

    print("1. Validando capturas y cargando el modelo registrado...")
    _, capturas, _ = _load_capture(raiz / "data/raw/batches" / lote_id)
    datos = _prepare(capturas)
    clasificador, _, comprobante = _load_candidate(raiz)

    identidad = registro["identidad_modelo"]
    if comprobante["model_sha256"] != identidad["HashModeloOriginalSHA256"]:
        raise ValueError("El modelo cargado difiere del usado históricamente.")

    print("2. Cargando la revisión histórica de E5 desde la caché...")
    encoder = SentenceTransformer(
        identidad["Encoder"],
        revision=identidad["RevisionEncoderLocal"],
        device="cpu",
        local_files_only=True,
    )

    longitudes = [
        len(encoder.tokenizer.encode(
            "passage: " + fragmento,
            add_special_tokens=True,
            truncation=False,
        ))
        for texto in datos["TextoLicitacion"]
        for fragmento in split_text(texto, encoder.tokenizer)
    ]
    if not longitudes or max(longitudes) > encoder.max_seq_length:
        raise ValueError("Los fragmentos no cumplen el límite del encoder.")

    print("3. Calculando las predicciones actuales...")
    vectores = encode_tenders(datos["TextoLicitacion"].tolist(), encoder)
    actual = build_predictions(datos, clasificador, vectores)
    historico = pd.read_csv(ruta_csv, dtype={"CodigoExterno": "string"})

    actual = actual.set_index("CodigoExterno").sort_index()
    historico = historico.set_index("CodigoExterno").sort_index()

    print("4. Comparando contra los resultados históricos...")
    if not actual.index.equals(historico.index):
        raise ValueError("Los códigos actuales e históricos son diferentes.")

    np.testing.assert_array_equal(
        actual["AfinidadPredicha"].to_numpy(),
        historico["AfinidadPredicha"].to_numpy(),
    )

    columnas = [f"ProbAfinidad{k}" for k in range(4)]
    np.testing.assert_allclose(
        actual[columnas].to_numpy(dtype=float),
        historico[columnas].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-6,
    )

    print(f"\nOK: coinciden las clases de las {len(actual)} licitaciones.")
    print("OK: las probabilidades coinciden dentro de la tolerancia numérica.")
    print("OK: la inferencia local reproduce la referencia histórica.")


if __name__ == "__main__":
    main()