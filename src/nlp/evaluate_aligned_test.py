"""Evalúa una vez en septiembre el candidato congelado de 299; no reentrena."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix

from src.nlp.train_affinity import encode_tenders
from src.nlp.train_temporal_affinity_v2 import file_hash, prediction_frame, scores

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/labeled/temporal_aligned'
FREEZE = DATA / 'frozen_candidate.json'
MODEL = ROOT / 'models/affinity_aligned_299.joblib'
REPORTS = ROOT / 'data/processed'
REPORT = REPORTS / 'affinity_aligned_299_test_report.json'
PREDICTIONS = REPORTS / 'affinity_aligned_299_test_predictions.csv'


def load_model():
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    if frozen['candidate_size'] != 299 or frozen['labeling_policy_version'] != '0.2.0':
        raise ValueError('La selección congelada no es el candidato alineado de 299.')
    for path, expected in frozen['input_sha256'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f'Cambió {path} desde la selección del candidato. Revisa la trazabilidad antes de evaluar test.')
    saved = joblib.load(MODEL)
    if saved.get('target') != 'Afinidad' or saved.get('pooling') != 'mean_normalized_fragments':
        raise ValueError('El modelo no tiene la representación esperada.')
    if saved.get('labeling_policy_version') != frozen['labeling_policy_version'] or saved.get('profile_version') != frozen['profile_version'] or saved.get('embedding_model') != frozen['embedding_model']:
        raise ValueError('El criterio, perfil o encoder no coincide con la selección congelada.')
    if saved.get('dataset_version') != 'temporal_aligned_299' or saved.get('train_filename') != 'train_299.csv' or len(saved['train_codes']) != 299:
        raise ValueError('El modelo no corresponde al entrenamiento alineado de 299.')
    for name, expected in saved['dataset_hashes'].items():
        if file_hash(DATA / name) != expected:
            raise ValueError(f'Cambió {name} desde el entrenamiento.')
    if not np.array_equal(saved['classifier'].classes_, [0, 1, 2, 3]):
        raise ValueError('El clasificador no contiene las cuatro clases ordenadas.')
    return saved, frozen


def load_test(saved, frozen):
    frame = pd.read_csv(DATA / 'test.csv', dtype={'CodigoExterno': 'string'})
    if len(frame) != frozen['test_size'] or frame['CodigoExterno'].isna().any() or not frame['CodigoExterno'].is_unique:
        raise ValueError('Cantidad o códigos inválidos en test.')
    if not frame['Conjunto'].eq('test').all():
        raise ValueError('El archivo contiene casos fuera de test.')
    dates = pd.to_datetime(frame['FechaPublicacion'], errors='raise')
    if dates.isna().any() or not dates.dt.strftime('%Y-%m').eq('2026-09').all():
        raise ValueError('Test debe contener publicaciones de septiembre de 2026.')
    if frame['TextoLicitacion'].isna().any() or frame['TextoLicitacion'].str.strip().eq('').any():
        raise ValueError('Hay textos vacíos en test.')
    y = pd.to_numeric(frame['Afinidad'], errors='raise')
    if y.isna().any() or not y.isin([0, 1, 2, 3]).all():
        raise ValueError('Hay etiquetas vacías o fuera de 0–3 en test.')
    frame['Afinidad'] = y.astype(int)
    known = set(saved['train_codes']) | set(saved['validation_codes'])
    if known & set(frame['CodigoExterno']):
        raise ValueError('Test comparte códigos con train o validación.')
    return frame


def evaluate(frame, saved, frozen):
    encoder = SentenceTransformer(saved['embedding_model'], device='cpu', local_files_only=True)
    embeddings = encode_tenders(frame['TextoLicitacion'].tolist(), encoder)
    if embeddings.shape[0] != len(frame):
        raise ValueError('La cantidad de embeddings no coincide con test.')
    predictions = saved['classifier'].predict(embeddings)
    result = scores(frame, predictions)
    ranked = prediction_frame(frame, saved['classifier'], embeddings)
    report = {
        'candidate': 'affinity_aligned_299', 'candidate_size': 299,
        'labeling_policy_version': saved['labeling_policy_version'], 'profile_version': saved['profile_version'],
        'model_sha256': file_hash(MODEL), 'test_sha256': file_hash(DATA / 'test.csv'),
        'frozen_selection_sha256': file_hash(FREEZE), 'test_size': len(frame),
        'publication_min': frame['FechaPublicacion'].min(), 'publication_max': frame['FechaPublicacion'].max(),
        'metrics': result,
        'classification_report': classification_report(frame['Afinidad'], predictions, labels=[0, 1, 2, 3], output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(frame['Afinidad'], predictions, labels=[0, 1, 2, 3]).tolist(),
        'confusion_matrix_note': 'Filas: afinidad etiquetada; columnas: afinidad predicha. Orden 0,1,2,3.',
        'test_human_confirmed': int(frame['ValidadaAfinidadHumano'].eq(1).sum()),
        'training_executed': False, 'candidate_changed_after_test': False,
        'selection_reason': frozen['selection_reason'],
        'note': 'Evaluación temporal del candidato seleccionado antes de ver estos resultados. Las etiquetas de septiembre son mayoritariamente asistidas y conservan su versión original; no equivalen a evaluación experta completa ni garantizan cobertura completa del mes. No ajustar retrospectivamente el candidato para mejorar esta evaluación.',
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(PREDICTIONS, index=False, encoding='utf-8-sig')
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nEvaluación de septiembre — candidato alineado de 299:')
    print(pd.Series(result).to_string())
    print('\nResultados por clase:')
    print(classification_report(frame['Afinidad'], predictions, labels=[0, 1, 2, 3], zero_division=0))
    print(f'Reporte: {REPORT}')
    print(f'Predicciones: {PREDICTIONS}')
    print('El modelo se conservó: se evaluó sin reentrenar ni comparar otros candidatos en septiembre.')
    return report


def main():
    if REPORT.exists() or PREDICTIONS.exists():
        raise FileExistsError('Ya existen resultados de esta evaluación. Consérvalos y revísalos antes de hacer otra evaluación.')
    saved, frozen = load_model()
    test = load_test(saved, frozen)
    print(f'Candidato congelado: 299 ejemplos. Evaluando {len(test)} licitaciones de septiembre.')
    print('Las métricas comparan mayoritariamente contra etiquetas asistidas.')
    evaluate(test, saved, frozen)


if __name__ == '__main__':
    main()
