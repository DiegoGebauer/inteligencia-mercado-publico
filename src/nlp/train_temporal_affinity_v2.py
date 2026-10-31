"""Amplía train y compara v1/v2 en el mismo agosto; septiembre queda reservado."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.nlp.train_affinity import encode_tenders

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/labeled/temporal_v2'
MODEL_V1 = ROOT / 'models/affinity_temporal_v1.joblib'
MODEL_V2 = ROOT / 'models/affinity_temporal_v2.joblib'
REPORTS = ROOT / 'data/processed'
PREDICTIONS = REPORTS / 'affinity_temporal_v2_validation_predictions.csv'
COMPARISON = REPORTS / 'affinity_temporal_v1_v2_comparison.json'
CLASSES = [0, 1, 2, 3]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['dataset_version'] != 'temporal_v2':
        raise ValueError('El manifest no corresponde a temporal_v2.')
    for name, expected in manifest['files_sha256'].items():
        if file_hash(DATA / name) != expected:
            raise ValueError(f'Cambió {name} respecto del paquete revisado.')
    if file_hash(MODEL_V1) != manifest['baseline_model_sha256']:
        raise ValueError('El modelo v1 cambió desde la preparación del paquete.')
    saved = joblib.load(MODEL_V1)
    if saved.get('target') != 'Afinidad' or saved.get('pooling') != 'mean_normalized_fragments':
        raise ValueError('El modelo v1 no tiene la representación esperada.')
    if saved.get('profile_version') != manifest['profile_version']:
        raise ValueError('La versión del perfil no coincide.')
    if saved.get('embedding_model') != manifest['embedding_model']:
        raise ValueError('El modelo de embeddings no coincide.')
    if saved.get('dataset_hashes') != manifest['base_dataset_sha256']:
        raise ValueError('El modelo v1 no corresponde al dataset base del paquete.')
    for name in ['validation.csv', 'test.csv']:
        if file_hash(DATA / name) != saved['dataset_hashes'][name]:
            raise ValueError(f'{name} cambió: la comparación exige la misma evaluación.')

    frames = {}
    # Se verifica el hash de test, pero no se cargan sus etiquetas ni se evalúa.
    for split, filename in [('train', 'train.csv'), ('val', 'validation.csv')]:
        frame = pd.read_csv(DATA / filename, dtype={'CodigoExterno': 'string'})
        if frame.empty or frame['CodigoExterno'].isna().any() or not frame['CodigoExterno'].is_unique:
            raise ValueError(f'Códigos inválidos en {split}.')
        if frame['TextoLicitacion'].isna().any() or frame['TextoLicitacion'].str.strip().eq('').any():
            raise ValueError(f'Textos vacíos en {split}.')
        if not frame['Conjunto'].eq(split).all():
            raise ValueError(f'Conjunto incorrecto en {split}.')
        affinity = pd.to_numeric(frame['Afinidad'], errors='raise')
        if affinity.isna().any() or not affinity.isin(CLASSES).all():
            raise ValueError(f'Afinidades inválidas en {split}.')
        frame['Afinidad'] = affinity.astype(int)
        dates = pd.to_datetime(frame['FechaPublicacion'], errors='raise')
        correct = dates.ge('2026-01-01') & dates.lt('2026-08-01') if split == 'train' else dates.dt.strftime('%Y-%m').eq('2026-08')
        if dates.isna().any() or not correct.all():
            raise ValueError(f'Fechas inválidas en {split}.')
        if len(frame) != manifest['splits'][split]['rows']:
            raise ValueError(f'Cantidad inesperada de casos en {split}.')
        frames[split] = frame
    together = pd.concat(frames.values(), ignore_index=True)
    normalized = together['TextoLicitacion'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    if not together['CodigoExterno'].is_unique or normalized.duplicated().any():
        raise ValueError('Hay códigos o textos repetidos entre train y validación.')
    if set(frames['train']['Afinidad']) != set(CLASSES):
        raise ValueError('Train debe contener las cuatro clases.')
    return frames, manifest, saved


def scores(frame, predictions):
    y = frame['Afinidad'].to_numpy(dtype=int)
    weights = pd.to_numeric(frame['PesoMuestra'], errors='raise').to_numpy()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Pesos de validación inválidos.')
    predictions = np.asarray(predictions)
    if predictions.shape != y.shape or not np.isin(predictions, CLASSES).all():
        raise ValueError('Predicciones inválidas.')
    return {
        'Exactitud': float(accuracy_score(y, predictions)),
        'F1Macro': float(f1_score(y, predictions, labels=CLASSES, average='macro', zero_division=0)),
        'MAE': float(mean_absolute_error(y, predictions)),
        'RecallRelevantes': float(recall_score(y >= 2, predictions >= 2, zero_division=0)),
        'ExactitudPonderada': float(accuracy_score(y, predictions, sample_weight=weights)),
        'F1MacroPonderado': float(f1_score(y, predictions, labels=CLASSES, average='macro', sample_weight=weights, zero_division=0)),
        'MAEPonderado': float(mean_absolute_error(y, predictions, sample_weight=weights)),
        'RecallRelevantesPonderado': float(recall_score(y >= 2, predictions >= 2, sample_weight=weights, zero_division=0)),
    }


def prediction_frame(frame, classifier, embeddings):
    if not np.array_equal(classifier.classes_, CLASSES):
        raise ValueError('El clasificador debe contener las cuatro clases ordenadas.')
    probabilities = classifier.predict_proba(embeddings)
    if probabilities.shape != (len(frame), 4) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Probabilidades inválidas.')
    result = frame.copy()
    result['AfinidadPredicha'] = classifier.predict(embeddings)
    if not result['AfinidadPredicha'].isin(CLASSES).all():
        raise ValueError('Afinidades predichas inválidas.')
    result['ErrorAbsoluto'] = abs(result['Afinidad'] - result['AfinidadPredicha'])
    for k in CLASSES:
        result[f'ProbAfinidad{k}'] = probabilities[:, k]
    result['AfinidadEsperada'] = probabilities @ np.asarray(CLASSES)
    result = result.sort_values(['AfinidadEsperada', 'ProbAfinidad3', 'CodigoExterno'], ascending=[False, False, True]).reset_index(drop=True)
    result.insert(0, 'Posicion', np.arange(1, len(result) + 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Reemplaza únicamente los resultados existentes de v2.')
    args = parser.parse_args()
    if not args.overwrite and any(p.exists() for p in [MODEL_V2, PREDICTIONS, COMPARISON]):
        raise FileExistsError('Ya hay resultados de v2. Revísalos antes de repetir con --overwrite.')
    frames, manifest, previous = load_inputs()
    train, validation = frames['train'], frames['val']
    print(f'Train: {len(train)}; validación de agosto: {len(validation)}. Septiembre reservado.')
    print('Las métricas comparan principalmente contra etiquetas asistidas.')
    encoder = SentenceTransformer(previous['embedding_model'], device='cpu', local_files_only=True)
    X_train = encode_tenders(train['TextoLicitacion'].tolist(), encoder)
    X_val = encode_tenders(validation['TextoLicitacion'].tolist(), encoder)
    candidates = []
    for c in [0.1, 1.0, 10.0]:
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42))
        classifier.fit(X_train, train['Afinidad'].to_numpy())
        predictions = classifier.predict(X_val)
        metrics = scores(validation, predictions)
        candidates.append((metrics['F1MacroPonderado'], -metrics['MAEPonderado'], -c, classifier, predictions, metrics))
        print(f'C={c}: {metrics}')
    best = max(candidates, key=lambda item: item[:3])
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, train['Afinidad'].to_numpy())
    metrics_v1 = scores(validation, previous['classifier'].predict(X_val))
    comparison = {
        'dataset_version': 'temporal_v2', 'profile_version': manifest['profile_version'],
        'validation_sha256': file_hash(DATA / 'validation.csv'),
        'baseline_model_sha256': file_hash(MODEL_V1),
        'train_sizes': {'v1': len(previous['train_codes']), 'v2': len(train)},
        'chosen_C_v2': -best[2],
        'candidates': [{'C': -item[2], 'metrics': item[5]} for item in candidates],
        'v1': metrics_v1, 'v2': best[5],
        'reference_majority': scores(validation, baseline.predict(X_val)),
        'validation_human_confirmed': int(validation['ValidadaAfinidadHumano'].eq(1).sum()),
        'test_evaluated': False,
        'note': 'Misma validación enriquecida y mayoritariamente asistida. Los recalls ponderados pueden depender de pocos casos con pesos altos.',
    }
    ranked_predictions = prediction_frame(validation, best[3], X_val)
    saved = {
        'classifier': best[3], 'embedding_model': previous['embedding_model'],
        'pooling': previous['pooling'], 'target': 'Afinidad',
        'dataset_version': 'temporal_v2', 'profile_version': manifest['profile_version'],
        'chosen_C': -best[2], 'validation_scores': best[5],
        'train_codes': train['CodigoExterno'].tolist(), 'validation_codes': validation['CodigoExterno'].tolist(),
        'dataset_hashes': {name: file_hash(DATA / name) for name in ['train.csv', 'validation.csv', 'test.csv']},
        'baseline_model_sha256': comparison['baseline_model_sha256'],
    }
    MODEL_V2.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved, MODEL_V2)
    ranked_predictions.to_csv(PREDICTIONS, index=False, encoding='utf-8-sig')
    COMPARISON.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nComparación sobre los mismos casos de agosto:')
    print(pd.DataFrame({'v1': comparison['v1'], 'v2': comparison['v2']}).to_string())
    print('\nResultados por clase de v2:')
    print(classification_report(validation['Afinidad'], best[4], labels=CLASSES, zero_division=0))
    print(f'Modelo nuevo: {MODEL_V2}')
    print(f'Comparación: {COMPARISON}')
    print('El modelo v1 se conserva. Septiembre sigue sin evaluarse.')


if __name__ == '__main__':
    main()
