"""Compara 199 y 299 ejemplos con etiquetas alineadas; reserva septiembre."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.nlp.train_affinity import encode_tenders
from src.nlp.train_temporal_affinity_v2 import file_hash, prediction_frame, scores

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/labeled/temporal_aligned'
REPORTS = ROOT / 'data/processed'
MODELS = ROOT / 'models'
COMPARISON = REPORTS / 'affinity_aligned_comparison.json'
CLASSES = [0, 1, 2, 3]


def validate_frame(frame, split):
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
        raise ValueError(f'Fechas incorrectas en {split}.')
    if split == 'train' and set(frame['Afinidad']) != set(CLASSES):
        raise ValueError('Train debe contener las cuatro clases.')
    return frame


def load_inputs():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['dataset_version'] != 'temporal_aligned' or manifest['labeling_policy_version'] != '0.2.0':
        raise ValueError('El paquete no tiene el criterio alineado esperado.')
    for name, expected in manifest['files_sha256'].items():
        if file_hash(DATA / name) != expected:
            raise ValueError(f'Cambió {name} desde la revisión. Versiona la corrección antes de entrenar.')
    for path, expected in manifest['required_code_sha256'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f'{path} no coincide con el código verificado al preparar el paquete.')

    small = validate_frame(pd.read_csv(DATA / 'train_199.csv', dtype={'CodigoExterno': 'string'}), 'train')
    large = validate_frame(pd.read_csv(DATA / 'train_299.csv', dtype={'CodigoExterno': 'string'}), 'train')
    validation = validate_frame(pd.read_csv(DATA / 'validation.csv', dtype={'CodigoExterno': 'string'}), 'val')
    if (len(small), len(large), len(validation)) != (199, 299, 60):
        raise ValueError('Se esperan 199, 299 y 60 casos.')
    small_codes = small['CodigoExterno'].tolist()
    large_by_code = large.set_index('CodigoExterno')
    if not set(small_codes).issubset(large_by_code.index):
        raise ValueError('Los 199 casos deben ser un subconjunto del train de 299.')
    for field in ['TextoLicitacion', 'Afinidad', 'FechaPublicacion']:
        if small[field].tolist() != large_by_code.loc[small_codes, field].tolist():
            raise ValueError(f'El subconjunto de 199 tiene diferencias en {field}.')
    together = pd.concat([large, validation], ignore_index=True)
    normalized = together['TextoLicitacion'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    if not together['CodigoExterno'].is_unique or normalized.duplicated().any():
        raise ValueError('Hay códigos o textos compartidos entre train y validación.')

    # Septiembre se verifica únicamente como archivo: no se cargan sus etiquetas.
    old_models = {}
    for name, expected in manifest['original_model_sha256'].items():
        path = MODELS / f'affinity_temporal_{name}.joblib'
        if file_hash(path) != expected:
            raise ValueError(f'El modelo original {name} cambió desde la preparación del paquete.')
        saved = joblib.load(path)
        if saved.get('embedding_model') != manifest['embedding_model'] or saved.get('pooling') != 'mean_normalized_fragments' or saved.get('target') != 'Afinidad':
            raise ValueError(f'El modelo original {name} tiene una representación distinta.')
        old_models[name] = saved
    return {'199': small, '299': large, 'validation': validation}, manifest, old_models


def train_candidates(frame, embeddings, validation, X_val):
    candidates = []
    y = frame['Afinidad'].to_numpy()
    for c in [0.1, 1.0, 10.0]:
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42))
        classifier.fit(embeddings, y)
        metrics = scores(validation, classifier.predict(X_val))
        candidates.append((metrics['F1MacroPonderado'], -metrics['MAEPonderado'], -c, classifier, metrics))
        print(f'Train={len(frame)}, C={c}: {metrics}')
    best = max(candidates, key=lambda item: item[:3])
    return best, [{'C': -item[2], 'metrics': item[4]} for item in candidates]


def run_experiment(frames, manifest, old_models):
    small, large, validation = frames['199'], frames['299'], frames['validation']
    encoder = SentenceTransformer(manifest['embedding_model'], device='cpu', local_files_only=True)
    # Los 199 casos originales reutilizan los mismos vectores, sin representar dos veces el texto.
    X_large = encode_tenders(large['TextoLicitacion'].tolist(), encoder)
    X_val = encode_tenders(validation['TextoLicitacion'].tolist(), encoder)
    if X_large.shape[0] != len(large) or X_val.shape[0] != len(validation):
        raise ValueError('La cantidad de embeddings no coincide con los datos.')
    index_by_code = {code: i for i, code in enumerate(large['CodigoExterno'])}
    small_indices = np.array([index_by_code[code] for code in small['CodigoExterno']], dtype=int)
    report = {
        'labeling_policy_version': manifest['labeling_policy_version'],
        'profile_version': manifest['profile_version'], 'validation_sha256': file_hash(DATA / 'validation.csv'),
        'evaluation_labels_changed': ['VAL170', 'VAL182'],
        'original_saved_models_on_revised_validation': {
            name: scores(validation, saved['classifier'].predict(X_val)) for name, saved in old_models.items()
        },
        'aligned_models': {}, 'test_evaluated': False,
        'note': 'Las versiones alineadas se entrenan de nuevo y se evalúan sobre la misma validación revisada. La auditoría fue motivada por resultados anteriores; agosto es validación de desarrollo, no prueba independiente. Las etiquetas siguen mayoritariamente asistidas.',
    }
    artifacts = []
    for size, frame, X_train in [('199', small, X_large[small_indices]), ('299', large, X_large)]:
        print(f'\nEntrenando versión alineada con {size} ejemplos...')
        best, candidates = train_candidates(frame, X_train, validation, X_val)
        classifier = best[3]
        baseline = DummyClassifier(strategy='most_frequent').fit(X_train, frame['Afinidad'].to_numpy())
        report['aligned_models'][size] = {
            'train_size': len(frame), 'chosen_C': -best[2], 'metrics': best[4], 'candidates': candidates,
            'reference_majority': scores(validation, baseline.predict(X_val)),
        }
        predictions = prediction_frame(validation, classifier, X_val)
        saved = {
            'classifier': classifier, 'embedding_model': manifest['embedding_model'],
            'pooling': 'mean_normalized_fragments', 'target': 'Afinidad',
            'profile_version': manifest['profile_version'], 'labeling_policy_version': manifest['labeling_policy_version'],
            'dataset_version': f'temporal_aligned_{size}', 'dataset_directory': 'data/labeled/temporal_aligned',
            'train_filename': f'train_{size}.csv', 'chosen_C': -best[2], 'validation_scores': best[4],
            'train_codes': frame['CodigoExterno'].tolist(), 'validation_codes': validation['CodigoExterno'].tolist(),
            'dataset_hashes': {name: file_hash(DATA / name) for name in [f'train_{size}.csv', 'validation.csv', 'test.csv']},
        }
        artifacts.append((size, saved, predictions))
    # Se escriben resultados después de terminar ambos entrenamientos.
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for size, saved, predictions in artifacts:
        joblib.dump(saved, MODELS / f'affinity_aligned_{size}.joblib')
        predictions.to_csv(REPORTS / f'affinity_aligned_{size}_validation_predictions.csv', index=False, encoding='utf-8-sig')
    COMPARISON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nComparación alineada sobre los mismos 60 casos de agosto:')
    print(pd.DataFrame({f'train_{size}': report['aligned_models'][size]['metrics'] for size in ['199', '299']}).to_string())
    print(f'\nComparación guardada: {COMPARISON}')
    print('Los modelos originales se conservan. Septiembre sigue sin evaluarse.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Reemplaza solamente los resultados alineados de esta comparación.')
    args = parser.parse_args()
    paths = [COMPARISON] + [MODELS / f'affinity_aligned_{size}.joblib' for size in ['199', '299']] + [REPORTS / f'affinity_aligned_{size}_validation_predictions.csv' for size in ['199', '299']]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Ya hay resultados alineados. Revísalos antes de repetir con --overwrite.')
    frames, manifest, old_models = load_inputs()
    print('Train: 199 frente a 299. Validación revisada: 60. Criterio de etiquetado: 0.2.0.')
    print('Las métricas comparan mayoritariamente contra etiquetas asistidas; septiembre queda reservado.')
    run_experiment(frames, manifest, old_models)


if __name__ == '__main__':
    main()
