"""Compara los modelos alineados en casos del piloto que ninguno vio al entrenar."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.nlp.predict_affinity_pilot import build_predictions, file_hash, tender_family
from src.nlp.train_affinity import encode_tenders

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'data/processed/affinity_aligned_pilot_contract.json'
REPORTS = ROOT / 'data/processed'
OUTPUT_NAMES = [
    'affinity_aligned_199_pilot_predictions.csv',
    'affinity_aligned_299_pilot_predictions.csv',
    'affinity_aligned_pilot_comparison.csv',
    'affinity_aligned_pilot_review.csv',
    'affinity_aligned_pilot_manifest.json',
]


def word_set(text):
    text = unicodedata.normalize('NFKD', str(text).lower())
    return set(re.findall(r'\w+', ''.join(c for c in text if not unicodedata.combining(c))))


def normalized_text(text):
    return ' '.join(str(text).lower().split())


def prepare_sample(pilot, known, threshold=0.85):
    required = {'CodigoExterno', 'Nombre', 'FechaPublicacion', 'TextoLicitacion'}
    if not required.issubset(pilot.columns):
        raise ValueError('Faltan datos originales del piloto.')
    if pilot.empty or pilot['CodigoExterno'].isna().any() or not pilot['CodigoExterno'].is_unique:
        raise ValueError('Códigos del piloto inválidos.')
    if pilot['TextoLicitacion'].isna().any() or pilot['TextoLicitacion'].str.strip().eq('').any():
        raise ValueError('Textos del piloto vacíos.')
    dates = pd.to_datetime(pilot['FechaPublicacion'], errors='raise')
    if dates.isna().any() or not (dates.ge('2026-01-01') & dates.lt('2026-08-01')).all():
        raise ValueError('El piloto debe ser de enero-julio; agosto y septiembre se reservan.')
    known = known.drop_duplicates('CodigoExterno')
    codes = set(known['CodigoExterno'])
    families = {tender_family(c) for c in codes}
    texts = set(known['TextoLicitacion'].map(normalized_text))
    token_sets = [(r.CodigoExterno, word_set(r.TextoLicitacion)) for r in known.itertuples(index=False)]
    keep = []
    exclusions = []
    for row in pilot.itertuples(index=False):
        code = row.CodigoExterno
        reason, match = None, None
        if code in codes:
            reason, match = 'codigo_conocido', code
        elif tender_family(code) in families:
            reason = 'familia_conocida'
        elif normalized_text(row.TextoLicitacion) in texts:
            reason = 'texto_conocido'
        else:
            words = word_set(row.TextoLicitacion)
            for other, tokens in token_sets:
                if len(words & tokens) / max(len(words | tokens), 1) >= threshold:
                    reason, match = 'texto_muy_parecido', other
                    break
        if reason:
            exclusions.append({'CodigoExterno': code, 'Motivo': reason, 'Coincidencia': match})
        else:
            keep.append(code)
    # Retener datos originales; descartar todas las predicciones y puntajes de v1.
    prediction_columns = {'Posicion', 'AfinidadPredicha', 'AfinidadEsperada', 'ScoreRelevancia', 'MargenPrediccion', 'Afinidad', 'EncajeServicio'}
    columns = [c for c in pilot.columns if c not in prediction_columns and not c.startswith('ProbAfinidad')]
    sample = pilot.loc[pilot['CodigoExterno'].isin(keep), columns].sort_values('CodigoExterno').reset_index(drop=True)
    if sample.empty:
        raise ValueError('No quedan casos nuevos para comparar.')
    return sample, exclusions


def load_inputs():
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    for path, expected in contract['input_sha256'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f'Cambió {path} desde la preparación del piloto comparativo.')
    models = {}
    for size, path in contract['model_paths'].items():
        saved = joblib.load(ROOT / path)
        if saved.get('target') != 'Afinidad' or saved.get('pooling') != 'mean_normalized_fragments':
            raise ValueError(f'Representación incorrecta en el modelo {size}.')
        if saved.get('labeling_policy_version') != '0.2.0' or saved.get('profile_version') != contract['profile_version'] or saved.get('embedding_model') != contract['embedding_model']:
            raise ValueError(f'Criterio, perfil o encoder distinto en el modelo {size}.')
        if len(saved['train_codes']) != int(size):
            raise ValueError(f'Tamaño inesperado de train en el modelo {size}.')
        if saved.get('train_filename') != f'train_{size}.csv' or saved.get('dataset_directory') != 'data/labeled/temporal_aligned':
            raise ValueError(f'Dataset inesperado en el modelo {size}.')
        for name, expected in saved['dataset_hashes'].items():
            if file_hash(ROOT / saved['dataset_directory'] / name) != expected:
                raise ValueError(f'Cambió {name} desde el entrenamiento del modelo {size}.')
        models[size] = saved
    pilot = pd.read_csv(ROOT / contract['pilot_source'], dtype={'CodigoExterno': 'string'})
    if len(pilot) != contract['source_size']:
        raise ValueError('Cambió la cantidad de casos del piloto original.')
    # De los archivos de evaluación se leen solo códigos y textos, nunca las etiquetas.
    known = pd.concat([
        pd.read_csv(ROOT / path, usecols=['CodigoExterno', 'TextoLicitacion'], dtype=str)
        for path in contract['exclusion_sources']
    ], ignore_index=True)
    sample, exclusions = prepare_sample(pilot, known, contract['similarity_threshold'])
    if sample['CodigoExterno'].tolist() != contract['expected_codes']:
        raise ValueError('La selección no coincide con la selección verificada.')
    return sample, exclusions, models, contract


def combine_rankings(sample, rankings):
    combined = sample.copy()
    fields = ['Posicion', 'AfinidadPredicha', 'AfinidadEsperada', 'MargenPrediccion']
    for size in ['199', '299']:
        predictions = rankings[size][['CodigoExterno'] + fields].rename(columns={c: f'{c}_{size}' for c in fields})
        if set(predictions['CodigoExterno']) != set(sample['CodigoExterno']):
            raise ValueError('Los modelos no predijeron los mismos códigos.')
        combined = combined.merge(predictions, on='CodigoExterno', validate='one_to_one')
    combined['DesacuerdoAfinidad'] = combined['AfinidadPredicha_199'] != combined['AfinidadPredicha_299']
    combined['DiferenciaPosicion'] = abs(combined['Posicion_199'] - combined['Posicion_299'])
    return combined.sort_values('Posicion_199').reset_index(drop=True)


def select_review(combined, seed=20261008):
    groups = {}
    for size in ['199', '299']:
        for code in combined.nsmallest(10, f'Posicion_{size}')['CodigoExterno']:
            groups.setdefault(code, []).append(f'top10_{size}')
    disagreements = combined.loc[combined['DesacuerdoAfinidad'] & ~combined['CodigoExterno'].isin(groups)]
    disagreements = disagreements.sort_values(['DiferenciaPosicion', 'CodigoExterno'], ascending=[False, True])
    for code in disagreements.head(10)['CodigoExterno']:
        groups[code] = ['desacuerdo']
    rest = combined.loc[~combined['CodigoExterno'].isin(groups)]
    for code in rest.sample(n=min(10, len(rest)), random_state=seed)['CodigoExterno']:
        groups[code] = ['contraste_aleatorio']
    review = combined.set_index('CodigoExterno').loc[list(groups)].reset_index()
    review.insert(0, 'IdRevision', [f'AUD{i:03d}' for i in range(1, len(review) + 1)])
    review['GrupoRevision'] = review['CodigoExterno'].map(lambda c: ', '.join(groups[c]))
    review['AfinidadRevision'] = ''
    review['Observacion'] = ''
    front = ['IdRevision', 'CodigoExterno', 'Nombre', 'GrupoRevision', 'AfinidadRevision', 'AfinidadPredicha_199', 'AfinidadPredicha_299', 'Posicion_199', 'Posicion_299', 'Observacion']
    return review[front + [c for c in review if c not in front]]


def run_comparison(sample, exclusions, models, contract):
    encoder = SentenceTransformer(contract['embedding_model'], device='cpu', local_files_only=True)
    embeddings = encode_tenders(sample['TextoLicitacion'].tolist(), encoder)
    if embeddings.shape[0] != len(sample):
        raise ValueError('La cantidad de embeddings no coincide con los casos.')
    rankings = {size: build_predictions(sample, models[size]['classifier'], embeddings) for size in ['199', '299']}
    comparison = combine_rankings(sample, rankings)
    review = select_review(comparison)
    manifest = {
        'purpose': 'Comparación exploratoria del ranking en casos nuevos para ambos modelos',
        'size': len(sample), 'original_pilot_size': contract['source_size'],
        'labeling_policy_version': '0.2.0', 'profile_version': contract['profile_version'],
        'embedding_model': contract['embedding_model'],
        'input_sha256': contract['input_sha256'],
        'exclusion_counts': dict(Counter(r['Motivo'] for r in exclusions)), 'exclusions': exclusions,
        'prediction_counts': {size: {str(k): int(v) for k, v in rankings[size]['AfinidadPredicha'].value_counts().sort_index().items()} for size in ['199', '299']},
        'disagreements': int(comparison['DesacuerdoAfinidad'].sum()), 'review_size': len(review),
        'ranking_score': 'ProbAfinidad1 + 2 * ProbAfinidad2 + 3 * ProbAfinidad3',
        'note': 'Piloto enriquecido sin etiquetas expertas completas: no calcula accuracy ni estima proporciones del mercado. La exclusión por similitud de tokens no garantiza eliminar toda similitud semántica. Septiembre no se evalúa. Los puntajes de probabilidad de modelos distintos no se tratan como calibrados entre sí.',
        'training_executed': False, 'test_evaluated': False,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    for size in ['199', '299']:
        rankings[size].to_csv(REPORTS / f'affinity_aligned_{size}_pilot_predictions.csv', index=False, encoding='utf-8-sig')
    comparison.to_csv(REPORTS / OUTPUT_NAMES[2], index=False, encoding='utf-8-sig')
    review.to_csv(REPORTS / OUTPUT_NAMES[3], index=False, encoding='utf-8-sig')
    (REPORTS / OUTPUT_NAMES[4]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for size in ['199', '299']:
        print(f'\nPrimeras 10 oportunidades: modelo {size}')
        print(rankings[size][['Posicion', 'CodigoExterno', 'AfinidadPredicha', 'Nombre']].head(10).to_string(index=False))
    print(f'\nCasos nuevos comparados: {len(sample)}; desacuerdos: {manifest["disagreements"]}; casos para revisión: {len(review)}.')
    print(f'Revisión: {REPORTS / OUTPUT_NAMES[3]}')
    print('Se aplicaron modelos guardados; no hubo entrenamiento ni evaluación de septiembre.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Reemplaza únicamente los resultados de esta comparación del piloto.')
    args = parser.parse_args()
    if not args.overwrite and any((REPORTS / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError('Ya existe esta comparación del piloto. Revísala antes de repetir con --overwrite.')
    sample, exclusions, models, contract = load_inputs()
    print(f'Del piloto original de {contract["source_size"]}, se compararán {len(sample)} casos nuevos para ambos modelos.')
    print('Exclusiones:', dict(Counter(r['Motivo'] for r in exclusions)))
    run_comparison(sample, exclusions, models, contract)


if __name__ == '__main__':
    main()
