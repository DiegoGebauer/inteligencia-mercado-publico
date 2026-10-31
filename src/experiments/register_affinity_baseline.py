"""Importa un resultado histórico en MLflow local; no entrena ni predice."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'Proyecto 2 - Afinidad Mercado Publico'
REGISTERED_MODEL = 'mercado_publico_afinidad_embeddings'
REPORT = 'data/processed/affinity_aligned_299_test_report.json'
SELECTION = 'data/labeled/temporal_aligned/frozen_candidate.json'
MODEL = 'models/affinity_aligned_299.joblib'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inspect_baseline(root):
    """Comprueba los archivos antes de escribir en MLflow o cargar el joblib."""
    report = read_json(root / REPORT)
    selection = read_json(root / SELECTION)
    if report['candidate'] != 'affinity_aligned_299' or report['candidate_size'] != 299:
        raise ValueError('El reporte no corresponde al candidato de 299.')
    if selection['candidate_model'] != MODEL or selection['candidate_size'] != 299:
        raise ValueError('La selección congelada corresponde a otro modelo.')
    if digest(root / SELECTION) != report['frozen_selection_sha256']:
        raise ValueError('Cambió la selección congelada.')
    expected = dict(selection['input_sha256'])
    expected[MODEL] = report['model_sha256']
    expected['data/labeled/temporal_aligned/test.csv'] = report['test_sha256']
    for rel, sha in expected.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != sha:
            raise ValueError(f'El archivo no coincide con el resultado congelado: {rel}')
    splits = {}
    for name, filename, size in [('train', 'train_299.csv', 299), ('validation', 'validation.csv', 60), ('test', 'test.csv', 60)]:
        path = root / 'data/labeled/temporal_aligned' / filename
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != size or len({r['CodigoExterno'] for r in rows}) != size:
            raise ValueError(f'Cantidad o códigos inesperados en {name}.')
        splits[name] = {'rows': size, 'months': sorted({r['FechaPublicacion'][:7] for r in rows}), 'sha256': digest(path)}
    if splits['train']['months'] != ['2026-01', '2026-02', '2026-04', '2026-05', '2026-06', '2026-07'] or splits['validation']['months'] != ['2026-08'] or splits['test']['months'] != ['2026-09']:
        raise ValueError('Los meses no coinciden con el corte histórico.')
    comparison = read_json(root / 'data/processed/affinity_aligned_comparison.json')
    validation = comparison['aligned_models']['299']
    if validation['metrics'] != selection['validation_metrics']['299']:
        raise ValueError('Las métricas de validación no coinciden con la selección.')
    for metrics in [report['metrics'], validation['metrics']]:
        if any(not math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError('Hay métricas no finitas.')
    identity = {'report': digest(root / REPORT), 'selection': digest(root / SELECTION), 'model': digest(root / MODEL)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return report, selection, validation, splits, identity, fingerprint


def register_baseline(root):
    report, selection, validation, splits, identity, fingerprint = inspect_baseline(root)
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec
        from mlflow.tracking import MlflowClient
        import joblib
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError('Instala primero requirements-mlflow.txt con constraints-mlflow.txt; consulta docs/mlflow_baseline.md.') from exc
    if mlflow.__version__ != '3.16.1':
        raise RuntimeError('Este paquete requiere MLflow 3.16.1 para mantener la ejecución reproducible.')
    saved = joblib.load(root / MODEL)
    classifier = saved['classifier']
    if saved['embedding_model'] != selection['embedding_model'] or saved['chosen_C'] != validation['chosen_C']:
        raise ValueError('La configuración guardada no coincide con la selección.')
    if saved['profile_version'] != report['profile_version'] or saved['labeling_policy_version'] != report['labeling_policy_version']:
        raise ValueError('La versión del perfil o criterio no coincide.')
    if set(classifier.classes_) != {0, 1, 2, 3} or classifier.n_features_in_ != 384:
        raise ValueError('El clasificador no tiene las cuatro clases y 384 dimensiones esperadas.')
    store = root / 'mlflow_local'
    artifacts = store / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    tracking_uri = 'sqlite:///' + (store / 'mlflow.db').resolve().as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT, artifact_location=artifacts.resolve().as_uri())
    previous = client.search_runs([experiment_id], filter_string=f"tags.baseline_fingerprint = '{fingerprint}' and attributes.status = 'FINISHED'", max_results=10)
    if previous:
        run_id = previous[0].info.run_id
        versions = [v for v in client.search_model_versions(f"name = '{REGISTERED_MODEL}'") if v.run_id == run_id]
        if not versions:
            raise RuntimeError('El run ya existe, pero falta su versión en Models. Conserva el registro y revisa antes de repetir.')
        version = str(max(versions, key=lambda v: int(v.version)).version)
        reused = True
    else:
        params = {
            'candidate': report['candidate'], 'train_size': splits['train']['rows'], 'validation_size': 60, 'test_size': 60,
            'C': saved['chosen_C'], 'embedding_model': saved['embedding_model'], 'pooling': saved['pooling'],
            'input_dimensions': classifier.n_features_in_, 'target': saved['target'], 'dataset_version': saved['dataset_version'],
            'profile_version': report['profile_version'], 'labeling_policy_version': report['labeling_policy_version'],
            'train_months': ','.join(splits['train']['months']), 'validation_month': '2026-08', 'test_month': '2026-09',
        }
        versions_env = {name: importlib.metadata.version(name) for name in ['mlflow', 'scikit-learn', 'numpy', 'scipy', 'joblib', 'pandas', 'cloudpickle', 'sentence-transformers']}
        tags = {
            'baseline_fingerprint': fingerprint, 'record_type': 'historical_import', 'training_executed': 'false', 'evaluation_executed': 'false',
            'model_sha256': identity['model'], 'test_report_sha256': identity['report'],
            'selection_reason': report['selection_reason'], 'model_scope': 'classifier_on_384_float32_embeddings',
            'label_quality': 'mostly_assisted', 'test_human_confirmed': str(report['test_human_confirmed']),
            'notes': 'Fecha del run: importación. El entrenamiento y la evaluación ocurrieron antes. El encoder E5 y la preparación de textos están fuera del clasificador registrado.',
        }
        with mlflow.start_run(experiment_id=experiment_id, run_name='baseline_299_importacion_historica', tags=tags) as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_metrics({'validation_' + k: float(v) for k, v in validation['metrics'].items()})
            mlflow.log_metrics({'test_' + k: float(v) for k, v in report['metrics'].items()})
            mlflow.log_dict({'splits': splits, 'identity': identity, 'environment_at_import': versions_env, 'historical_import': True}, 'provenance/baseline.json')
            mlflow.log_dict({'train': saved['train_codes'], 'validation': saved['validation_codes']}, 'provenance/codes.json')
            files = set(selection['input_sha256']) | {REPORT, SELECTION, 'config/company_profile.yaml', 'data/labeled/temporal_aligned/criterio.md', 'data/labeled/temporal_aligned/manifest.json', 'src/experiments/register_affinity_baseline.py'}
            for rel in sorted(files):
                path = root / rel
                mlflow.log_artifact(str(path), artifact_path='baseline/' + Path(rel).parent.as_posix())
            signature = ModelSignature(inputs=Schema([TensorSpec(np.dtype('float32'), (-1, 384))]), outputs=Schema([TensorSpec(np.dtype('int64'), (-1,))]))
            info = mlflow.sklearn.log_model(
                sk_model=classifier, name='classifier_embeddings', signature=signature,
                registered_model_name=REGISTERED_MODEL, await_registration_for=60,
                serialization_format='cloudpickle',
                pip_requirements=[f'{name}=={versions_env[name]}' for name in ['mlflow', 'scikit-learn', 'numpy', 'scipy', 'joblib', 'cloudpickle']],
                metadata={'input': 'E5 normalized mean embeddings; not raw tender text', 'original_joblib_sha256': identity['model']},
            )
            version = str(info.registered_model_version)
            if version in {'None', ''}:
                raise RuntimeError('MLflow no devolvió una versión del modelo registrado.')
        reused = False
        client.set_model_version_tag(REGISTERED_MODEL, version, 'original_joblib_sha256', identity['model'])
        client.set_model_version_tag(REGISTERED_MODEL, version, 'scope', 'classifier_on_embeddings')
        client.set_model_version_tag(REGISTERED_MODEL, version, 'record_type', 'historical_import')
        client.set_registered_model_alias(REGISTERED_MODEL, 'candidato', version)
    receipt = {'tracking_uri': tracking_uri, 'experiment_id': experiment_id, 'run_id': run_id, 'registered_model': REGISTERED_MODEL, 'model_version': version, 'model_uri': f'models:/{REGISTERED_MODEL}/{version}', 'model_sha256': identity['model'], 'historical_import': True, 'scope': 'classifier_on_384_float32_embeddings'}
    destination = root / 'data/processed/affinity_mlflow_baseline_registration.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('Registro existente reutilizado.' if reused else 'Baseline histórica registrada correctamente.')
    print(f'Experimento: {EXPERIMENT}\nRun: {run_id}\nModelo: {REGISTERED_MODEL}\nVersión: {version}')
    print(f'Comprobante: {destination}')
    print('No se entrenó ni se ejecutaron predicciones. Las métricas se importaron de los reportes guardados.')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verifica el resultado histórico sin escribir ni necesitar MLflow.')
    parser.add_argument('--project-root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.check:
        _, _, _, splits, _, _ = inspect_baseline(root)
        print(json.dumps({'baseline_verified': True, 'splits': splits}, indent=2))
    else:
        register_baseline(root)


if __name__ == '__main__':
    main()
