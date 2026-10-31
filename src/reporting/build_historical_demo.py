"""Genera una demo HTML local con los resultados guardados; no ejecuta modelos."""
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlencode

import yaml

ROOT = Path(__file__).resolve().parents[2]


def tender_url(code):
    if not re.fullmatch(r'[0-9]+-[0-9]+-[A-Z0-9]+', code):
        raise ValueError(f'Código de licitación inválido: {code!r}')
    return 'https://www.mercadopublico.cl/Procurement/Modules/RFB/DetailsAcquisition.aspx?' + urlencode({'idlicitacion': code})


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    text = unicodedata.normalize('NFKD', str(value).lower())
    return ''.join(c for c in text if not unicodedata.combining(c))


def build_demo(root, template, output):
    data = root / 'data/processed'
    report = json.loads((data / 'affinity_aligned_299_test_report.json').read_text(encoding='utf-8'))
    if report['candidate'] != 'affinity_aligned_299' or report['candidate_size'] != 299:
        raise ValueError('El reporte no corresponde al candidato evaluado de 299.')
    if file_hash(root / 'models/affinity_aligned_299.joblib') != report['model_sha256']:
        raise ValueError('El modelo cambió desde la evaluación de septiembre.')
    if file_hash(root / 'data/labeled/temporal_aligned/test.csv') != report['test_sha256']:
        raise ValueError('Las etiquetas de septiembre cambiaron desde la evaluación.')
    first = {r['CodigoExterno']: r for r in read_csv(data / 'affinity_aligned_199_pilot_predictions.csv')}
    second = read_csv(data / 'affinity_aligned_299_pilot_predictions.csv')
    if len(first) != 893 or len(second) != 893 or set(first) != {r['CodigoExterno'] for r in second}:
        raise ValueError('Se esperan los mismos 893 casos del piloto comparativo.')
    pilot_manifest = json.loads((data / 'affinity_aligned_pilot_manifest.json').read_text(encoding='utf-8'))
    if pilot_manifest['input_sha256']['models/affinity_aligned_299.joblib'] != report['model_sha256']:
        raise ValueError('El piloto corresponde a otro modelo distinto del evaluado.')
    for rel, expected in pilot_manifest['input_sha256'].items():
        if file_hash(root / rel) != expected:
            raise ValueError(f'Cambió {rel} respecto de la ejecución del piloto.')
    audit = json.loads((data / 'affinity_aligned_pilot_audit_closed.json').read_text(encoding='utf-8'))
    if file_hash(Path(audit['source'])) != audit['source_sha256']:
        raise ValueError('El archivo de revisión no coincide con la auditoría cerrada.')
    reviews = {r['CodigoExterno']: r for r in audit['cases']}
    profile = yaml.safe_load((root / 'config/company_profile.yaml').read_text(encoding='utf-8'))
    if profile['profile_version'] != report['profile_version']:
        raise ValueError('El perfil no coincide con el usado en la evaluación.')
    area_names = {
        'business_intelligence': 'BI y visualización', 'data_engineering': 'Ingeniería de datos',
        'data_science': 'Analítica y ciencia de datos', 'consulting_and_training': 'Consultoría y capacitación',
    }
    records = []
    for r in second:
        a = first[r['CodigoExterno']]
        if a['TextoLicitacion'] != r['TextoLicitacion']:
            raise ValueError('Los modelos compararon textos diferentes.')
        text = normalized(r['TextoLicitacion'])
        areas, terms = [], []
        for key, service in profile['relevant_services'].items():
            matches = [term for term in service['keywords'] if re.search(r'\b' + re.escape(normalized(term)) + r'\b', text)]
            if matches:
                areas.append(area_names.get(key, key))
                terms.extend(matches)
        review = reviews.get(r['CodigoExterno'])
        records.append({
            'code': r['CodigoExterno'], 'title': r['Nombre'], 'description': r['Descripcion'],
            'url': tender_url(r['CodigoExterno']),
            'text': r['TextoLicitacion'], 'published': r['FechaPublicacion'], 'status': r['Estado'],
            'rank199': int(a['Posicion']), 'rank299': int(r['Posicion']),
            'grade199': int(a['AfinidadPredicha']), 'grade299': int(r['AfinidadPredicha']),
            'score199': float(a['AfinidadEsperada']), 'score299': float(r['AfinidadEsperada']),
            'reviewGrade': int(review['AfinidadPropuesta']) if review else None,
            'human': bool(review and review['ValidadaAfinidadHumano'] == 1),
            'reviewReason': review['MotivoPropuesta'] if review else '',
            'areas': areas, 'terms': sorted(set(terms)),
        })
    matrix = report['confusion_matrix']
    correct = sum(matrix[i][i] for i in range(4))
    relevant_total = sum(sum(row) for row in matrix[2:])
    relevant_recovered = sum(sum(row[2:]) for row in matrix[2:])
    if (correct, relevant_total, relevant_recovered) != (49, 17, 13):
        raise ValueError('Las cifras del reporte cambiaron respecto del resultado revisado.')
    if sum(r['human'] for r in records) != 6:
        raise ValueError('Se esperan las seis revisiones humanas de la auditoría cerrada.')
    payload = {'records': records, 'test': report, 'correct': correct, 'relevantTotal': relevant_total, 'relevantRecovered': relevant_recovered}
    serialized = json.dumps(payload, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html_template = template.read_text(encoding='utf-8')
    if html_template.count('%%DATA_JSON%%') != 1:
        raise ValueError('La plantilla debe contener un único marcador de datos.')
    html = html_template.replace('%%DATA_JSON%%', serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    export = output.with_name('oportunidades_looker.csv')
    with export.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['CodigoExterno', 'Nombre', 'FechaPublicacion', 'EstadoHistorico', 'Posicion299', 'Afinidad299', 'Puntaje299', 'Posicion199', 'Afinidad199', 'Puntaje199', 'AfinidadRevision', 'RevisionConfirmadaHumano', 'UrlMercadoPublico', 'TipoMuestra'])
        writer.writeheader()
        for r in records:
            writer.writerow(dict(zip(writer.fieldnames, [r['code'], r['title'], r['published'], r['status'], r['rank299'], r['grade299'], r['score299'], r['rank199'], r['grade199'], r['score199'], r['reviewGrade'], int(r['human']), r['url'], 'Piloto histórico enero–julio 2026'])))
    print(f'Demo histórica creada: {output}')
    print('Abre el HTML en el navegador. Contiene 893 licitaciones, filtros, detalle, revisiones y evaluación.')
    print('No se consultaron licitaciones en vivo ni se ejecutó entrenamiento o inferencia.')
    print(f'Tabla para Looker Studio: {export}')
    return payload


def main():
    build_demo(ROOT, ROOT / 'assets/historical_demo_template.html', ROOT / 'data/processed/demo_mercado_publico.html')


if __name__ == '__main__':
    main()
