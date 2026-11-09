"""Empaquetado con archivos inventados; sin abrir joblib ni cargar E5."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from src.experiments import export_model_bundle as bundle


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    root, encoder = tmp_path / "project", tmp_path / "cache"
    model_sha = write(root / "models/affinity_aligned_299.joblib", b"modelo-ficticio-no-deserializar")
    monkeypatch.setattr(bundle, "MODEL_SHA", model_sha)
    receipt = {"registered_model": bundle.MODEL_NAME, "model_version": "1", "model_sha256": model_sha,
               "run_id": "run-test", "model_uri": f"models:/{bundle.MODEL_NAME}/1"}
    write(root / "data/processed/affinity_mlflow_baseline_registration.json", bundle.canonical(receipt))
    hashes = {}
    for relative in bundle.CODE_FILES:
        hashes[relative] = write(root / relative, b"# codigo ficticio\n")
    for relative in ["config/company_profile.yaml", "data/labeled/temporal_aligned/criterio.md"]:
        hashes[relative] = write(root / relative, b"configuracion de referencia\n")
    output = io.StringIO(newline="")
    fields = ["CodigoExterno", "TextoLicitacion", "HashTextoSHA256", "AfinidadPredicha"]
    fields += [f"ProbAfinidad{i}" for i in range(4)]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for i in range(5):
        text = f"Texto de prueba {i}"
        row = {"CodigoExterno": f"{i + 1}-1-LE26", "TextoLicitacion": text,
               "HashTextoSHA256": hashlib.sha256(text.encode()).hexdigest(), "AfinidadPredicha": "0"}
        row.update({f"ProbAfinidad{j}": value for j, value in enumerate([0.7, 0.1, 0.1, 0.1])})
        writer.writerow(row)
    base = root / "data/processed/batches" / bundle.REFERENCE_LOT / "intento"
    csv_sha = write(base / "predicciones.csv", output.getvalue().encode())
    log = {"estado": "completo", "filas": 5, "predicciones_sha256": csv_sha,
           "identidad_modelo": {"HashModeloOriginalSHA256": model_sha, "ModeloRunId": "run-test",
                                "ModeloURI": receipt["model_uri"], "Encoder": bundle.ENCODER,
                                "RevisionEncoderLocal": bundle.REVISION,
                                "VersionPerfil": "0.1.0", "VersionCriterio": "0.2.0"},
           "versiones": {name: "1.0-test" for name in ["numpy", "scikit-learn", "pandas",
                                                        "sentence-transformers", "transformers", "torch"]},
           "python": "3.13-test", "originales_verificados_sha256": hashes}
    write(base / "registro_ejecucion.json", bundle.canonical(log))
    for name in bundle.ENCODER_FILES:
        write(encoder / name, b"archivo encoder ficticio")
    write(encoder / "config.json", bundle.canonical({"hidden_size": 384, "max_position_embeddings": 512}))
    write(encoder / "modules.json", bundle.canonical([
        {"path": "", "type": "sentence_transformers.models.Transformer"},
        {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ]))
    return root, encoder


def test_plan_solo_lectura_y_lista_cerrada_de_archivos(env):
    root, encoder = env
    write(root / ".env", b"SECRETO_NO_EXPORTAR")
    write(root / "mlflow_local/mlflow.db", b"no incluir")
    before = sorted(str(path) for path in root.rglob("*"))
    plan = bundle.build_plan(root, encoder)
    names = {item.name for item in plan["payloads"]}
    assert len(names) == 14
    assert not any(".env" in name or "mlflow" in name for name in names)
    assert before == sorted(str(path) for path in root.rglob("*"))


def test_rechaza_modelo_modificado(env):
    root, encoder = env
    (root / "models/affinity_aligned_299.joblib").write_bytes(b"otro modelo")
    with pytest.raises(ValueError, match="clasificador cambió"):
        bundle.build_plan(root, encoder)


def test_rechaza_csv_de_referencia_modificado(env):
    root, encoder = env
    path = next((root / "data/processed/batches").rglob("predicciones.csv"))
    path.write_text("csv modificado", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV de referencia"):
        bundle.build_plan(root, encoder)


@pytest.mark.parametrize("relative", ["config/company_profile.yaml", "src/nlp/train_affinity.py"])
def test_rechaza_configuracion_o_codigo_cambiados(env, relative):
    root, encoder = env
    (root / relative).write_bytes(b"cambio no registrado")
    with pytest.raises(ValueError, match="Cambió"):
        bundle.build_plan(root, encoder)


def test_rechaza_encoder_de_otra_dimension(env):
    root, encoder = env
    (encoder / "config.json").write_text('{"hidden_size":768,"max_position_embeddings":512}', encoding="utf-8")
    with pytest.raises(ValueError, match="encoder no cumple"):
        bundle.build_plan(root, encoder)


def test_encoder_ausente_no_inicia_descarga(env):
    root, encoder = env
    with pytest.raises(FileNotFoundError):
        bundle.build_plan(root, encoder / "inexistente")


def test_paquete_creado_tiene_archivos_verificados_y_directorio_normalize(env):
    root, encoder = env
    plan = bundle.build_plan(root, encoder)
    path = bundle.create_archive(plan)
    manifest = bundle.verify_archive(path, plan["manifest"]["bundle_id"])
    assert manifest["model"]["sha256"] == bundle.MODEL_SHA
    with tarfile.open(path, "r:gz") as archive:
        assert archive.getmember("encoder/2_Normalize").isdir()
        assert len(archive.getmembers()) == 17


def test_repetir_exportacion_reutiliza_sin_sobrescribir(env):
    root, encoder = env
    plan = bundle.build_plan(root, encoder)
    path = bundle.create_archive(plan)
    before = (path.stat().st_mtime_ns, bundle.sha_file(path))
    assert bundle.create_archive(plan) == path
    assert before == (path.stat().st_mtime_ns, bundle.sha_file(path))


def test_fuente_cambiada_despues_del_plan_no_publica_paquete(env):
    root, encoder = env
    plan = bundle.build_plan(root, encoder)
    (encoder / "model.safetensors").write_bytes(b"archivo cambiado de longitud suficiente")
    with pytest.raises((ValueError, OSError)):
        bundle.create_archive(plan)
    assert not plan["archive"].exists()
    assert not list(plan["archive"].parent.glob("*.tmp"))


@pytest.mark.parametrize("kind", ["traversal", "symlink", "extra"])
def test_rechaza_rutas_enlaces_y_miembros_extra(env, kind):
    root, encoder = env
    path = bundle.create_archive(bundle.build_plan(root, encoder))
    bad = root / f"{kind}.tar.gz"
    with tarfile.open(path, "r:gz") as source, tarfile.open(bad, "w:gz") as target:
        for member in source.getmembers():
            data = source.extractfile(member).read() if member.isfile() else None
            target.addfile(member, io.BytesIO(data) if data is not None else None)
        member = tarfile.TarInfo("../fuera" if kind == "traversal" else "intruso")
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "../fuera"
        target.addfile(member)
    with pytest.raises(ValueError):
        bundle.verify_archive(bad)
    assert not (root.parent / "fuera").exists()


def test_archivo_existente_invalido_no_se_reemplaza(env):
    root, encoder = env
    plan = bundle.build_plan(root, encoder)
    plan["archive"].parent.mkdir(parents=True)
    plan["archive"].write_bytes(b"no es un tar valido")
    with pytest.raises(tarfile.TarError):
        bundle.create_archive(plan)
    assert plan["archive"].read_bytes() == b"no es un tar valido"


def test_cli_plan_no_crea_archivo(env, monkeypatch):
    root, encoder = env
    monkeypatch.setattr(bundle, "ROOT", root)
    assert bundle.main(["--encoder-dir", str(encoder)]) == 0
    assert not (root / "data/processed/deployment").exists()


def test_identidad_esperada_distinta_se_rechaza(env):
    root, encoder = env
    path = bundle.create_archive(bundle.build_plan(root, encoder))
    with pytest.raises(ValueError, match="identidad"):
        bundle.verify_archive(path, expected_id="0" * 64)
