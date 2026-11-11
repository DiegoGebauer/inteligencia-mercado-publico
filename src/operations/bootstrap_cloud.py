"""Descarga por generación y prepara un workspace aislado, sin captura o inferencia.

Sin --preparar sólo lee código y referencias locales. Con --preparar necesita
ADC y realiza lecturas de Cloud Storage. No ejecuta el pipeline ni escribe cloud.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote

from src.experiments import export_model_bundle as bundle
from src.inference import portable_model as portable
from src.operations import state_checkpoint as checkpoint


ROOT = Path(__file__).resolve().parents[2]
PROJECT = "inteligencia-mercado-publico"
MODEL_REF = "config/cloud/model_bundle.json"
STATE_REF = "config/cloud/state_checkpoint.json"
SCHEMAS = "config/bigquery/history_schemas.json"
MAX_SOURCE_BYTES = 8 * 1024**2


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def parse_object(reference):
    if not isinstance(reference, dict):
        raise ValueError("Referencia de objeto inválida.")
    if not re.fullmatch(r"[a-f0-9]{64}", str(reference.get("id", ""))):
        raise ValueError("Falta la identidad del paquete.")
    if not re.fullmatch(r"[a-f0-9]{64}", str(reference.get("sha256", ""))):
        raise ValueError("Falta un SHA256 válido.")
    if not isinstance(reference.get("generation"), str) or not re.fullmatch(r"[1-9][0-9]*", reference["generation"]):
        raise ValueError("La generación debe ser un entero positivo guardado como texto.")
    if type(reference.get("bytes")) is not int or not 0 < reference["bytes"] <= 1024**3:
        raise ValueError("Tamaño de objeto inválido.")
    if reference.get("format") != "tar.gz":
        raise ValueError("Formato de objeto no admitido.")
    uri = reference.get("gcs_uri", "")
    match = re.fullmatch(r"gs://([a-z0-9][a-z0-9._-]+)/([^\s?#\\]+)", uri) if isinstance(uri, str) else None
    if not match:
        raise ValueError("La referencia debe ser una URI gs:// sin query ni fragmento.")
    bucket, name = match.groups()
    return {"url": f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/{quote(name, safe='')}",
            "params": {"alt": "media", "generation": reference["generation"]}}


def _read_regular(root, name):
    root = Path(root).resolve()
    path = root / name
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"No se admiten enlaces en la preparación: {name}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Falta un archivo regular: {name}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Un archivo de código o configuración es demasiado grande.")
    return path.read_bytes()


def source_files(root):
    root = Path(root).resolve()
    names = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")}
    names.update(p.relative_to(root).as_posix() for p in (root / "sql/bigquery").rglob("*.sql"))
    names.update((MODEL_REF, STATE_REF, SCHEMAS))
    if not set(checkpoint.CODE).issubset(names) or not set(portable.RUNTIME_FILES).difference({"config/company_profile.yaml"}).issubset(names):
        raise ValueError("El checkout no contiene el código requerido por el ejecutor portable.")
    files, total = {}, 0
    for name in sorted(names):
        data = _read_regular(root, name)
        total += len(data)
        if total > MAX_SOURCE_BYTES or len(files) >= 500:
            raise ValueError("El código/configuración supera el límite previsto para la preparación.")
        files[name] = data
    return files


def build_plan(root=ROOT, destination=None):
    root = Path(root).resolve()
    files = source_files(root)
    model = checkpoint.read_json(files[MODEL_REF])
    state = checkpoint.read_json(files[STATE_REF])
    if model.get("schema_version") != 1 or state.get("schema_version") != 1 or state.get("purpose") != "closed_incremental_state":
        raise ValueError("Versión de referencias no admitida.")
    objects = {"modelo": model["bundle"], "estado": state["checkpoint"]}
    for reference in objects.values():
        parse_object(reference)
    if type(objects["estado"].get("files")) is not int or not 1 <= objects["estado"]["files"] <= checkpoint.MAX_FILES:
        raise ValueError("La referencia del estado necesita su cantidad de archivos.")
    if not isinstance(state.get("counts"), dict):
        raise ValueError("Faltan los conteos esperados del estado.")
    if destination is None:
        destination = root / "data/processed/arranque_cloud_prueba_01"
    else:
        destination = Path(destination)
        if not destination.is_absolute():
            destination = root / destination
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Usa una carpeta de destino nueva que todavía no exista.")
    if not destination.parent.is_dir() or destination.parent.resolve() != destination.parent:
        raise ValueError("El directorio padre debe existir y no contener enlaces.")
    if destination == root or root.is_relative_to(destination) or (root / "src").is_relative_to(destination):
        raise ValueError("El destino no puede contener el checkout original.")
    identity = {"schema_version": 1, "project": PROJECT, "objects": objects,
                "code_and_config_sha256": {name: checkpoint.sha(data) for name, data in files.items()}}
    return {"root": root, "destination": destination, "files": files,
            "model": model, "state": state, "identity": identity,
            "plan_id": checkpoint.sha(checkpoint.canonical(identity))}


def plan_summary(plan):
    return {"plan_id": plan["plan_id"], "destino_local": str(plan["destination"]),
            "objetos": plan["identity"]["objects"],
            "bytes_a_descargar": sum(ref["bytes"] for ref in plan["identity"]["objects"].values()),
            "conteos_esperados": plan["state"]["counts"],
            "inferencias": 0, "consultas_mercado_publico": 0, "escrituras_cloud": 0}


def authorized_session():
    # Importar o planificar no busca credenciales ni usa la red.
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
                                         quota_project_id=PROJECT)
    return AuthorizedSession(credentials, max_refresh_attempts=1, refresh_timeout=30)


def download_object(reference, destination, session):
    """Un GET de la generación fijada; conserva un .part si se interrumpe."""
    request = parse_object(reference)
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() or partial.exists():
        raise FileExistsError("La descarga necesita archivos de destino nuevos.")
    start = time.monotonic()
    with session.get(request["url"], params=request["params"], stream=True, timeout=(10, 120),
                     headers={"Accept-Encoding": "identity"}, allow_redirects=False) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Cloud Storage respondió HTTP {response.status_code}; no se descargó la generación esperada.")
        generation = response.headers.get("x-goog-generation")
        if generation is not None and str(generation) != reference["generation"]:
            raise ValueError("La respuesta corresponde a otra generación.")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != reference["bytes"]:
            raise ValueError("El tamaño informado por Cloud Storage difiere de la referencia.")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise ValueError("La respuesta usa una codificación inesperada; no se transformará el archivo.")
        digest, received = hashlib.sha256(), 0
        with partial.open("xb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if time.monotonic() - start > 600:
                    raise TimeoutError("La descarga superó el tiempo permitido.")
                if not chunk:
                    continue
                received += len(chunk)
                if received > reference["bytes"]:
                    raise ValueError("La descarga excede el tamaño aprobado.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if received != reference["bytes"] or digest.hexdigest() != reference["sha256"]:
        raise ValueError("La descarga no coincide con el tamaño/SHA256 aprobado.")
    with destination.open("xb") as output, partial.open("rb") as source:
        shutil.copyfileobj(source, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if bundle.sha_file(destination) != reference["sha256"]:
        raise ValueError("El archivo final no conserva su SHA256.")
    partial.unlink()
    return destination


def _install_source(plan, workspace):
    for name, data in plan["files"].items():
        # El modelo referenciado dentro del checkpoint debe conservar sus bytes.
        if name == MODEL_REF:
            if checkpoint.read_json((workspace / name).read_bytes()) != plan["model"]:
                raise ValueError("El checkpoint referencia un modelo distinto del que se descargó.")
            continue
        target = workspace / name
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"El código recuperado difiere del checkout actual: {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)


def prepare_workspace(root=ROOT, destination=None, *, session_factory=authorized_session):
    plan = build_plan(root, destination)
    required = 2 * sum(obj["bytes"] for obj in plan["identity"]["objects"].values()) + checkpoint.MAX_TOTAL + 128 * 1024**2
    if shutil.disk_usage(plan["destination"].parent).free < required:
        raise OSError("Falta espacio local para descargar y verificar los dos paquetes.")
    target = plan["destination"]
    target.mkdir(exist_ok=False)
    downloads = target / "descargas"
    downloads.mkdir()
    report_path = target / "bootstrap.json"
    report = {"schema_version": 1, "plan_id": plan["plan_id"], "estado": "en_curso", "fase": "inicio",
              "inicio_utc": _utc(), "referencias": plan["identity"]["objects"],
              "codigo_y_config_sha256": plan["identity"]["code_and_config_sha256"], "escrituras_cloud": 0}

    def save():
        # Esta carpeta pertenece sólo a este intento; no es el diario incremental.
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    session = None
    try:
        session = session_factory()
        paths = {}
        for kind, filename in (("modelo", "modelo.tar.gz"), ("estado", "estado.tar.gz")):
            report["fase"] = "descarga_" + kind
            save()
            print("Descargando generación fijada:", kind, flush=True)
            paths[kind] = download_object(plan["identity"]["objects"][kind], downloads / filename, session)
        report["fase"] = "validacion_estado"
        save()
        manifest = checkpoint.verify_archive(paths["estado"], expected_sha256=plan["state"]["checkpoint"]["sha256"])
        if (manifest["checkpoint_id"] != plan["state"]["checkpoint"]["id"]
                or manifest["conteos"] != plan["state"]["counts"]
                or len(manifest["files"]) != plan["state"]["checkpoint"]["files"]):
            raise ValueError("La identidad o conteos del estado no coinciden con la referencia.")
        if checkpoint._code_hashes(plan["root"]) != manifest["code_sha256"]:
            raise ValueError("El checkpoint necesita otra versión del código; no se restaurará automáticamente.")
        report["fase"] = "recuperacion_local"
        save()
        workspace = target / "workspace"
        checkpoint.restore_checkpoint(paths["estado"], workspace, code_root=plan["root"],
                                      expected_sha256=plan["state"]["checkpoint"]["sha256"])
        _install_source(plan, workspace)
        report["fase"] = "validacion_modelo"
        save()
        portable.inspect_bundle(paths["modelo"], workspace)
        if source_files(plan["root"]) != plan["files"]:
            raise RuntimeError("Cambió el código o una referencia mientras se preparaba el workspace.")
        report.update(estado="preparado", fase="listo", workspace=str(workspace),
                      modelo_local=str(paths["modelo"]), conteos=manifest["conteos"],
                      nota="Preparación local verificada desde GCS. No ejecuta captura, inferencia ni publicación. No persiste nuevo avance ni coordina trabajadores.")
        return report
    except BaseException as error:
        report.update(estado="interrumpido" if isinstance(error, KeyboardInterrupt) else "fallido", tipo_error=type(error).__name__)
        raise
    finally:
        if session is not None:
            session.close()
        report["fin_utc"] = _utc()
        save()
        print("Registro del intento:", report_path, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--destino", type=Path, default=None)
    parser.add_argument("--preparar", action="store_true")
    args = parser.parse_args(argv)
    if not args.preparar:
        print(json.dumps(plan_summary(build_plan(args.root, args.destino)), ensure_ascii=False, indent=2))
        print("Sólo plan: sin credenciales, red, inferencia ni escritura de archivos.")
        return 0
    report = prepare_workspace(args.root, args.destino)
    print("Workspace recuperado y modelo validado:", report["workspace"])
    print(json.dumps({"modelo_local": report["modelo_local"], "conteos": report["conteos"],
                      "estado": report["estado"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
