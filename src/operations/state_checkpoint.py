"""Planifica, crea y recupera un checkpoint local; no contacta servicios cloud.

La recuperación siempre crea una carpeta NUEVA para inspección. No sustituye
el data/ del proyecto ni activa una ejecución. V1 requiere el diario completo.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from uuid import uuid4

from src.reporting import build_batch_history as history


ROOT = Path(__file__).resolve().parents[2]
STATE = "data/processed/incremental/estado.json"
LOCK = "data/processed/incremental/ejecucion.lock"
CODE = ("src/reporting/build_batch_history.py", "src/run_incremental.py",
        "src/run_incremental_batch.py", "src/ingestion/recent_batch.py")
MAX_FILE = 64 * 1024**2
MAX_TOTAL = 512 * 1024**2
MAX_FILES = 20_000
PATTERNS = (
    r"data/raw/batches/[^/]+/(?:captura|listado)\.json",
    r"data/raw/batches/[^/]+/details/[A-Za-z0-9-]+\.json",
    r"data/processed/batches/[^/]+/[^/]+/(?:registro_ejecucion\.json|predicciones\.csv)",
    r"data/processed/reviews/[^/]+/[^/]+\.csv",
    r"data/processed/incremental/estado\.json",
    r"data/processed/incremental/ejecuciones/[^/]+\.json",
    r"data/processed/history/[a-f0-9]{20}/(?:resumen\.json|lotes\.csv|capturas\.csv|lote_captura\.csv|predicciones\.csv|revisiones\.csv|vista_actual\.csv)",
    r"data/labeled/(?:[A-Za-z0-9_-]+/)+criterio\.md",
    r"config/company_profile\.yaml", r"config/cloud/model_bundle\.json",
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("JSON con claves duplicadas.")
        result[key] = value
    return result


def read_json(data):
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_pairs)


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Ruta no admitida en el checkpoint.")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(p in {".", ".."} for p in path.parts):
        raise ValueError("Ruta no admitida en el checkpoint.")
    for part in path.parts:
        if (part.endswith((".", " ")) or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(10)], *[f"LPT{i}" for i in range(10)]}):
            raise ValueError("Ruta incompatible con Windows.")
    if not any(re.fullmatch(pattern, name) for pattern in PATTERNS):
        raise ValueError(f"Archivo fuera de la lista permitida: {name}")
    return name


def _regular(root, name):
    path = root / name
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError(f"No se admiten enlaces: {name}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Falta un archivo regular: {name}")
    if path.stat().st_size > MAX_FILE:
        raise ValueError(f"Archivo demasiado grande para V1: {name}")
    return path


def _code_hashes(root):
    return {name: sha(_regular(root, name).read_bytes()) for name in CODE}


def _operational_files(root):
    names = set()
    for relative in ("data/raw/batches", "data/processed/batches", "data/processed/reviews", "data/processed/incremental"):
        base = root / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError(f"No se admiten enlaces: {path.relative_to(root)}")
            if path.is_dir():
                continue
            name = path.relative_to(root).as_posix()
            if name == LOCK:
                continue
            names.add(safe_name(name))
    return names


def validate_state(root):
    """Valida fuentes e historial referenciado; sólo lee archivos."""
    root = Path(root).resolve()
    state = read_json(_regular(root, STATE).read_bytes())
    if state.get("schema_version") != 1 or state.get("fase") != "completo":
        raise ValueError("V1 exige el diario en fase completo; conserva y revisa una operación pendiente.")
    inventory = history._inventory(root)
    if not inventory["capturas"]:
        raise ValueError("No hay capturas para respaldar.")
    # Valida las rutas antes de permitir que el constructor las lea.
    names = _operational_files(root)
    original_names = set(names)
    for name in names:
        _regular(root, name)
    for group in inventory.values():
        for name in group:
            safe_name(name)
    sources = history._Sources(root)
    lots, captures, links, by_lot, manifests = history._captures(sources, inventory["capturas"])
    predictions, prediction_index, incomplete = history._predictions(
        sources, inventory["predicciones"], captures, by_lot, manifests)
    reviews = history._reviews(sources, inventory["revisiones"], prediction_index)
    # Un archivo sin manifiesto no se interpreta como trabajo terminado.
    raw_lots = {PurePosixPath(name).parts[3] for name in names if name.startswith("data/raw/batches/")}
    if raw_lots != set(manifests):
        raise ValueError("Hay archivos de captura sin manifiesto. Conserva y revisa antes de respaldar.")
    attempts = {str(PurePosixPath(name).parent) for name in names if name.startswith("data/processed/batches/")}
    if attempts != {str(PurePosixPath(name).parent) for name in inventory["predicciones"]}:
        raise ValueError("Hay predicciones sin registro de ejecución.")
    if state.get("lote") is not None and state["lote"] not in manifests:
        raise ValueError("El diario apunta a un lote inexistente.")
    # Revisiones pueden necesitar un criterio fuera de los directorios operativos.
    names.update(safe_name(name) for name in sources.contents)
    snapshot = state.get("historial")
    if not isinstance(snapshot, str) or not re.fullmatch(r"data/processed/history/[a-f0-9]{20}", snapshot):
        raise ValueError("El diario no referencia un historial válido.")
    summary_name = snapshot + "/resumen.json"
    summary = read_json(_regular(root, summary_name).read_bytes())
    inputs = sources.hashes()
    builder_hash = sha(_regular(root, CODE[0]).read_bytes())
    fingerprint = history._identity("history", history.SCHEMA_VERSION, builder_hash, inputs)
    if snapshot.rsplit("/", 1)[1] != fingerprint[:20]:
        raise ValueError("El historial del diario no corresponde a las fuentes actuales.")
    history._verify_snapshot(root / snapshot, fingerprint, inputs, builder_hash)
    if state.get("publicado_bigquery"):
        publication = state.get("publicacion", {})
        if publication.get("historial") != fingerprint or not publication.get("job_id"):
            raise ValueError("El comprobante local de publicación no corresponde al historial.")
    for name in ("resumen.json", *history.TABLE_FIELDS):
        names.add(snapshot + "/" + name)
    names.update(("config/company_profile.yaml", "config/cloud/model_bundle.json"))
    if len(names) > MAX_FILES:
        raise ValueError("Demasiados archivos para V1.")
    payload = {}
    total = 0
    for name in sorted(names):
        safe_name(name)
        path = _regular(root, name)
        total += path.stat().st_size
        if total > MAX_TOTAL:
            raise ValueError("El estado supera el límite de 512 MiB de V1.")
        payload[name] = path.read_bytes()
    sources.verify_unchanged()
    if history._inventory(root) != inventory or _operational_files(root) != original_names:
        raise RuntimeError("El inventario cambió mientras se leía. Repite cuando no haya ejecuciones.")
    counts = {"lotes": len(lots), "capturas": len(captures), "predicciones": len(predictions),
              "revisiones": len(reviews), "vista_actual": len(history._current(captures, predictions, reviews)),
              "intentos_incompletos": incomplete}
    if any(summary["conteos"].get(key) != value for key, value in counts.items() if key != "intentos_incompletos"):
        raise ValueError("Los conteos del historial no coinciden.")
    return payload, counts


def build_plan(root=ROOT, *, locked=False):
    root = Path(root).resolve()
    if not locked and (root / LOCK).exists():
        raise RuntimeError("Hay una ejecución o bloqueo pendiente. No retires el bloqueo automáticamente.")
    code = _code_hashes(root)
    payload, counts = validate_state(root)
    identity = {"schema_version": 1, "purpose": "closed_incremental_state",
                "code_sha256": code, "conteos": counts,
                "files": {name: {"bytes": len(data), "sha256": sha(data)} for name, data in payload.items()}}
    manifest = dict(identity, checkpoint_id=sha(canonical(identity)))
    return {"root": root, "payload": payload, "manifest": manifest,
            "archive": root / "data/processed/checkpoints" / f"estado-{manifest['checkpoint_id'][:20]}.tar.gz"}


def _check_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "closed_incremental_state":
        raise ValueError("Versión de checkpoint no admitida.")
    identity = {key: value for key, value in manifest.items() if key != "checkpoint_id"}
    if manifest.get("checkpoint_id") != sha(canonical(identity)):
        raise ValueError("Cambió la identidad del checkpoint.")
    files = manifest["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Cantidad de archivos inválida.")
    total, folded = 0, set()
    for name, item in files.items():
        safe_name(name)
        if name.casefold() in folded:
            raise ValueError("Rutas duplicadas para Windows.")
        folded.add(name.casefold())
        if type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= MAX_FILE:
            raise ValueError("Tamaño inválido.")
        if not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", ""))):
            raise ValueError("Huella inválida.")
        total += item["bytes"]
    if total > MAX_TOTAL or STATE not in files:
        raise ValueError("Estado ausente o demasiado grande.")
    if set(manifest["code_sha256"]) != set(CODE):
        raise ValueError("Contrato de código inválido.")
    if any(not re.fullmatch(r"[a-f0-9]{64}", str(digest)) for digest in manifest["code_sha256"].values()):
        raise ValueError("Huellas de código inválidas.")


def verify_archive(path, *, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            raise ValueError("SHA256 esperado inválido.")
        if file_sha(path) != expected_sha256:
            raise ValueError("El archivo no coincide con el SHA256 esperado.")
    with tarfile.open(path, "r:gz") as archive:
        # Lectura acotada antes de leer los contenidos. Rechaza enlaces y duplicados.
        members, declared_bytes = {}, 0
        for member in archive:
            declared_bytes += member.size
            if (not member.isfile() or member.name in members or member.size < 0
                    or member.size > MAX_FILE or len(members) >= MAX_FILES + 1
                    or declared_bytes > MAX_TOTAL + 8 * 1024**2):
                raise ValueError("Miembro de archivo no admitido.")
            if member.name != "checkpoint.json":
                safe_name(member.name)
            members[member.name] = member
        if "checkpoint.json" not in members or members["checkpoint.json"].size > 8 * 1024**2:
            raise ValueError("Falta el manifiesto del checkpoint.")
        with archive.extractfile(members["checkpoint.json"]) as stream:
            manifest = read_json(stream.read())
        _check_manifest(manifest)
        if set(members) != {"checkpoint.json", *manifest["files"]}:
            raise ValueError("Archivos añadidos o ausentes.")
        for name, item in manifest["files"].items():
            if members[name].size != item["bytes"]:
                raise ValueError(f"Tamaño alterado: {name}")
            digest = hashlib.sha256()
            with archive.extractfile(members[name]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Contenido alterado: {name}")
    return manifest


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _lock(root):
    path = root / LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    token = canonical({"pid": os.getpid(), "checkpoint": uuid4().hex})
    with path.open("xb") as stream:
        stream.write(token)
    try:
        yield
    finally:
        if path.is_file() and path.read_bytes() == token:
            path.unlink()


def create_checkpoint(root=ROOT):
    root = Path(root).resolve()
    with _lock(root):
        plan = build_plan(root, locked=True)
        output = plan["archive"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = verify_archive(output)
            if existing != plan["manifest"]:
                raise ValueError("El archivo existente tiene otra identidad.")
            return output
        if shutil.disk_usage(output.parent).free < 2 * sum(len(b) for b in plan["payload"].values()) + 64 * 1024**2:
            raise OSError("Falta espacio para crear el checkpoint.")
        temporary = output.with_name(output.name + "." + uuid4().hex + ".tmp")
        created = False
        try:
            with temporary.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    payload = dict(plan["payload"], **{"checkpoint.json": canonical(plan["manifest"])})
                    for name, data in sorted(payload.items()):
                        item = tarfile.TarInfo(name)
                        item.size, item.mode, item.mtime = len(data), 0o600, 0
                        archive.addfile(item, io.BytesIO(data))
            verify_archive(temporary)
            # Detecta cambios de archivos, configuración, código o inventario durante la creación.
            if build_plan(root, locked=True)["manifest"] != plan["manifest"]:
                raise RuntimeError("Las fuentes cambiaron mientras se creaba el checkpoint.")
            with output.open("xb") as destination:
                created = True
                with temporary.open("rb") as source:
                    shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            verify_archive(output)
            return output
        except BaseException:
            if created:
                output.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)


def restore_checkpoint(path, destination, *, code_root=ROOT, expected_sha256=None):
    """Restaura únicamente en un destino inexistente, en forma de staging local."""
    path = Path(path).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("La recuperación necesita una carpeta nueva, que aún no exista.")
    if not destination.parent.is_dir() or destination.parent.resolve() != destination.parent:
        raise ValueError("El directorio padre debe existir y no contener enlaces.")
    required = 2 * path.stat().st_size + MAX_TOTAL + 64 * 1024**2
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError("Falta espacio para verificar y recuperar el checkpoint en staging.")
    with path.open("rb") as source:
        # El archivo privado de trabajo evita leer dos versiones si cambia el original.
        with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=destination.parent) as temporary:
            private = Path(temporary) / "estado.tar.gz"
            with private.open("xb") as out:
                shutil.copyfileobj(source, out)
            manifest = verify_archive(private, expected_sha256=expected_sha256)
            if _code_hashes(Path(code_root).resolve()) != manifest["code_sha256"]:
                raise ValueError("El código difiere del checkpoint. Usa la misma versión y finales de línea.")
            staging = Path(temporary) / "contenido"
            staging.mkdir()
            with tarfile.open(private, "r:gz") as archive:
                for name in manifest["files"]:
                    target = staging.joinpath(*PurePosixPath(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(name) as incoming, target.open("xb") as out:
                        shutil.copyfileobj(incoming, out)
            # El constructor necesita el mismo código para verificar la identidad histórica.
            for name in CODE:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = _regular(Path(code_root).resolve(), name).read_bytes()
                if sha(data) != manifest["code_sha256"][name]:
                    raise RuntimeError("El código cambió durante la recuperación.")
                with target.open("xb") as out:
                    out.write(data)
            payload, counts = validate_state(staging)
            if counts != manifest["conteos"] or set(payload) != set(manifest["files"]):
                raise ValueError("La recuperación no reproduce el inventario.")
            for name, data in payload.items():
                if sha(data) != manifest["files"][name]["sha256"]:
                    raise ValueError("La recuperación perdió integridad.")
            # Reserva exclusiva; nunca mezcla ni sustituye un directorio del usuario.
            destination.mkdir(exist_ok=False)
            try:
                for child in staging.iterdir():
                    shutil.move(str(child), str(destination / child.name))
            except BaseException:
                # Conserva el destino parcial para inspección; no afirma éxito.
                raise RuntimeError(f"Recuperación interrumpida; conserva y revisa {destination}.") from None
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--crear", action="store_true")
    modes.add_argument("--verificar", type=Path)
    modes.add_argument("--recuperar", type=Path)
    parser.add_argument("--destino", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    if args.destino and not args.recuperar:
        parser.error("--destino sólo se usa con --recuperar.")
    if args.sha256 and not (args.verificar or args.recuperar):
        parser.error("--sha256 requiere --verificar o --recuperar.")
    if args.recuperar:
        if not args.destino:
            parser.error("--recuperar requiere --destino.")
        manifest = restore_checkpoint(args.recuperar, args.destino, code_root=args.root, expected_sha256=args.sha256)
        print("Recuperación verificada en carpeta separada:", args.destino)
        print(json.dumps(manifest["conteos"], ensure_ascii=False, indent=2))
    elif args.verificar:
        manifest = verify_archive(args.verificar, expected_sha256=args.sha256)
        print("Integridad del archivo verificada:", manifest["checkpoint_id"])
        print("La prueba de recuperación valida también las relaciones y el historial.")
    else:
        if args.crear:
            path = create_checkpoint(args.root)
            manifest = verify_archive(path)
            print("Checkpoint creado y verificado:", path)
            print("SHA256:", file_sha(path))
            print("Bytes comprimidos:", path.stat().st_size)
        else:
            plan = build_plan(args.root)
            manifest = plan["manifest"]
            print("Destino al crear:", plan["archive"])
            print("Sólo plan: no se escribieron archivos ni se contactó la API, MLflow o Google Cloud.")
        print(json.dumps({"checkpoint_id": manifest["checkpoint_id"], "archivos": len(manifest["files"]),
                          "bytes_sin_comprimir": sum(i["bytes"] for i in manifest["files"].values()),
                          "conteos": manifest["conteos"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
