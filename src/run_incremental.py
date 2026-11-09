"""Backfill por rangos: un segmento de hasta diez detalles por ejecución.

Sin --ejecutar sólo lee y describe el trabajo. El avance se reconstruye de
capturas y predicciones verificadas; un diario conserva la operación pendiente.
Una fecha agotada significa agotada en el listado guardado, no mercado completo.
"""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from src.ingestion import recent_batch as batch
from src.reporting import build_batch_history as history
from src.warehouse.load_history import apply_history, prepare_history


ROOT = Path(__file__).resolve().parents[1]
MULTI_SEGMENT_VERSION = 1
STATE_REL = Path("data/processed/incremental/estado.json")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def single_writer(root):
    """Sólo coordina este ejecutor; no lanzar run_pipeline en paralelo."""
    path = root / STATE_REL.parent / "ejecucion.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    token = json.dumps({"pid": os.getpid(), "token": uuid4().hex})
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(token)
    except FileExistsError:
        raise RuntimeError(
            f"Hay otra ejecución o un cierre abrupto. Revisa el bloqueo: {path}. "
            "No lo retires mientras una ejecución siga activa."
        ) from None
    try:
        yield
    finally:
        if path.is_file() and path.read_text(encoding="utf-8") == token:
            path.unlink()


def date_range(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days >= 31:
        raise ValueError("Usa un rango ordenado de entre 1 y 31 días.")
    # Se trabaja con días cerrados, según la fecha local del equipo.
    if last >= date.today():
        raise ValueError("El rango debe terminar antes de hoy.")
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _folder(root, lot_id):
    if not isinstance(lot_id, str) or not lot_id:
        raise ValueError("Identificador de lote inválido.")
    base = (root / "data/raw/batches").resolve()
    path = (base / lot_id).resolve()
    if not isinstance(lot_id, str) or path.parent != base or path.is_symlink():
        raise ValueError("Identificador de lote inválido.")
    return path


@dataclass(frozen=True)
class Listing:
    lot_id: str
    raw: bytes
    captured_at: str
    codes: tuple


def read_inventory(root):
    """Reutiliza las validaciones del historial, sin escribir un snapshot."""
    inventory = history._inventory(root)
    sources = history._Sources(root)
    lots, captures, links, by_lot, manifests = history._captures(sources, inventory["capturas"])
    predictions, prediction_index, _ = history._predictions(
        sources, inventory["predicciones"], captures, by_lot, manifests
    )
    history._reviews(sources, inventory["revisiones"], prediction_index)
    dates = {row["LoteId"]: row["FechaConsultada"] for row in lots}
    done, listings, unfinished = {}, {}, []
    predicted_lots = {row["LoteId"] for row in predictions}
    for row in predictions:
        done.setdefault(dates[row["LoteId"]], set()).add(row["CodigoExterno"])
    for lot_id, (_, manifest) in manifests.items():
        day = manifest["plan"]["fecha_iso"]
        if manifest["estado"] == "completo" and lot_id not in predicted_lots:
            unfinished.append((day, lot_id))
        if not manifest.get("listado"):
            continue
        data = sources.read(Path("data/raw/batches") / lot_id / "listado.json")
        codes = tuple(sorted({batch._code(r.get("CodigoExterno"))
                              for r in batch._records(json.loads(data.decode("utf-8-sig")))}))
        listing = Listing(lot_id, data, manifest["listado"]["captura_utc"], codes)
        old = listings.get(day)
        if old is None or (listing.captured_at, lot_id) > (old.captured_at, old.lot_id):
            listings[day] = listing
    sources.verify_unchanged()
    if history._inventory(root) != inventory:
        raise RuntimeError("Las entradas cambiaron durante el inventario. Repite el plan.")
    return listings, done, sorted(unfinished)


def select_work(days, listings, done, unfinished, limit):
    rows, work = [], None
    wanted = {day.isoformat() for day in days}
    pending = [(day, lot) for day, lot in unfinished if day in wanted]
    for day in days:
        key = day.isoformat()
        listing = listings.get(key)
        if listing is None:
            rows.append({"fecha": key, "listado": "pendiente_de_consultar"})
            if work is None:
                work = {"tipo": "capturar", "fecha": key, "desplazamiento": 0,
                        "limite": limit, "codigos": None}
            continue
        completed = set(listing.codes) & done.get(key, set())
        remaining = [i for i, code in enumerate(listing.codes) if code not in completed]
        rows.append({"fecha": key, "codigos_en_listado": len(listing.codes),
                     "con_prediccion_local": len(completed), "pendientes": len(remaining),
                     "listado_capturado_utc": listing.captured_at})
        if work is None and remaining:
            start = remaining[0]
            stop = start
            # Sólo un tramo consecutivo pendiente; no recaptura un código ya terminado.
            while stop < len(listing.codes) and stop - start < limit:
                if listing.codes[stop] in completed:
                    break
                stop += 1
            work = {"tipo": "capturar", "fecha": key, "desplazamiento": start,
                    "limite": stop - start, "codigos": list(listing.codes[start:stop])}
    if pending:
        work = {"tipo": "predecir_existente", "fecha": pending[0][0], "lote": pending[0][1]}
    return rows, work


def _new_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:8]


def capture_segment(root, lot_id, day, limit, offset, listing=None, original=None, *, control=None):
    """Guarda el listado real completo; cada lote selecciona un tramo verificable."""
    from src.api.mercado_publico_client import get_licitaciones_por_fecha, get_licitacion_por_codigo
    from src.ingestion.capture_store import guardar_captura

    if original is not None:
        previous = original.get("control_api", {})
        if previous.get("bloqueado_por_429"):
            deadline = previous.get("reintentar_despues_utc")
            if not deadline or datetime.now(timezone.utc) < datetime.fromisoformat(deadline):
                raise RuntimeError("La captura anterior requiere esperar por 429 antes de recuperarla.")
    folder = _folder(root, lot_id)
    folder.mkdir(parents=True, exist_ok=False)
    control = control if control is not None else batch._RequestControl()
    first_attempt = control.attempts
    first_event = len(control.events)
    manifest = {"schema_version": 1, "lote_id": lot_id,
                "plan": batch.plan_lote(date.fromisoformat(day).strftime("%d%m%Y"), limit),
                "inicio_utc": batch._utc(), "estado": "fallido", "seleccionados": [],
                "detalles": [], "consultas_api": 0,
                "seleccion_incremental": {"version": 1, "desplazamiento": offset},
                "nota": "Segmento del listado guardado. Completo no equivale a cobertura del día."}
    if original is not None:
        manifest["origen"] = {"lote_id": original["lote_id"],
                                "captura_sha256": batch._hash(_folder(root, original["lote_id"]) / "captura.json")}
    try:
        if listing is None:
            payload = control.request(get_licitaciones_por_fecha, manifest["plan"]["fecha_consulta"])
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            listed_at = batch._utc()
        else:
            raw, listed_at = listing.raw, listing.captured_at
            payload = json.loads(raw.decode("utf-8-sig"))
            manifest["listado_reutilizado_de_lote"] = listing.lot_id
        rows = batch._records(payload)
        codes = sorted({batch._code(row.get("CodigoExterno")) for row in rows})
        selected = codes[offset:offset + limit]
        if codes and not selected:
            raise ValueError("El tramo no existe en el listado guardado.")
        with (folder / "listado.json").open("xb") as stream:
            stream.write(raw)
        manifest.update(filas_listado=len(rows), codigos_unicos_listado=len(codes), seleccionados=selected)
        manifest["listado"] = {"archivo": "listado.json", "sha256": batch._hash(folder / "listado.json"),
                                "captura_utc": listed_at}
        manifest["detalles"] = [{"codigo": code, "estado": "fallido", "motivo": "no_intentado"}
                                 for code in selected]
        reusable = {row["codigo"]: row for row in original["detalles"]
                    if row["estado"] == "correcto"} if original else {}
        for index, code in enumerate(selected):
            if code in reusable:
                entry = dict(reusable[code])
                source = _folder(root, original["lote_id"]) / entry["archivo"]
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError("Cambió un detalle reutilizado.")
                destination = folder / entry["archivo"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(data)
                entry["reutilizado_de_lote"] = original["lote_id"]
                manifest["detalles"][index] = entry
                continue
            before = control.attempts
            entry = batch._capture_detail(code, folder, lambda value: control.request(get_licitacion_por_codigo, value))
            entry["intentos_http"] = control.attempts - before
            manifest["detalles"][index] = entry
            if control.blocked:
                break
    except Exception as error:
        manifest.update(batch._error_info(error))
    finally:
        if manifest.get("listado"):
            manifest["estado"] = ("vacio" if not manifest["codigos_unicos_listado"] else
                                    "completo" if all(row["estado"] == "correcto" for row in manifest["detalles"])
                                    else "incompleto")
        summary = control.summary()
        summary["eventos"] = summary["eventos"][first_event:]
        manifest.update(consultas_api=control.attempts - first_attempt, control_api=summary, fin_utc=batch._utc())
        guardar_captura(manifest, folder / "captura.json")
    print("Lote incremental:", folder, "| Estado:", manifest["estado"], "| Consultas:", manifest["consultas_api"])
    if manifest["estado"] not in {"completo", "vacio"}:
        raise RuntimeError("Captura pendiente. El diario conserva el lote para continuar.")
    return folder


def _prediction(root, lot_id):
    results = []
    for path in (root / "data/processed/batches" / lot_id).glob("*/registro_ejecucion.json"):
        if _read(path).get("estado") == "completo":
            csv_path = path.parent / "predicciones.csv"
            if not csv_path.is_file():
                raise ValueError("Falta el CSV de una predicción completa.")
            results.append(csv_path)
    if len(results) > 1:
        raise ValueError("Hay varias predicciones completas para un lote.")
    return results[0] if results else None


def _target(args):
    return {key: getattr(args, key) for key in ("project", "dataset", "location")}


def publish(root, snapshot, state, args):
    base = (root / "data/processed/history").resolve()
    snapshot = snapshot.resolve()
    if snapshot.parent != base:
        raise ValueError("El historial debe estar en data/processed/history.")
    # Validar antes de marcar un posible envío, para distinguir fallos locales.
    plan = prepare_history(snapshot, root=root)
    state.update(fase="publicacion_en_curso", destino=_target(args),
                 historial=snapshot.relative_to(root).as_posix())
    _atomic_json(root / STATE_REL, state)
    result = apply_history(plan, args.project, args.dataset, location=args.location, max_bytes=args.max_bytes)
    print("Publicación BigQuery confirmada:", json.dumps(result, ensure_ascii=False, indent=2))
    state["publicacion"] = result


def finish(root, state, args):
    folder = _folder(root, state["lote"])
    manifest = _read(folder / "captura.json")
    if manifest["estado"] == "completo":
        batch._load_capture(folder)
        if _prediction(root, folder.name) is None:
            state["fase"] = "prediccion_pendiente"
            _atomic_json(root / STATE_REL, state)
            batch.predecir_lote(folder, raiz=root)
        else:
            print("Reutilizando predicción completa del lote:", folder.name)
    elif manifest["estado"] != "vacio":
        raise ValueError("La captura debe completarse antes de predecir.")
    snapshot = history.build_history(root=root)
    if args.publicar_bigquery:
        publish(root, snapshot, state, args)
    state.update(fase="completo", publicado_bigquery=bool(args.publicar_bigquery), fin_utc=batch._utc())
    _atomic_json(root / STATE_REL, state)
    print("Segmento terminado. Ejecuta el mismo rango para avanzar por los pendientes.")


def execute(root, days, args, *, control=None, locked=False):
    with (nullcontext() if locked else single_writer(root)):
        listings, done, unfinished = read_inventory(root)
        state_path = root / STATE_REL
        state = _read(state_path) if state_path.is_file() else None
        if args.reintentar_publicacion and (not state or state.get("fase") != "publicacion_en_curso"):
            raise ValueError("No hay una publicación sin confirmar para reintentar.")
        if state and state["fase"] != "completo":
            if state["fase"] == "publicacion_en_curso":
                if not args.reintentar_publicacion or not args.publicar_bigquery or state["destino"] != _target(args):
                    raise RuntimeError(
                        "Publicación anterior sin confirmación. Revisa el job en BigQuery. "
                        "Después usa --ejecutar --publicar-bigquery --reintentar-publicacion "
                        "con el mismo destino; no se capturarán códigos nuevos."
                    )
                if state.get("lote") is None:
                    publish(root, root / state["historial"], state, args)
                    state.update(fase="completo", publicado_bigquery=True, fin_utc=batch._utc())
                    _atomic_json(state_path, state)
                    return 0
            folder = _folder(root, state["lote"])
            if not (folder / "captura.json").is_file():
                raise RuntimeError(f"Interrupción sin manifiesto final en {folder}. Conserva los archivos y revisa antes de retomar.")
            manifest = _read(folder / "captura.json")
            if manifest["estado"] == "incompleto":
                batch._load_capture(folder, permitir_incompleto=True)
                missing = sum(row["estado"] != "correcto" for row in manifest["detalles"])
                if missing > args.limite:
                    raise ValueError(f"La recuperación necesita hasta {missing} detalles; aumenta --limite hasta ese valor.")
                previous_control = manifest.get("control_api", {})
                if previous_control.get("bloqueado_por_429"):
                    deadline = previous_control.get("reintentar_despues_utc")
                    if not deadline or datetime.now(timezone.utc) < datetime.fromisoformat(deadline):
                        raise RuntimeError("Todavía no se puede retomar la captura limitada por 429.")
                raw = (folder / "listado.json").read_bytes()
                listing = Listing(folder.name, raw, manifest["listado"]["captura_utc"], ())
                offset = manifest.get("seleccion_incremental", {}).get("desplazamiento", 0)
                new_id = _new_id()
                state.update(lote=new_id, fase="captura_pendiente")
                _atomic_json(state_path, state)
                capture_segment(root, new_id, manifest["plan"]["fecha_iso"], manifest["plan"]["limite"],
                                offset, listing=listing, original=manifest, control=control)
            elif manifest["estado"] == "fallido" and not manifest.get("listado"):
                new_id = _new_id()
                state.update(lote=new_id, fase="captura_pendiente")
                _atomic_json(state_path, state)
                capture_segment(root, new_id, state["fecha"], args.limite, 0, control=control)
            finish(root, state, args)
            return 0
        _, work = select_work(days, listings, done, unfinished, args.limite)
        if work is None:
            print("No quedan pendientes en los listados guardados del rango.")
            if args.publicar_bigquery:
                # Una corrida previa local puede publicarse después sin nueva API.
                snapshot = history.build_history(root=root)
                state = {"schema_version": 1, "fase": "publicacion_pendiente", "lote": None,
                         "inicio_utc": batch._utc()}
                publish(root, snapshot, state, args)
                state.update(fase="completo", publicado_bigquery=True, fin_utc=batch._utc())
                _atomic_json(state_path, state)
            return 0
        lot_id = work.get("lote") or _new_id()
        state = {"schema_version": 1, "fase": "captura_pendiente", "lote": lot_id,
                 "fecha": work["fecha"], "inicio_utc": batch._utc()}
        _atomic_json(state_path, state)
        if work["tipo"] == "capturar":
            capture_segment(root, lot_id, work["fecha"], work["limite"], work["desplazamiento"],
                            listing=listings.get(work["fecha"]), control=control)
        finish(root, state, args)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde", required=True, help="Primera fecha: AAAA-MM-DD.")
    parser.add_argument("--hasta", required=True, help="Última fecha inclusiva: AAAA-MM-DD.")
    parser.add_argument("--limite", type=int, default=10, help="Hasta diez detalles nuevos por ejecución.")
    parser.add_argument("--ejecutar", action="store_true")
    parser.add_argument("--publicar-bigquery", action="store_true")
    parser.add_argument("--reintentar-publicacion", action="store_true")
    parser.add_argument("--project", default="inteligencia-mercado-publico")
    parser.add_argument("--dataset", default="mercado_publico")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--max-bytes", type=int, default=1_073_741_824)
    args = parser.parse_args(argv)
    try:
        days = date_range(args.desde, args.hasta)
        batch.plan_lote(days[0].strftime("%d%m%Y"), args.limite)
        if args.max_bytes <= 0:
            raise ValueError("--max-bytes debe ser positivo.")
        if args.reintentar_publicacion and not (args.ejecutar and args.publicar_bigquery):
            raise ValueError("--reintentar-publicacion requiere --ejecutar y --publicar-bigquery.")
    except ValueError as error:
        parser.error(str(error))
    root = ROOT.resolve()
    listings, done, unfinished = read_inventory(root)
    rows, work = select_work(days, listings, done, unfinished, args.limite)
    state_path = root / STATE_REL
    state = _read(state_path) if state_path.is_file() else None
    pending = state if state and state.get("fase") != "completo" else None
    print(json.dumps({"cobertura": rows, "siguiente_segmento": work, "operacion_pendiente": pending,
                      "limite_detalles_nuevos": args.limite,
                      "nota": "Orden cronológico; listados guardados; una sola fecha por ejecución. "
                              "Predicción local no equivale a publicación confirmada ni a cobertura total del mercado."},
                     ensure_ascii=False, indent=2))
    if not args.ejecutar:
        print("Sólo plan: no se contactó la API ni BigQuery, no se cargó el modelo ni se escribieron archivos.")
        return 0
    return execute(root, days, args)


if __name__ == "__main__":
    raise SystemExit(main())
