"""Ejecuta varios segmentos incrementales con presupuestos y un único bloqueo.

Sin --ejecutar sólo muestra el plan. Cada segmento conserva la recuperación
del ejecutor anterior y, si se solicita, confirma BigQuery antes de seguir.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from uuid import uuid4

from src import run_incremental as inc


ROOT = Path(__file__).resolve().parents[1]


def pending_state(root):
    path = root / inc.STATE_REL
    state = inc._read(path) if path.is_file() else None
    return state if state and state.get("fase") != "completo" else None


def ordered_days(days, order):
    if order not in {"recientes", "antiguas"}:
        raise ValueError("Orden desconocido.")
    return sorted(days, reverse=order == "recientes")


def rotate_days(days, processed_day):
    """Un segmento por fecha antes de volver a la misma, dentro de la corrida."""
    return ([day for day in days if day.isoformat() != processed_day]
            + [day for day in days if day.isoformat() == processed_day])


def next_task(root, days, limit):
    listings, done, unfinished = inc.read_inventory(root)
    coverage, work = inc.select_work(days, listings, done, unfinished, limit)
    pending = pending_state(root)
    if pending:
        lot_id = pending.get("lote")
        if pending["fase"] == "publicacion_en_curso":
            return coverage, {"tipo": "reanudar_publicacion", "lote": lot_id,
                              "fecha": pending.get("fecha"), "detalles": 0, "listados": 0}, pending
        folder = inc._folder(root, lot_id)
        if not (folder / "captura.json").is_file():
            raise RuntimeError(f"Operación sin manifiesto final: {folder}. Revisa antes de continuar.")
        manifest = inc._read(folder / "captura.json")
        if manifest["estado"] == "incompleto":
            details = sum(row["estado"] != "correcto" for row in manifest["detalles"])
            return coverage, {"tipo": "recuperar_captura", "lote": lot_id,
                              "fecha": manifest["plan"]["fecha_iso"], "detalles": details, "listados": 0}, pending
        if manifest["estado"] == "fallido" and not manifest.get("listado"):
            return coverage, {"tipo": "reintentar_listado", "lote": lot_id,
                              "fecha": pending["fecha"], "detalles": limit, "listados": 1}, pending
        return coverage, {"tipo": "terminar_lote", "lote": lot_id,
                          "fecha": manifest["plan"]["fecha_iso"], "detalles": 0, "listados": 0}, pending
    if work is None:
        return coverage, None, None
    task = dict(work)
    if work["tipo"] == "predecir_existente":
        task.update(detalles=0, listados=0)
    else:
        task.update(detalles=work["limite"], listados=int(work["codigos"] is None))
    return coverage, task, None


def new_details(root, http_calls):
    """Cuenta códigos que se intentaron consultar ahora; no suma copias históricas."""
    if not http_calls:
        return 0
    state = inc._read(root / inc.STATE_REL)
    path = inc._folder(root, state["lote"]) / "captura.json"
    if not path.is_file():
        return 0
    return sum(row.get("intentos_http", 0) > 0 and not row.get("reutilizado_de_lote")
               for row in inc._read(path)["detalles"])


def run(root, days, args, *, clock=time.monotonic):
    queue = ordered_days(days, args.orden)
    start = clock()
    control = inc.batch._RequestControl()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid4().hex[:8]
    report = {
        "schema_version": 1, "ejecucion_id": run_id, "inicio_utc": inc.batch._utc(),
        "estado": "en_curso", "operaciones": [], "detalles_consultados": 0,
        "consultas_api": 0, "configuracion": vars(args).copy(),
        "coordinador_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "motor_sha256": hashlib.sha256(Path(inc.__file__).read_bytes()).hexdigest(),
    }
    report_path = root / "data/processed/incremental/ejecuciones" / f"{run_id}.json"
    with inc.single_writer(root):
        initial_pending = pending_state(root)
        if args.reintentar_publicacion and (not initial_pending or initial_pending["fase"] != "publicacion_en_curso"):
            raise ValueError("No hay una publicación sin confirmar para reintentar.")
        inc._atomic_json(report_path, report)
        try:
            for index in range(args.max_segmentos):
                if clock() - start >= args.max_segundos:
                    report["motivo_fin"] = "tiempo_para_iniciar_segmentos_agotado"
                    break
                remaining = args.max_detalles - report["detalles_consultados"]
                segment_limit = max(1, min(args.limite, remaining))
                _, task, pending = next_task(root, queue, segment_limit)
                if task is None:
                    if not report["operaciones"] and args.publicar_bigquery:
                        # También permite publicar trabajo completado localmente antes.
                        options = SimpleNamespace(**vars(args))
                        options.reintentar_publicacion = False
                        inc.execute(root, queue, options, control=control, locked=True)
                        report["operaciones"].append({"tipo": "publicar_historial_existente", "consultas_api": 0,
                                                      "detalles_consultados": 0})
                    report["motivo_fin"] = "sin_pendientes_en_listados_guardados"
                    break
                if pending and pending["fase"] == "publicacion_en_curso":
                    if (not args.reintentar_publicacion or not args.publicar_bigquery
                            or pending["destino"] != inc._target(args)):
                        raise RuntimeError("Publicación anterior sin confirmar. Revisa el job en BigQuery antes de reintentar.")
                if task["tipo"] == "recuperar_captura" and task["detalles"] > args.limite:
                    raise ValueError(f"La recuperación necesita --limite de al menos {task['detalles']}.")
                if task["detalles"] > remaining:
                    report["motivo_fin"] = "presupuesto_detalles"
                    break
                # Reserva el peor caso de un reintento 429 por consulta lógica.
                worst_case = 2 * (task["detalles"] + task["listados"])
                if control.attempts + worst_case > args.max_consultas:
                    report["motivo_fin"] = "presupuesto_consultas_api"
                    break
                if clock() - start >= args.max_segundos:
                    report["motivo_fin"] = "tiempo_para_iniciar_segmentos_agotado"
                    break
                options = SimpleNamespace(**vars(args))
                options.limite = segment_limit
                options.reintentar_publicacion = bool(pending and pending["fase"] == "publicacion_en_curso"
                                                      and args.reintentar_publicacion)
                print(f"\nSegmento {index + 1}/{args.max_segmentos}: {task['fecha']} | {task['tipo']}")
                before = control.attempts
                try:
                    inc.execute(root, queue, options, control=control, locked=True)
                except BaseException:
                    report["detalles_consultados"] += new_details(root, control.attempts - before)
                    raise
                count = new_details(root, control.attempts - before)
                report["detalles_consultados"] += count
                report["consultas_api"] = control.attempts
                state = inc._read(root / inc.STATE_REL)
                publication = state.get("publicacion", {})
                report["operaciones"].append({"tipo": task["tipo"], "fecha": state.get("fecha"),
                                              "lote": state.get("lote"), "detalles_consultados": count,
                                              "consultas_api": control.attempts - before,
                                              "publicado_bigquery": state.get("publicado_bigquery", False),
                                              "job_id": publication.get("job_id")})
                inc._atomic_json(report_path, report)
                if options.reintentar_publicacion:
                    # Un reintento explícito sólo resuelve la publicación pendiente.
                    report["motivo_fin"] = "publicacion_reanudada"
                    break
                queue = rotate_days(queue, state.get("fecha"))
            else:
                report["motivo_fin"] = "maximo_segmentos"
            report["estado"] = "completo"
        except BaseException as error:
            report["estado"] = "interrumpido" if isinstance(error, KeyboardInterrupt) else "fallido"
            report["tipo_error"] = type(error).__name__
            report["motivo_fin"] = "operacion_pendiente_revisar_diario"
            raise
        finally:
            report.update(consultas_api=control.attempts, duracion_segundos=round(clock() - start, 3),
                          fin_utc=inc.batch._utc())
            inc._atomic_json(report_path, report)
            print("\nResumen de corrida:", json.dumps(report, ensure_ascii=False, indent=2))
            print("Registro:", report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde", required=True, help="AAAA-MM-DD, inclusive.")
    parser.add_argument("--hasta", required=True, help="AAAA-MM-DD, inclusive; antes de hoy.")
    parser.add_argument("--limite", type=int, default=10, help="Máximo de detalles por segmento: 1–10.")
    parser.add_argument("--max-detalles", type=int, default=30, help="Presupuesto de códigos consultados: 1–100.")
    parser.add_argument("--max-segmentos", type=int, default=3, help="Hasta 20 segmentos por corrida.")
    parser.add_argument("--max-consultas", type=int, default=None, help="Máximo de solicitudes HTTP, incluidos reintentos.")
    parser.add_argument("--max-segundos", type=int, default=600, help="Tiempo para iniciar segmentos; no interrumpe uno activo.")
    parser.add_argument("--orden", choices=["recientes", "antiguas"], default="recientes")
    parser.add_argument("--ejecutar", action="store_true")
    parser.add_argument("--publicar-bigquery", action="store_true")
    parser.add_argument("--reintentar-publicacion", action="store_true")
    parser.add_argument("--project", default="inteligencia-mercado-publico")
    parser.add_argument("--dataset", default="mercado_publico")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--max-bytes", type=int, default=1_073_741_824)
    args = parser.parse_args(argv)
    if getattr(inc, "MULTI_SEGMENT_VERSION", None) != 1:
        raise RuntimeError("Copia también la versión actualizada de src/run_incremental.py incluida en la entrega.")
    try:
        days = inc.date_range(args.desde, args.hasta)
        inc.batch.plan_lote(days[0].strftime("%d%m%Y"), args.limite)
        if not 1 <= args.max_detalles <= 100 or not 1 <= args.max_segmentos <= 20:
            raise ValueError("Usa --max-detalles entre 1 y 100 y --max-segmentos entre 1 y 20.")
        if not 1 <= args.max_segundos <= 3600 or args.max_bytes <= 0:
            raise ValueError("Usa --max-segundos entre 1 y 3600 y --max-bytes positivo.")
        if args.max_consultas is None:
            args.max_consultas = 2 * (args.max_detalles + args.max_segmentos)
        if not 1 <= args.max_consultas <= 1000:
            raise ValueError("--max-consultas debe estar entre 1 y 1000.")
        if args.reintentar_publicacion and not (args.ejecutar and args.publicar_bigquery):
            raise ValueError("--reintentar-publicacion requiere --ejecutar y --publicar-bigquery.")
    except ValueError as error:
        parser.error(str(error))
    root = ROOT.resolve()
    queue = ordered_days(days, args.orden)
    coverage, task, pending = next_task(root, queue, min(args.limite, args.max_detalles))
    print(json.dumps({"cobertura": coverage, "orden_inicial": [day.isoformat() for day in queue],
                      "primer_trabajo": task, "operacion_pendiente": pending,
                      "presupuestos": {"detalles_por_segmento": args.limite, "detalles_totales": args.max_detalles,
                                       "segmentos": args.max_segmentos, "solicitudes_api": args.max_consultas,
                                       "segundos_para_iniciar_segmentos": args.max_segundos},
                      "nota": "Una fecha por segmento; se rota tras terminar. Fechas sin listado requieren consulta. "
                              "El plan posterior depende de esas respuestas. Cada publicación se confirma antes de seguir."},
                     ensure_ascii=False, indent=2))
    if not args.ejecutar:
        print("Sólo plan: sin API, BigQuery, inferencia ni escritura de archivos.")
        return 0
    run(root, days, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
