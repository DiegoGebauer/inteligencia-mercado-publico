"""Una corrida acotada desde un workspace recuperado; produce un checkpoint local.

Sin --ejecutar sólo lee y muestra el plan del ejecutor. Con --ejecutar consulta
Mercado Público, predice en CPU y publica BigQuery. No escribe Cloud Storage.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from src.experiments import export_model_bundle as bundle
from src.inference import portable_model as portable
from src.operations import state_checkpoint as checkpoint
from src import run_incremental as incremental


ROOT = Path(__file__).resolve().parents[2]
TICKET = "MERCADO_PUBLICO_TICKET"
PROJECT = "inteligencia-mercado-publico"


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def read_ticket_file(path):
    from dotenv import dotenv_values
    return dotenv_values(path).get(TICKET)


def child_environment(root, model, *, need_ticket):
    environment = os.environ.copy()
    environment[portable.ENVIRONMENT_VARIABLE] = str(model)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if need_ticket and not environment.get(TICKET, "").strip():
        ticket = read_ticket_file(Path(root) / ".env")
        if not isinstance(ticket, str) or not ticket.strip():
            raise RuntimeError("Falta MERCADO_PUBLICO_TICKET en el entorno o .env del proyecto principal.")
        environment[TICKET] = ticket
    return environment


def build_plan(root=ROOT, bootstrap_dir=None, *, desde, hasta, limite=5):
    root = Path(root).resolve()
    days = incremental.date_range(desde, hasta)
    if type(limite) is not int or not 1 <= limite <= 5:
        raise ValueError("Esta prueba admite de 1 a 5 detalles y un único segmento.")
    directory = Path(bootstrap_dir) if bootstrap_dir is not None else Path("data/processed/arranque_cloud_prueba_01")
    directory = directory.resolve() if directory.is_absolute() else (root / directory).resolve()
    report = checkpoint.read_json((directory / "bootstrap.json").read_bytes())
    if report.get("estado") != "preparado" or report.get("fase") != "listo":
        raise ValueError("El arranque no está preparado; revisa bootstrap.json.")
    workspace = Path(report["workspace"]).resolve()
    model = Path(report["modelo_local"]).resolve()
    if workspace != directory / "workspace" or model != directory / "descargas/modelo.tar.gz" or workspace == root:
        raise ValueError("Las rutas del arranque no corresponden a su carpeta separada.")
    for name, expected in report["codigo_y_config_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (workspace / relative).resolve().is_relative_to(workspace):
            raise ValueError("Hay una ruta de código inválida en bootstrap.json.")
        if bundle.sha_file(workspace / relative) != expected:
            raise ValueError(f"Cambió el código o configuración recuperada: {name}")
    model_plan = portable.inspect_bundle(model, workspace)
    if model_plan["descriptor"]["bundle"] != report["referencias"]["modelo"]:
        raise ValueError("El modelo del workspace difiere del arranque confirmado.")
    before = checkpoint.build_plan(workspace)
    # El constructor del proceso padre y del hijo debe conservar los mismos bytes.
    if bundle.sha_file(Path(checkpoint.history.__file__)) != before["manifest"]["code_sha256"]["src/reporting/build_batch_history.py"]:
        raise ValueError("El constructor de historial del proyecto principal difiere del recuperado.")
    state = checkpoint.read_json((workspace / checkpoint.STATE).read_bytes())
    if state.get("fase") != "completo":
        raise ValueError("Hay una operación incremental pendiente; revisa su diario antes de continuar.")
    failed = []
    for path in (directory / "corridas").glob("*.json"):
        if checkpoint.read_json(path.read_bytes()).get("estado") != "completo":
            failed.append(path.name)
    if failed:
        raise RuntimeError("Hay una corrida sin finalizar en corridas/. Conserva y revisa su registro antes de repetir.")
    command = [sys.executable, "-B", "-m", "src.run_incremental_batch", "--desde", days[0].isoformat(),
               "--hasta", days[-1].isoformat(), "--limite", str(limite), "--max-detalles", str(limite),
               "--max-segmentos", "1", "--max-consultas", str(2 * (limite + 1)),
               "--max-segundos", "300", "--orden", "recientes", "--project", PROJECT,
               "--dataset", "mercado_publico", "--location", "us-central1", "--max-bytes", "1073741824"]
    return {"root": root, "directory": directory, "workspace": workspace, "model": model,
            "command": command, "before": before["manifest"], "state_sha256": bundle.sha_file(workspace / checkpoint.STATE),
            "desde": desde, "hasta": hasta, "limite": limite}


@contextmanager
def single_cycle(directory):
    path = directory / "corrida_recuperada.lock"
    token = checkpoint.canonical({"pid": os.getpid(), "token": uuid4().hex})
    with path.open("xb") as stream:
        stream.write(token)
    try:
        yield
    finally:
        if path.is_file() and path.read_bytes() == token:
            path.unlink()


def show_plan(plan, *, runner=subprocess.run):
    print(json.dumps({"workspace": str(plan["workspace"]), "modelo_local": str(plan["model"]),
                      "conteos_actuales": plan["before"]["conteos"], "maximo_detalles": plan["limite"],
                      "maximo_segmentos": 1, "modo_modelo": "portable_local"}, ensure_ascii=False, indent=2))
    result = runner(plan["command"], cwd=plan["workspace"], env=child_environment(plan["root"], plan["model"], need_ticket=False))
    if result.returncode != 0:
        raise RuntimeError("El plan del ejecutor recuperado no terminó correctamente.")
    print("Sólo plan: no se cargó el ticket, no hubo captura, inferencia o publicación.")


def execute_plan(plan, *, runner=subprocess.run):
    # Buscar el ticket antes de marcar una operación iniciada; nunca se escribe en el reporte.
    environment = child_environment(plan["root"], plan["model"], need_ticket=True)
    with single_cycle(plan["directory"]):
        current = build_plan(plan["root"], plan["directory"], desde=plan["desde"], hasta=plan["hasta"], limite=plan["limite"])
        if current["before"]["checkpoint_id"] != plan["before"]["checkpoint_id"]:
            raise RuntimeError("Cambió el estado desde la preparación del plan.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid4().hex[:8]
        path = plan["directory"] / "corridas" / f"{run_id}.json"
        report = {"schema_version": 1, "estado": "en_curso", "fase": "pipeline", "inicio_utc": _utc(),
                  "desde": plan["desde"], "hasta": plan["hasta"], "limite": plan["limite"],
                  "workspace": str(plan["workspace"]), "checkpoint_entrada": plan["before"]["checkpoint_id"],
                  "estado_entrada_sha256": plan["state_sha256"], "checkpoint_publicado_gcs": False,
                  "lanzador_sha256": bundle.sha_file(Path(__file__))}

        def save():
            incremental._atomic_json(path, report)

        save()
        try:
            command = [*plan["command"], "--ejecutar", "--publicar-bigquery"]
            result = runner(command, cwd=plan["workspace"], env=environment)
            if result.returncode != 0:
                raise RuntimeError("La corrida no terminó correctamente. Revisa el diario del workspace y cualquier job BigQuery antes de repetir.")
            state_path = plan["workspace"] / checkpoint.STATE
            state = checkpoint.read_json(state_path.read_bytes())
            if (state.get("fase") != "completo" or state.get("publicado_bigquery") is not True
                    or not state.get("publicacion", {}).get("job_id")
                    or bundle.sha_file(state_path) == plan["state_sha256"]):
                raise ValueError("No se confirmó una finalización nueva con publicación BigQuery.")
            report.update(fase="checkpoint", job_id=state["publicacion"]["job_id"])
            save()
            archive = checkpoint.create_checkpoint(plan["workspace"])
            manifest = checkpoint.verify_archive(archive)
            report.update(estado="completo", fase="checkpoint_local_verificado", checkpoint_local=str(archive),
                          checkpoint_id=manifest["checkpoint_id"], sha256=bundle.sha_file(archive),
                          bytes=archive.stat().st_size, archivos=len(manifest["files"]), conteos=manifest["conteos"],
                          nota="Publicación BigQuery confirmada. Checkpoint nuevo guardado localmente; falta subirlo a GCS y actualizar su referencia.")
            return report
        except BaseException as error:
            report.update(estado="interrumpido" if isinstance(error, KeyboardInterrupt) else "fallido", tipo_error=type(error).__name__)
            raise
        finally:
            report["fin_utc"] = _utc()
            save()
            print("Registro de la corrida:", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arranque", type=Path, default=None)
    parser.add_argument("--desde", required=True)
    parser.add_argument("--hasta", required=True)
    parser.add_argument("--limite", type=int, default=5)
    parser.add_argument("--ejecutar", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(args.root, args.arranque, desde=args.desde, hasta=args.hasta, limite=args.limite)
    if not args.ejecutar:
        show_plan(plan)
        return 0
    result = execute_plan(plan)
    print("Corrida recuperada finalizada; checkpoint local listo para subir:")
    print(json.dumps({name: result[name] for name in ("checkpoint_local", "checkpoint_id", "sha256", "bytes", "archivos", "conteos", "job_id", "checkpoint_publicado_gcs")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
