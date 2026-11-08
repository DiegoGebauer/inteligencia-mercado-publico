"""Captura un lote pequeño y aplica el candidato existente, sin entrenarlo.

Importar este módulo no consulta la API, no carga modelos y no escribe archivos.
Las dos etapas se llaman explícitamente desde el notebook 08.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "mercado_publico_afinidad_embeddings"
CODE_FILES = [
    "src/api/mercado_publico_client.py",
    "src/ingestion/capture_store.py",
    "src/ingestion/recent_batch.py",
    "src/processing/normalize_tenders.py",
    "src/nlp/build_tender_texts.py",
    "src/nlp/rank_tenders.py",
    "src/nlp/train_affinity.py",
    "src/nlp/predict_affinity_pilot.py",
    "src/run_incremental.py",
]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _code(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", value):
        raise ValueError("La respuesta contiene un código de licitación inválido.")
    return value


def _records(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("Listado"), list):
        raise ValueError("La respuesta no contiene un Listado válido.")
    if any(not isinstance(row, dict) for row in payload["Listado"]):
        raise ValueError("Listado contiene registros inválidos.")
    return payload["Listado"]


def _error_info(error):
    """Extrae sólo categorías conocidas del cliente; nunca vuelca URLs ni tickets."""
    info = {"tipo_error": type(error).__name__}
    message = str(error)
    status = re.fullmatch(r"Error al consultar Mercado Público\. Estado: ([1-5][0-9]{2}|sin respuesta)", message)
    if status:
        value = status.group(1)
        info["motivo"] = "http" if value.isdigit() else "conexion_o_timeout"
        if value.isdigit():
            info["estado_http"] = int(value)
    elif message == "La API respondió, pero el contenido no es un JSON válido.":
        info["motivo"] = "respuesta_no_json"
    elif message == "No se encontró MERCADO_PUBLICO_TICKET en el archivo .env":
        info["motivo"] = "ticket_no_configurado"
    else:
        info["motivo"] = "validacion" if isinstance(error, ValueError) else "error_sin_detalle_seguro"
    return info


class _RequestControl:
    """Cadencia por lote; un reintento 429 y corte si persiste la limitación.

    Las pausas elegidas no representan una cuota oficial de Mercado Público.
    Un Retry-After mayor de 60 s detiene el lote en vez de reintentar antes.
    """
    def __init__(self, *, clock=time.monotonic, wait=time.sleep):
        self.clock, self.wait = clock, wait
        self.interval = 3
        self.fallback_wait = 30
        self.max_wait = 60
        self.next_allowed = 0
        self.blocked = False
        self.attempts = 0
        self.events = []
        self.retry_after_utc = None

    def request(self, fetch, parameter):
        if self.blocked:
            raise RuntimeError("Error al consultar Mercado Público. Estado: 429")
        for attempt in range(2):
            delay = max(0, self.next_allowed - self.clock())
            if delay:
                self.wait(delay)
            self.attempts += 1
            event = {"parametro": parameter, "intento": attempt + 1, "espera_previa_segundos": delay}
            try:
                result = fetch(parameter)
            except Exception as error:
                info = _error_info(error)
                event.update(info)
                self.events.append(event)
                self.next_allowed = self.clock() + self.interval
                if info.get("estado_http") != 429:
                    raise
                server_wait = getattr(error, "retry_after_seconds", None)
                wait_seconds = max(self.interval, server_wait if server_wait is not None else self.fallback_wait)
                event["espera_requerida_segundos"] = wait_seconds
                if attempt == 1 or wait_seconds > self.max_wait:
                    self.blocked = True
                    try:
                        self.retry_after_utc = (datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)).isoformat()
                    except OverflowError:
                        self.retry_after_utc = None
                    raise
                self.next_allowed = self.clock() + wait_seconds
            else:
                self.next_allowed = self.clock() + self.interval
                event["estado"] = "correcto"
                self.events.append(event)
                return result

    def summary(self):
        return {"intervalo_segundos": self.interval, "maximo_reintentos_429": 1,
                "espera_429_sin_encabezado_segundos": self.fallback_wait,
                "maxima_espera_automatica_segundos": self.max_wait,
                "bloqueado_por_429": self.blocked,
                "reintentar_despues_utc": self.retry_after_utc,
                "eventos": self.events}


def _capture_detail(code, folder, fetch):
    from src.ingestion.capture_store import guardar_captura

    entry = {"codigo": code, "estado": "fallido"}
    try:
        detail = fetch(code)
        entry["captura_utc"] = _utc()
        path = folder / "details" / f"{code}.json"
        guardar_captura(detail, path)
        entry.update(archivo=path.relative_to(folder).as_posix(), sha256=_hash(path))
        records = _records(detail)
        if len(records) != 1 or records[0].get("CodigoExterno") != code:
            raise ValueError("El detalle no corresponde al código solicitado.")
        items = records[0].get("Items")
        if not isinstance(items, dict) or not isinstance(items.get("Listado"), list):
            raise ValueError("El detalle no contiene Items.Listado.")
        entry["estado"] = "correcto"
    except Exception as error:
        entry.update(_error_info(error))
    return entry


def plan_lote(fecha: str, limite: int = 5) -> dict:
    """Describe el trabajo sin consultas, modelos ni escritura. Fecha: ddmmaaaa."""
    if not isinstance(fecha, str) or len(fecha) != 8 or not fecha.isdigit():
        raise ValueError("La fecha debe tener formato ddmmaaaa.")
    parsed = datetime.strptime(fecha, "%d%m%Y")
    if isinstance(limite, bool) or not isinstance(limite, int) or not 1 <= limite <= 10:
        raise ValueError("El límite debe ser un entero entre 1 y 10.")
    return {
        "fecha_consulta": fecha,
        "fecha_iso": parsed.date().isoformat(),
        "limite": limite,
        "maximo_consultas_logicas": 1 + limite,
        "maximo_consultas_api": 2 * (1 + limite),
        "control_consultas": "intervalo 3 s; un reintento 429; espera Retry-After o 30 s; corte si persiste o espera >60 s",
        "seleccion": "primeros códigos únicos en orden alfabético; sin filtro temático",
        "modelo": f"{MODEL_NAME}@candidato, contrastado con el comprobante local",
        "salidas": "JSON del lote, predicciones.csv y registro_ejecucion.json",
    }


def capturar_lote(fecha: str, limite: int = 5, *, raiz: Path = ROOT) -> Path:
    """Consulta una fecha y hasta diez detalles; conserva respuestas por lote nuevo.

    No reintenta solicitudes automáticamente. Si falla un detalle, conserva los
    demás y registra el fallo. La predicción exige una captura completa.
    """
    from src.api.mercado_publico_client import get_licitaciones_por_fecha, get_licitacion_por_codigo
    from src.ingestion.capture_store import guardar_captura

    plan = plan_lote(fecha, limite)
    control = _RequestControl()
    raiz = Path(raiz).resolve()
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    folder = raiz / "data" / "raw" / "batches" / batch_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "lote_id": batch_id, "plan": plan,
        "inicio_utc": _utc(), "estado": "fallido", "seleccionados": [],
        "detalles": [], "consultas_api": 0,
        "nota": "Captura operativa pequeña; no es una muestra de evaluación ni estima el universo.",
    }
    try:
        payload = control.request(get_licitaciones_por_fecha, fecha)
        listed_at = _utc()
        guardar_captura(payload, folder / "listado.json")
        manifest["listado"] = {"archivo": "listado.json", "sha256": _hash(folder / "listado.json"), "captura_utc": listed_at}
        rows = _records(payload)
        codes = sorted({_code(row.get("CodigoExterno")) for row in rows})
        manifest["filas_listado"] = len(rows)
        manifest["codigos_unicos_listado"] = len(codes)
        manifest["seleccionados"] = codes[:limite]
        for code in manifest["seleccionados"]:
            attempts_before = control.attempts
            entry = _capture_detail(code, folder, lambda value: control.request(get_licitacion_por_codigo, value))
            entry["intentos_http"] = control.attempts - attempts_before
            if not entry["intentos_http"]:
                entry["omitido_por_espera"] = True
            manifest["detalles"].append(entry)
        manifest["estado"] = (
            "vacio" if not codes else
            "completo" if all(row["estado"] == "correcto" for row in manifest["detalles"]) else
            "incompleto"
        )
    except Exception as error:
        manifest.update(_error_info(error))
    finally:
        manifest["consultas_api"] = control.attempts
        manifest["control_api"] = control.summary()
        manifest["fin_utc"] = _utc()
        guardar_captura(manifest, folder / "captura.json")
    print("Lote:", folder)
    print("Estado:", manifest["estado"], "| Consultas:", manifest["consultas_api"])
    print("Detalles correctos:", sum(row["estado"] == "correcto" for row in manifest["detalles"]))
    if manifest["estado"] not in {"completo", "vacio"}:
        raise RuntimeError(f"La captura no se completó. Revisa el registro local: {folder / 'captura.json'}") from None
    return folder


def _load_capture(folder, *, permitir_incompleto=False):
    """Verifica identidad, integridad y conjunto completo antes de procesar."""
    manifest = _read(folder / "captura.json")
    states = {"completo", "incompleto"} if permitir_incompleto else {"completo"}
    if manifest["lote_id"] != folder.name or manifest["estado"] not in states:
        raise ValueError("Se requiere un lote completo y no vacío.")
    selected = manifest["seleccionados"]
    if not 1 <= len(selected) <= 10 or len(set(selected)) != len(selected):
        raise ValueError("La selección del lote es inválida.")
    if [row["codigo"] for row in manifest["detalles"]] != selected:
        raise ValueError("Los detalles no coinciden con la selección registrada.")
    listing = manifest["listado"]
    if listing["archivo"] != "listado.json" or _hash(folder / "listado.json") != listing["sha256"]:
        raise ValueError("La captura del listado cambió.")
    plan = plan_lote(manifest["plan"]["fecha_consulta"], manifest["plan"]["limite"])
    codes = sorted({_code(row.get("CodigoExterno")) for row in _records(_read(folder / "listado.json"))})
    incremental = manifest.get("seleccion_incremental")
    offset = 0
    if incremental is not None:
        if not isinstance(incremental, dict) or incremental.get("version") != 1:
            raise ValueError("Versión de selección incremental inválida.")
        offset = incremental.get("desplazamiento")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Desplazamiento incremental inválido.")
    if selected != codes[offset:offset + plan["limite"]]:
        raise ValueError("La selección no corresponde al listado y límite registrados.")
    records, sources = [], {}
    for entry in manifest["detalles"]:
        code = _code(entry["codigo"])
        if permitir_incompleto and entry["estado"] == "fallido":
            continue
        expected = f"details/{code}.json"
        if entry["estado"] != "correcto" or entry["archivo"] != expected:
            raise ValueError("Hay un detalle incompleto o una ruta inesperada.")
        path = folder / expected
        if path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError("El detalle sale del directorio del lote.")
        if _hash(path) != entry["sha256"]:
            raise ValueError(f"La captura cambió: {code}")
        detail = _records(_read(path))
        if len(detail) != 1 or detail[0].get("CodigoExterno") != code:
            raise ValueError("La identidad del detalle es incorrecta.")
        records.extend(detail)
        sources[code] = entry
    return manifest, records, sources


def recuperar_lote(carpeta_lote: Path, *, raiz: Path = ROOT) -> Path:
    """Crea un lote derivado: copia detalles válidos y consulta sólo los fallidos.

    No modifica capturas anteriores, ni repite el listado. Hace un único intento
    por detalle faltante y conserva su diagnóstico seguro si vuelve a fallar.
    """
    from src.api.mercado_publico_client import get_licitacion_por_codigo
    from src.ingestion.capture_store import guardar_captura

    root, source_folder = Path(raiz).resolve(), Path(carpeta_lote).resolve()
    control = _RequestControl()
    if source_folder.parent != (root / "data/raw/batches").resolve():
        raise ValueError("El lote debe estar directamente en data/raw/batches.")
    original_hash = _hash(source_folder / "captura.json")
    original, _, valid_sources = _load_capture(source_folder, permitir_incompleto=True)
    if original["estado"] != "incompleto":
        raise ValueError("Sólo se recuperan capturas incompletas; un lote completo continúa desde predicción.")
    previous_control = original.get("control_api", {})
    if previous_control.get("bloqueado_por_429"):
        deadline = previous_control.get("reintentar_despues_utc")
        if not deadline:
            raise RuntimeError("El servidor indicó una espera fuera de rango; revisa el registro antes de retomar.")
        if datetime.now(timezone.utc) < datetime.fromisoformat(deadline):
            raise RuntimeError(f"El lote está en espera por 429. Retoma después de {deadline}.")
    missing = [entry["codigo"] for entry in original["detalles"] if entry["estado"] == "fallido"]
    if not missing:
        raise ValueError("No hay detalles fallidos para recuperar.")
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    folder = source_folder.parent / batch_id
    folder.mkdir(exist_ok=False)
    manifest = {
        "schema_version": 1, "lote_id": batch_id, "plan": original["plan"],
        "inicio_utc": _utc(), "estado": "fallido", "seleccionados": original["seleccionados"],
        "detalles": [], "consultas_api": 0, "detalles_reutilizados": len(valid_sources),
        "origen": {"lote_id": source_folder.name, "captura_sha256": original_hash},
        "nota": "Lote derivado. Conserva fechas originales de los detalles reutilizados; los recuperados tienen nueva fecha de captura.",
    }
    if "seleccion_incremental" in original:
        manifest["seleccion_incremental"] = dict(original["seleccion_incremental"])

    def copy_capture(relative, expected_hash):
        source = source_folder / relative
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Una captura de origen cambió antes de copiarla.")
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)

    try:
        copy_capture("listado.json", original["listado"]["sha256"])
        manifest["listado"] = dict(original["listado"])
        for key in ["filas_listado", "codigos_unicos_listado"]:
            manifest[key] = original[key]
        for code in original["seleccionados"]:
            if code in valid_sources:
                entry = dict(valid_sources[code])
                copy_capture(entry["archivo"], entry["sha256"])
                entry["reutilizado_de_lote"] = source_folder.name
            else:
                attempts_before = control.attempts
                entry = _capture_detail(code, folder, lambda value: control.request(get_licitacion_por_codigo, value))
                entry["intentos_http"] = control.attempts - attempts_before
                if not entry["intentos_http"]:
                    entry["omitido_por_espera"] = True
            manifest["detalles"].append(entry)
        if _hash(source_folder / "captura.json") != original_hash:
            raise ValueError("El registro de origen cambió durante la recuperación.")
        manifest["estado"] = "completo" if all(entry["estado"] == "correcto" for entry in manifest["detalles"]) else "incompleto"
    except Exception as error:
        manifest.update(_error_info(error))
    finally:
        manifest["consultas_api"] = control.attempts
        manifest["control_api"] = control.summary()
        manifest["fin_utc"] = _utc()
        guardar_captura(manifest, folder / "captura.json")
    print("Lote derivado:", folder)
    print("Detalles reutilizados:", manifest["detalles_reutilizados"], "| Consultas nuevas:", manifest["consultas_api"])
    for entry in manifest["detalles"]:
        if entry["estado"] != "correcto":
            print("Falló:", entry["codigo"], "| Motivo:", entry.get("motivo"), "| Estado HTTP:", entry.get("estado_http", "no disponible"))
    print("Estado:", manifest["estado"])
    if manifest["estado"] != "completo":
        raise RuntimeError(f"La recuperación no se completó. Revisa: {folder / 'captura.json'}") from None
    return folder


def _prepare(records):
    import pandas as pd
    from src.processing.normalize_tenders import (
        build_tender_rows, build_tender_item_rows, build_tenders_dataframe,
        build_tender_items_dataframe, validate_normalized_data,
    )
    from src.nlp.build_tender_texts import build_tender_texts, join_texts

    tenders = build_tenders_dataframe(build_tender_rows(records))
    tenders["CodigoExterno"] = tenders["CodigoExterno"].astype("string")
    items = build_tender_items_dataframe(build_tender_item_rows(records))
    if items.empty:
        items = pd.DataFrame(columns=["CodigoExterno", "ItemPosicion", "NombreProducto", "Descripcion"])
    items["CodigoExterno"] = items["CodigoExterno"].astype("string")
    validate_normalized_data(tenders, items)
    text_fields = tenders.reindex(columns=["CodigoExterno", "Nombre", "Descripcion"])
    if items.empty:
        # La misma unión histórica, sin inventar ítems cuando no existen.
        texts = text_fields[["CodigoExterno"]].copy()
        texts["TextoLicitacion"] = text_fields.apply(lambda row: join_texts([row["Nombre"], row["Descripcion"]]), axis=1)
    else:
        texts = build_tender_texts(text_fields, items.reindex(columns=["CodigoExterno", "NombreProducto", "Descripcion"]))
    if texts["TextoLicitacion"].str.strip().eq("").any():
        raise ValueError("Hay licitaciones sin texto para representar.")
    columns = [
        "CodigoExterno", "Nombre", "Comprador_CodigoOrganismo", "Comprador_NombreOrganismo",
        "Comprador_RegionUnidad", "Fechas_FechaPublicacion", "Fechas_FechaCierre", "CodigoEstado", "Estado",
    ]
    return tenders.reindex(columns=columns).merge(texts, on="CodigoExterno", validate="one_to_one").rename(columns={
        "Fechas_FechaPublicacion": "FechaPublicacion", "Fechas_FechaCierre": "FechaCierre", "Estado": "EstadoCapturado",
    })


def _load_candidate(root):
    """Resuelve y carga la versión fija desde una copia temporal del catálogo."""
    import sqlite3
    import tempfile
    import joblib
    import mlflow
    import mlflow.sklearn
    import numpy as np
    from mlflow.models import Model
    from mlflow.tracking import MlflowClient

    receipt = _read(root / "data/processed/affinity_mlflow_baseline_registration.json")
    original = root / "models/affinity_aligned_299.joblib"
    if receipt["registered_model"] != MODEL_NAME or str(receipt["model_version"]) != "1":
        raise ValueError("El comprobante ya no describe el candidato acordado.")
    if _hash(original) != receipt["model_sha256"]:
        raise ValueError("El modelo original cambió respecto del comprobante.")
    saved = joblib.load(original)
    tracking_before, registry_before = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    with tempfile.TemporaryDirectory(prefix="proyecto2-lote-modelo-") as temporary:
        temp = Path(temporary).resolve()
        if temp.is_relative_to(root):
            raise RuntimeError("La copia del catálogo debe quedar fuera del proyecto.")
        uri = "sqlite:///" + (temp / "catalogo.db").as_posix()
        try:
            source = sqlite3.connect((root / "mlflow_local/mlflow.db").as_uri() + "?mode=ro", uri=True)
            try:
                target = sqlite3.connect(str(temp / "catalogo.db"))
                try:
                    source.backup(target)
                finally:
                    target.close()
            finally:
                source.close()
            mlflow.set_tracking_uri(uri)
            mlflow.set_registry_uri(uri)
            client = MlflowClient()
            candidate = client.get_model_version_by_alias(MODEL_NAME, "candidato")
            if (str(candidate.version) != str(receipt["model_version"])
                    or candidate.run_id != receipt["run_id"] or candidate.status != "READY"
                    or client.get_run(candidate.run_id).info.status != "FINISHED"):
                raise ValueError("El alias candidato ya no coincide con la versión y run acordados.")
            model_uri = f"models:/{MODEL_NAME}/{candidate.version}"
            destination = temp / "modelo"
            destination.mkdir()
            classifier = mlflow.sklearn.load_model(model_uri, dst_path=str(destination))
            descriptions = list(destination.rglob("MLmodel"))
            if len(descriptions) != 1:
                raise ValueError("No hay una única descripción MLmodel.")
            description = Model.load(str(descriptions[0]))
            spec = description.signature.inputs.inputs[0]
            if (tuple(spec.shape) != (-1, 384) or str(spec.type) != "float32"
                    or description.metadata["original_joblib_sha256"] != receipt["model_sha256"]
                    or classifier.n_features_in_ != 384
                    or [type(step).__name__ for _, step in classifier.steps] != ["StandardScaler", "LogisticRegression"]):
                raise ValueError("El contrato o la huella MLflow no coincide con el candidato.")
            np.testing.assert_array_equal(classifier.classes_, [0, 1, 2, 3])
            if classifier.get_params()["logisticregression__C"] != 0.1:
                raise ValueError("El candidato no tiene C=0.1.")
            for step, attributes in [(0, ["mean_", "scale_"]), (1, ["coef_", "intercept_", "classes_"])]:
                for attribute in attributes:
                    np.testing.assert_array_equal(getattr(classifier.steps[step][1], attribute), getattr(saved["classifier"].steps[step][1], attribute))
        finally:
            mlflow.set_tracking_uri(tracking_before)
            mlflow.set_registry_uri(registry_before)
            # Cerrar únicamente los motores de la copia, también si falla la carga.
            from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
            from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
            for store in [TrackingStore, RegistryStore]:
                engine = store._engine_map.pop(uri, None)
                if engine is not None:
                    engine.dispose()
    return classifier, saved, receipt


def _originals(root):
    paths = [root / name for name in CODE_FILES] + [
        root / "models/affinity_aligned_299.joblib", root / "mlflow_local/mlflow.db",
        root / "data/processed/affinity_mlflow_baseline_registration.json", root / "config/company_profile.yaml",
    ]
    paths += sorted(path for path in (root / "mlflow_local/artifacts").rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): _hash(path) for path in paths}


def predecir_lote(carpeta_lote: Path, *, raiz: Path = ROOT) -> Path:
    """Lee capturas existentes y genera un CSV y registro nuevos, sin API."""
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    from importlib.metadata import version
    from src.ingestion.capture_store import guardar_captura

    root, folder = Path(raiz).resolve(), Path(carpeta_lote).resolve()
    if folder.parent != (root / "data/raw/batches").resolve():
        raise ValueError("El lote debe estar directamente en data/raw/batches.")
    for package, expected in {"mlflow": "3.16.1", "scikit-learn": "1.9.1"}.items():
        if version(package) != expected:
            raise RuntimeError(f"La versión de {package} difiere de la utilizada en el notebook 05.")
    capture, records, sources = _load_capture(folder)
    before = _originals(root)
    output_base = root / "data/processed/batches" / folder.name
    if output_base.exists():
        for previous in output_base.glob("*/registro_ejecucion.json"):
            if _read(previous).get("estado") == "completo":
                raise FileExistsError(f"El lote ya tiene una predicción completa: {previous.parent}")
    # Un intento fallido se conserva y permite repetir sólo la etapa local.
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    output = output_base / attempt
    output.mkdir(parents=True, exist_ok=False)
    log = {"schema_version": 1, "lote_id": folder.name, "inicio_utc": _utc(), "estado": "fallido",
           "hash_captura_sha256": _hash(folder / "captura.json"), "consultas_api_en_prediccion": 0}
    try:
        import numpy as np
        from huggingface_hub import try_to_load_from_cache
        from sentence_transformers import SentenceTransformer
        from src.nlp.rank_tenders import split_text
        from src.nlp.train_affinity import encode_tenders
        from src.nlp.predict_affinity_pilot import build_predictions

        classifier, saved, receipt = _load_candidate(root)
        if saved.get("pooling") != "mean_normalized_fragments" or saved.get("target") != "Afinidad":
            raise ValueError("La preparación del candidato no coincide con la representación esperada.")
        encoder_name = saved["embedding_model"]
        if encoder_name != "intfloat/multilingual-e5-small":
            raise ValueError("El candidato referencia otro encoder.")
        config = try_to_load_from_cache(encoder_name,"config.json",revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",)
        if not isinstance(config, str) or Path(config).parent.parent.name != "snapshots":
            raise RuntimeError("Falta la configuración local de E5; no se iniciarán descargas.")
        revision = Path(config).parent.name
        encoder = SentenceTransformer(encoder_name, device="cpu", local_files_only=True, revision=revision)
        if encoder.get_embedding_dimension() != 384:
            raise ValueError("El encoder no produce 384 dimensiones.")
        frame = _prepare(records)
        for text in frame["TextoLicitacion"]:
            chunks = split_text(text, encoder.tokenizer)
            if not chunks or any(len(encoder.tokenizer.encode("passage: " + chunk, add_special_tokens=True, truncation=False)) > encoder.max_seq_length for chunk in chunks):
                raise ValueError("Hay entradas que exceden el límite del encoder.")
        embeddings = encode_tenders(frame["TextoLicitacion"].tolist(), encoder)
        if (embeddings.shape != (len(frame), 384) or embeddings.dtype != np.float32
                or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)):
            raise ValueError("La matriz de embeddings no cumple el contrato.")
        result = build_predictions(frame, classifier, embeddings)
        result["LoteId"] = folder.name
        result["OrigenArchivo"] = result["CodigoExterno"].map(lambda code: (folder / sources[code]["archivo"]).relative_to(root).as_posix())
        result["HashFuenteSHA256"] = result["CodigoExterno"].map(lambda code: sources[code]["sha256"])
        result["FechaCapturaUTC"] = result["CodigoExterno"].map(lambda code: sources[code]["captura_utc"])
        result["HashTextoSHA256"] = result["TextoLicitacion"].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
        identity = {
            "ModeloAlias": "candidato", "ModeloVersion": str(receipt["model_version"]),
            "ModeloURI": f"models:/{MODEL_NAME}/{receipt['model_version']}", "ModeloRunId": receipt["run_id"],
            "HashModeloOriginalSHA256": receipt["model_sha256"], "Encoder": encoder_name,
            "RevisionEncoderLocal": revision, "VersionPerfil": saved["profile_version"],
            "VersionCriterio": saved["labeling_policy_version"], "FechaPrediccionUTC": _utc(),
            "EstadoRevision": "prediccion_pendiente_de_revision_humana",
        }
        for name, value in identity.items():
            result[name] = value
        if _originals(root) != before:
            raise RuntimeError("Cambió un archivo original durante la ejecución. Revisa antes de continuar.")
        _load_capture(folder)
        if _hash(folder / "captura.json") != log["hash_captura_sha256"]:
            raise RuntimeError("El registro de captura cambió durante la predicción.")
        csv_path = output / "predicciones.csv"
        with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
            result.to_csv(stream, index=False)
        log.update(
            estado="completo", filas=len(result), identidad_modelo=identity,
            captura=folder.relative_to(root).as_posix(),
            distribucion_predicha={str(key): int(count) for key, count in result["AfinidadPredicha"].value_counts().sort_index().items()},
            predicciones_sha256=_hash(csv_path),
            originales_verificados_sha256=before,
            versiones={name: version(name) for name in ["mlflow", "scikit-learn", "numpy", "pandas", "sentence-transformers", "transformers", "torch"]},
            python=sys.version.split()[0],
            seleccion=capture["plan"],
            nota="Todas las clases se conservan. Las probabilidades no estiman adjudicación. No hay etiquetas humanas nuevas ni evaluación de calidad.",
        )
    except Exception as error:
        log["tipo_error"] = type(error).__name__
        # Sólo fallos locales: esta etapa no tiene acceso al cliente de la API.
        raise
    finally:
        log["fin_utc"] = _utc()
        guardar_captura(log, output / "registro_ejecucion.json")
    print("Predicciones:", csv_path)
    print("Registro:", output / "registro_ejecucion.json")
    return csv_path
