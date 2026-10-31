"""Pruebas con respuestas inventadas; sin API, pesos E5 ni entrenamiento."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / "src/ingestion/capture_store.py").is_file():
    # También permite verificar la propuesta antes de copiarla al proyecto.
    PROJECT = Path(r"D:\Python\Proyectos-Github\inteligencia-mercado-publico")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))
MODULE = Path(__file__).resolve().parents[1] / "src/ingestion/recent_batch.py"
spec = importlib.util.spec_from_file_location("batch_under_test", MODULE)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def detail(code):
    return {"Listado": [{
        "CodigoExterno": code, "Nombre": "Servicio de datos", "Descripcion": "Unificar reportes",
        "Estado": "Publicada", "Items": {"Listado": [{"NombreProducto": "Reportes", "Descripcion": "Tableros"}]},
    }]}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.test_base = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-lote-", dir=self.test_base)
        self.root = Path(self.temporary.name).resolve()
        self.client = types.ModuleType("src.api.mercado_publico_client")
        self.client.get_licitaciones_por_fecha = lambda date: {"Listado": [
            {"CodigoExterno": "2-2-LE26"}, {"CodigoExterno": "1-1-LE26"}, {"CodigoExterno": "1-1-LE26"},
        ]}
        self.client.get_licitacion_por_codigo = detail
        self.mock_api = patch.dict(sys.modules, {"src.api.mercado_publico_client": self.client})
        self.mock_api.start()
        self.clock_value = 0
        def fake_wait(seconds):
            self.clock_value += seconds
        original_control = batch._RequestControl
        self.control_factory = patch.object(batch, "_RequestControl", side_effect=lambda: original_control(clock=lambda: self.clock_value, wait=fake_wait))
        self.control_factory.start()

    def tearDown(self):
        self.control_factory.stop()
        self.mock_api.stop()
        assert self.root.parent == self.test_base and self.root.name.startswith("prueba-lote-")
        self.temporary.cleanup()

    def capture(self, limit=2):
        return batch.capturar_lote("14102026", limit, raiz=self.root)

    def test_plan_rejects_invalid_date_and_limits_without_writes(self):
        for date, limit in [("31022026", 5), ("2026-10-14", 5), ("14102026", 11), ("14102026", True)]:
            with self.assertRaises(ValueError):
                batch.plan_lote(date, limit)
        self.assertEqual(batch.plan_lote("14102026")["maximo_consultas_api"], 12)
        self.assertFalse(list(self.root.iterdir()))

    def test_capture_limits_deduplicates_and_preserves_separate_runs(self):
        folder = self.capture(1)
        manifest, records, _ = batch._load_capture(folder)
        self.assertEqual(manifest["seleccionados"], ["1-1-LE26"])
        self.assertEqual(manifest["consultas_api"], 2)
        original_bytes = (folder / "details/1-1-LE26.json").read_bytes()
        another = self.capture(1)
        self.assertNotEqual(folder, another)
        self.assertEqual((folder / "details/1-1-LE26.json").read_bytes(), original_bytes)
        self.assertEqual(len(records), 1)

    def test_api_failure_retains_responses_and_hides_error_messages(self):
        def fail(code):
            if code.startswith("2-"):
                raise RuntimeError("ticket=secreto-de-prueba")
            return detail(code)
        self.client.get_licitacion_por_codigo = fail
        with self.assertRaises(RuntimeError):
            self.capture()
        folder = next((self.root / "data/raw/batches").iterdir())
        text = (folder / "captura.json").read_text(encoding="utf-8")
        self.assertNotIn("secreto", text)
        self.assertEqual(json.loads(text)["estado"], "incompleto")
        self.assertTrue((folder / "details/1-1-LE26.json").is_file())
        with self.assertRaises(ValueError):
            batch._load_capture(folder)

    def test_empty_day_is_recorded_without_detail_calls(self):
        self.client.get_licitaciones_por_fecha = lambda date: {"Listado": []}
        folder = self.capture()
        manifest = batch._read(folder / "captura.json")
        self.assertEqual(manifest["estado"], "vacio")
        self.assertEqual(manifest["consultas_api"], 1)
        with self.assertRaises(ValueError):
            batch._load_capture(folder)

    def test_wrong_code_and_modified_capture_are_rejected(self):
        folder = self.capture(1)
        (folder / "details/1-1-LE26.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            batch._load_capture(folder)
        self.client.get_licitacion_por_codigo = lambda code: detail("9-9-LE26")
        with self.assertRaises(RuntimeError):
            self.capture(1)

    def test_safe_error_diagnostics_keep_http_status_without_secrets(self):
        status = batch._error_info(RuntimeError("Error al consultar Mercado Público. Estado: 503"))
        self.assertEqual(status["estado_http"], 503)
        self.assertEqual(status["motivo"], "http")
        timeout = batch._error_info(RuntimeError("Error al consultar Mercado Público. Estado: sin respuesta"))
        self.assertEqual(timeout["motivo"], "conexion_o_timeout")
        unsafe = batch._error_info(RuntimeError("https://ejemplo.test/?ticket=secreto"))
        self.assertNotIn("secreto", json.dumps(unsafe))

    def test_capture_429_retries_once_and_counts_actual_requests(self):
        calls = []
        def fetch(code):
            calls.append(code)
            if len(calls) == 1:
                raise RuntimeError("Error al consultar Mercado Público. Estado: 429")
            return detail(code)
        self.client.get_licitacion_por_codigo = fetch
        folder = self.capture(1)
        manifest = batch._read(folder / "captura.json")
        self.assertEqual(manifest["consultas_api"], 3)
        self.assertEqual(manifest["detalles"][0]["intentos_http"], 2)
        self.assertEqual(manifest["control_api"]["eventos"][-1]["espera_previa_segundos"], 30)
        self.assertGreaterEqual(self.clock_value, 33)

    def test_persistent_429_stops_remaining_http_requests(self):
        def fail(code):
            raise RuntimeError("Error al consultar Mercado Público. Estado: 429")
        with patch.object(self.client, "get_licitacion_por_codigo", side_effect=fail) as fetch:
            with self.assertRaises(RuntimeError):
                self.capture()
        self.assertEqual(fetch.call_count, 2)
        folder = next((self.root / "data/raw/batches").iterdir())
        manifest = batch._read(folder / "captura.json")
        self.assertEqual(manifest["consultas_api"], 3)
        self.assertTrue(manifest["control_api"]["bloqueado_por_429"])
        self.assertTrue(manifest["detalles"][1]["omitido_por_espera"])
        self.assertEqual(manifest["detalles"][1]["intentos_http"], 0)

    def test_retry_after_longer_than_budget_stops_without_premature_retry(self):
        error = RuntimeError("Error al consultar Mercado Público. Estado: 429")
        error.retry_after_seconds = 120
        with patch.object(self.client, "get_licitacion_por_codigo", side_effect=error) as fetch:
            with self.assertRaises(RuntimeError):
                self.capture()
        self.assertEqual(fetch.call_count, 1)
        folder = next((self.root / "data/raw/batches").iterdir())
        manifest = batch._read(folder / "captura.json")
        self.assertEqual(manifest["consultas_api"], 2)
        self.assertEqual(manifest["control_api"]["eventos"][-1]["espera_requerida_segundos"], 120)
        with patch.object(self.client, "get_licitacion_por_codigo") as fetch:
            with self.assertRaisesRegex(RuntimeError, "en espera"):
                batch.recuperar_lote(folder, raiz=self.root)
        fetch.assert_not_called()

    def test_retry_after_is_respected_when_shorter_than_budget(self):
        calls = []
        def fetch(code):
            calls.append(code)
            if len(calls) == 1:
                error = RuntimeError("Error al consultar Mercado Público. Estado: 429")
                error.retry_after_seconds = 17
                raise error
            return detail(code)
        self.client.get_licitacion_por_codigo = fetch
        folder = self.capture(1)
        manifest = batch._read(folder / "captura.json")
        self.assertEqual(manifest["control_api"]["eventos"][-1]["espera_previa_segundos"], 17)

    def test_recovery_queries_only_missing_details_and_keeps_original_bytes(self):
        def fail_one(code):
            if code.startswith("2-"):
                raise RuntimeError("Error al consultar Mercado Público. Estado: 503")
            return detail(code)
        self.client.get_licitacion_por_codigo = fail_one
        with self.assertRaises(RuntimeError):
            self.capture()
        source = next((self.root / "data/raw/batches").iterdir())
        before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
        with patch.object(self.client, "get_licitaciones_por_fecha", side_effect=AssertionError("No repetir el listado")), patch.object(self.client, "get_licitacion_por_codigo", side_effect=detail) as fetch:
            recovered = batch.recuperar_lote(source, raiz=self.root)
        fetch.assert_called_once_with("2-2-LE26")
        manifest, records, _ = batch._load_capture(recovered)
        self.assertEqual(manifest["consultas_api"], 1)
        self.assertEqual(manifest["detalles_reutilizados"], 1)
        self.assertEqual(manifest["origen"]["captura_sha256"], batch._hash(source / "captura.json"))
        self.assertEqual(len(records), 2)
        for name, data in before.items():
            self.assertEqual((source / name).read_bytes(), data)
        self.assertEqual((recovered / "details/1-1-LE26.json").read_bytes(), before["details/1-1-LE26.json"])
        with self.assertRaises(ValueError):
            batch.recuperar_lote(recovered, raiz=self.root)

    def test_recovery_retains_new_failure_and_rejects_corrupted_original(self):
        def fail_one(code):
            if code.startswith("2-"):
                raise RuntimeError("Error al consultar Mercado Público. Estado: sin respuesta")
            return detail(code)
        self.client.get_licitacion_por_codigo = fail_one
        with self.assertRaises(RuntimeError):
            self.capture()
        source = next((self.root / "data/raw/batches").iterdir())
        with self.assertRaises(RuntimeError):
            batch.recuperar_lote(source, raiz=self.root)
        derived = next(path for path in (self.root / "data/raw/batches").iterdir() if path != source)
        manifest = batch._read(derived / "captura.json")
        self.assertEqual(manifest["estado"], "incompleto")
        self.assertEqual(manifest["detalles"][1]["motivo"], "conexion_o_timeout")
        (source / "details/1-1-LE26.json").write_text("{}", encoding="utf-8")
        with patch.object(self.client, "get_licitacion_por_codigo", side_effect=AssertionError("No consultar datos corruptos")) as fetch:
            with self.assertRaises(ValueError):
                batch.recuperar_lote(source, raiz=self.root)
        fetch.assert_not_called()

    def test_normalization_keeps_identity_and_supports_zero_items(self):
        records = detail("1-1-LE26")["Listado"]
        frame = batch._prepare(records)
        self.assertEqual(frame.iloc[0]["CodigoExterno"], "1-1-LE26")
        self.assertEqual(frame.iloc[0]["EstadoCapturado"], "Publicada")
        self.assertIn("Tableros", frame.iloc[0]["TextoLicitacion"])
        self.assertNotIn("Comprador_NombreUsuario", frame)
        records[0]["Items"]["Listado"] = []
        frame = batch._prepare(records)
        self.assertEqual(frame.iloc[0]["TextoLicitacion"], "Servicio de datos | Unificar reportes")

    def test_prediction_exports_all_classes_and_prevents_second_complete_run(self):
        import numpy as np
        folder = self.capture()
        encoder_module = types.ModuleType("sentence_transformers")
        encoder = types.SimpleNamespace(
            max_seq_length=512, get_embedding_dimension=lambda: 384,
            tokenizer=types.SimpleNamespace(encode=lambda *a, **kw: [1, 2]),
        )
        encoder_module.SentenceTransformer = lambda *a, **kw: encoder
        training = types.ModuleType("src.nlp.train_affinity")
        training.encode_tenders = lambda texts, encoder: np.tile(np.r_[np.float32(1), np.zeros(383, dtype=np.float32)], (len(texts), 1))
        ranking = types.ModuleType("src.nlp.rank_tenders")
        ranking.split_text = lambda text, tokenizer: [text]
        hub = types.ModuleType("huggingface_hub")
        hub.try_to_load_from_cache = lambda *a, **kw: str(self.root / "snapshots/revision-prueba/config.json")
        classifier = types.SimpleNamespace(
            classes_=np.array([0, 1, 2, 3]),
            predict=lambda values: np.array([0, 3]),
            predict_proba=lambda values: np.array([[.7, .1, .1, .1], [.1, .1, .1, .7]]),
        )
        saved = {"target": "Afinidad", "pooling": "mean_normalized_fragments", "embedding_model": "intfloat/multilingual-e5-small", "profile_version": "0.1.0", "labeling_policy_version": "0.2.0"}
        receipt = {"model_version": "1", "run_id": "run-prueba", "model_sha256": "hash-prueba"}
        modules = {"sentence_transformers": encoder_module, "src.nlp.train_affinity": training,
                   "src.nlp.rank_tenders": ranking, "huggingface_hub": hub}
        import pandas as pd
        with patch.dict(sys.modules, modules), patch.object(batch, "_load_candidate", return_value=(classifier, saved, receipt)), patch.object(batch, "_originals", return_value={"original": "huella"}):
            csv = batch.predecir_lote(folder, raiz=self.root)
            result = pd.read_csv(csv)
            self.assertEqual(set(result["AfinidadPredicha"]), {0, 3})
            self.assertEqual(result["CodigoExterno"].tolist(), ["2-2-LE26", "1-1-LE26"])
            self.assertEqual(set(result["EstadoRevision"]), {"prediccion_pendiente_de_revision_humana"})
            self.assertTrue(result["HashFuenteSHA256"].notna().all())
            log = batch._read(csv.parent / "registro_ejecucion.json")
            self.assertEqual(log["estado"], "completo")
            self.assertEqual(log["consultas_api_en_prediccion"], 0)
            self.assertEqual(log["predicciones_sha256"], batch._hash(csv))
            with self.assertRaises(FileExistsError):
                batch.predecir_lote(folder, raiz=self.root)

    def test_failed_prediction_can_retry_without_api_or_losing_attempt(self):
        folder = self.capture(1)
        imports = {}
        for name, attribute in [("sentence_transformers", "SentenceTransformer"), ("huggingface_hub", "try_to_load_from_cache"),
                                ("src.nlp.rank_tenders", "split_text"), ("src.nlp.train_affinity", "encode_tenders"),
                                ("src.nlp.predict_affinity_pilot", "build_predictions")]:
            module = types.ModuleType(name)
            setattr(module, attribute, lambda *a, **kw: None)
            imports[name] = module
        with patch.dict(sys.modules, imports), patch.object(batch, "_originals", return_value={}), patch.object(batch, "_load_candidate", side_effect=RuntimeError("modelo no disponible")):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    batch.predecir_lote(folder, raiz=self.root)
        logs = list((self.root / "data/processed/batches" / folder.name).glob("*/registro_ejecucion.json"))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all(batch._read(path)["estado"] == "fallido" for path in logs))
        self.assertFalse(list((self.root / "data/processed/batches").rglob("predicciones.csv")))


if __name__ == "__main__":
    unittest.main()
