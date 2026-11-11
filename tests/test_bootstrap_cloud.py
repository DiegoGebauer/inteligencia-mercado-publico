"""Arranque con paquetes y HTTP sintéticos: no usa credenciales, red o ML."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.experiments import export_model_bundle as bundle
from src.inference import portable_model as portable
from src.operations import bootstrap_cloud as bootstrap
from src.operations import state_checkpoint as checkpoint
from src.reporting import build_batch_history as history


PROJECT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, data, generation, status=200):
        self.data, self.status_code = data, status
        self.headers = {"Content-Length": str(len(data)), "x-goog-generation": generation}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        for start in range(0, len(self.data), 97):
            yield self.data[start:start + 97]


class Session:
    def __init__(self, objects):
        self.objects, self.calls, self.closed = objects, [], False
        self.mutate = None

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.mutate:
            mutate, self.mutate = self.mutate, None
            mutate()
        response = self.objects.get((url, kwargs["params"]["generation"]))
        return response or Response(b"", kwargs["params"]["generation"], 404)

    def close(self):
        self.closed = True


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-bootstrap-", dir=self.cwd)
        self.base = Path(self.temporary.name).resolve()
        self.root, self.destination = self.base / "proyecto", self.base / "arranque"
        self.root.mkdir()
        self.approved = patch.object(bundle, "MODEL_SHA", checkpoint.sha("clasificador sintético".encode()))
        self.approved.start()
        self.addCleanup(self.approved.stop)
        for name in set(checkpoint.CODE) | set(portable.RUNTIME_FILES):
            if name in checkpoint.CODE:
                self.write(name, (PROJECT / name).read_bytes())
            else:
                self.write(name, b"# fuente sintetica\n")
        self.write(bootstrap.SCHEMAS, b"{}")
        profile = self.write("config/company_profile.yaml", b"name: consultora\n")
        payload = [bundle.payload_bytes("classifier.joblib", "clasificador sintético".encode()),
                   bundle.payload_bytes("regression_cases.json", b"{}"),
                   bundle.payload_bytes("company_profile.yaml", profile.read_bytes()),
                   bundle.payload_bytes("labeling_policy.md", b"criterio de prueba")]
        payload += [bundle.payload_bytes("encoder/" + name, b"encoder sintetico") for name in bundle.ENCODER_FILES]
        identity = {"schema_version": 1, "model": {"name": bundle.MODEL_NAME, "version": "1",
                    "uri": f"models:/{bundle.MODEL_NAME}/1", "run_id": "run-test", "sha256": bundle.MODEL_SHA},
                    "encoder": {"name": bundle.ENCODER, "revision": bundle.REVISION, "dimension": 384, "max_tokens": 512},
                    "representation": dict(portable.REPRESENTATION), "profile_version": "0.1", "labeling_policy_version": "0.2",
                    "inference_code_sha256": {name: bundle.sha_file(self.root / name) for name in bundle.CODE_FILES},
                    "directories": list(bundle.DIRECTORIES),
                    "files": {item.name: {"bytes": item.size, "sha256": item.sha256} for item in payload}}
        manifest = dict(identity, bundle_id=checkpoint.sha(bundle.canonical(identity)))
        model_path = bundle.create_archive({"archive": self.root / "data/processed/deployment/modelo.tar.gz",
                                            "manifest": manifest, "payloads": payload, "bytes": sum(item.size for item in payload)})
        self.model_bytes = model_path.read_bytes()
        self.model = {"schema_version": 1, "model": manifest["model"], "encoder": manifest["encoder"],
                      "bundle": {"id": manifest["bundle_id"], "gcs_uri": "gs://bucket-prueba/modelos/modelo.tar.gz",
                                 "generation": "111", "sha256": checkpoint.sha(self.model_bytes),
                                 "bytes": len(self.model_bytes), "format": "tar.gz"}}
        self.json(bootstrap.MODEL_REF, self.model)
        listed = self.json("data/raw/batches/lote/listado.json", {"Listado": [{"CodigoExterno": "1-1-LE26"}]})
        detail = self.json("data/raw/batches/lote/details/1-1-LE26.json", {"Listado": [{"CodigoExterno": "1-1-LE26", "Nombre": "Datos"}]})
        self.detail_bytes = detail.read_bytes()
        self.json("data/raw/batches/lote/captura.json", {"lote_id": "lote", "estado": "completo", "seleccionados": ["1-1-LE26"],
                  "inicio_utc": "2026-10-16T03:00:00+00:00", "fin_utc": "2026-10-16T03:01:00+00:00",
                  "consultas_api": 2, "plan": {"fecha_iso": "2026-10-14"},
                  "listado": {"archivo": "listado.json", "sha256": checkpoint.sha(listed.read_bytes())},
                  "detalles": [{"codigo": "1-1-LE26", "estado": "correcto", "archivo": "details/1-1-LE26.json",
                                "sha256": checkpoint.sha(self.detail_bytes), "captura_utc": "2026-10-16T03:01:00+00:00"}]})
        snapshot = history.build_history(self.root)
        self.json(checkpoint.STATE, {"schema_version": 1, "fase": "completo", "lote": "lote",
                                   "historial": snapshot.relative_to(self.root).as_posix(), "publicado_bigquery": False})
        checkpoint_path = checkpoint.create_checkpoint(self.root)
        state_manifest = checkpoint.verify_archive(checkpoint_path)
        self.state_bytes = checkpoint_path.read_bytes()
        self.state = {"schema_version": 1, "purpose": "closed_incremental_state", "counts": state_manifest["conteos"],
                      "checkpoint": {"id": state_manifest["checkpoint_id"], "gcs_uri": "gs://bucket-prueba/estado/estado.tar.gz",
                                     "generation": "222", "sha256": checkpoint.sha(self.state_bytes), "bytes": len(self.state_bytes),
                                     "format": "tar.gz", "files": len(state_manifest["files"])}}
        self.json(bootstrap.STATE_REF, self.state)
        self.write(".env", b"NO_COPIAR=secreto")
        self.model_url = bootstrap.parse_object(self.model["bundle"])["url"]
        self.state_url = bootstrap.parse_object(self.state["checkpoint"])["url"]
        self.session = Session({(self.model_url, "111"): Response(self.model_bytes, "111"),
                                (self.state_url, "222"): Response(self.state_bytes, "222")})
        self.factory = Mock(return_value=self.session)

    def tearDown(self):
        assert self.base.parent == self.cwd and self.base.name.startswith("prueba-bootstrap-")
        self.temporary.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def json(self, name, data):
        return self.write(name, checkpoint.canonical(data))

    def prepare(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            return bootstrap.prepare_workspace(self.root, self.destination, session_factory=self.factory)

    def report(self):
        return json.loads((self.destination / "bootstrap.json").read_text(encoding="utf-8"))

    def test_plan_reads_only_and_does_not_consult_credentials(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(bootstrap, "authorized_session", side_effect=AssertionError("No credenciales")):
            plan = bootstrap.build_plan(self.root, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual(bootstrap.plan_summary(plan)["escrituras_cloud"], 0)

    def test_prepare_downloads_fixed_generations_and_recovers_cloud_bytes(self):
        self.write("data/raw/batches/lote/details/1-1-LE26.json", b"LOCAL DIFERENTE")
        with patch.object(portable, "_load_local", side_effect=AssertionError("No inferencia")) as load:
            result = self.prepare()
        self.assertEqual(result["estado"], "preparado")
        self.assertEqual(len(self.session.calls), 2)
        self.assertEqual([call[1]["params"]["generation"] for call in self.session.calls], ["111", "222"])
        workspace = Path(result["workspace"])
        self.assertEqual((workspace / "data/raw/batches/lote/details/1-1-LE26.json").read_bytes(), self.detail_bytes)
        self.assertEqual(result["conteos"], self.state["counts"])
        self.assertFalse((workspace / ".env").exists())
        self.assertFalse((workspace / "mlflow_local").exists())
        self.assertEqual(self.report()["fase"], "listo")
        self.assertTrue(self.session.closed)
        load.assert_not_called()

    def test_url_encodes_object_path_and_includes_generation(self):
        request = bootstrap.parse_object(self.model["bundle"])
        self.assertEqual(request["params"], {"alt": "media", "generation": "111"})
        self.assertIn("modelos%2Fmodelo.tar.gz", request["url"])

    def test_existing_destination_is_never_replaced(self):
        self.destination.mkdir()
        marker = self.destination / "conservar.txt"
        marker.write_bytes(b"CONSERVAR")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.factory.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"CONSERVAR")

    def test_malformed_reference_is_rejected_before_authentication(self):
        self.state["checkpoint"]["generation"] = "latest"
        self.json(bootstrap.STATE_REF, self.state)
        with self.assertRaises(ValueError):
            self.prepare()
        self.factory.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_http_failure_does_not_fall_back_to_local_model(self):
        self.session.objects[(self.model_url, "111")] = Response(b"", "111", 404)
        with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
            self.prepare()
        self.assertEqual(self.report()["estado"], "fallido")
        self.assertEqual(len(self.session.calls), 1)
        self.assertFalse((self.destination / "workspace").exists())

    def test_wrong_response_generation_is_rejected(self):
        self.session.objects[(self.model_url, "111")].headers["x-goog-generation"] = "999"
        with self.assertRaisesRegex(ValueError, "otra generación"):
            self.prepare()
        self.assertFalse((self.destination / "workspace").exists())

    def test_payload_hash_mismatch_is_rejected(self):
        response = self.session.objects[(self.model_url, "111")]
        response.data = b"X" * len(response.data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.assertTrue((self.destination / "descargas/modelo.tar.gz.part").exists())
        self.assertFalse((self.destination / "descargas/modelo.tar.gz").exists())

    def test_oversized_response_is_rejected_before_writing_excess_bytes(self):
        response = self.session.objects[(self.model_url, "111")]
        response.headers.pop("Content-Length")
        response.data += b"EXTRA"
        with self.assertRaisesRegex(ValueError, "excede"):
            self.prepare()
        self.assertNotEqual(self.report()["estado"], "preparado")

    def test_wrong_content_length_is_rejected(self):
        self.session.objects[(self.model_url, "111")].headers["Content-Length"] = "1"
        with self.assertRaisesRegex(ValueError, "tamaño informado"):
            self.prepare()

    def test_code_incompatible_with_checkpoint_is_rejected(self):
        self.write("src/run_incremental.py", b"cambio no compatible")
        with self.assertRaisesRegex(ValueError, "otra versión del código"):
            self.prepare()
        self.assertFalse((self.destination / "workspace").exists())

    def test_wrong_checkpoint_identity_is_rejected(self):
        self.state["checkpoint"]["id"] = "0" * 64
        self.json(bootstrap.STATE_REF, self.state)
        with self.assertRaisesRegex(ValueError, "identidad o conteos"):
            self.prepare()
        self.assertFalse((self.destination / "workspace").exists())

    def test_wrong_checkpoint_counts_are_rejected(self):
        self.state["counts"]["capturas"] = 999
        self.json(bootstrap.STATE_REF, self.state)
        with self.assertRaisesRegex(ValueError, "identidad o conteos"):
            self.prepare()

    def test_different_model_reference_inside_checkpoint_is_not_overwritten(self):
        self.model["model"]["run_id"] = "otro-run"
        self.json(bootstrap.MODEL_REF, self.model)
        with self.assertRaisesRegex(ValueError, "modelo distinto"):
            self.prepare()
        restored = checkpoint.read_json((self.destination / "workspace" / bootstrap.MODEL_REF).read_bytes())
        self.assertEqual(restored["model"]["run_id"], "run-test")

    def test_source_change_during_download_cannot_produce_ready_workspace(self):
        self.session.mutate = lambda: self.write(bootstrap.SCHEMAS, b'{"cambio":true}')
        with self.assertRaisesRegex(RuntimeError, "Cambió el código"):
            self.prepare()
        self.assertEqual(self.report()["estado"], "fallido")

    def test_failed_authentication_is_recorded_without_exception_secrets(self):
        self.factory.side_effect = RuntimeError("TOKEN_NO_REGISTRAR")
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertEqual(self.report()["tipo_error"], "RuntimeError")
        self.assertNotIn("TOKEN_NO_REGISTRAR", (self.destination / "bootstrap.json").read_text())

    def test_download_deadline_is_enforced(self):
        with patch.object(bootstrap.time, "monotonic", side_effect=[0, 601]):
            with self.assertRaises(TimeoutError):
                bootstrap.download_object(self.model["bundle"], self.base / "fuera.tar.gz", self.session)

    def test_existing_partial_file_is_not_reused_or_overwritten(self):
        destination = self.base / "modelo.tar.gz"
        partial = self.base / "modelo.tar.gz.part"
        partial.write_bytes(b"CONSERVAR")
        with self.assertRaises(FileExistsError):
            bootstrap.download_object(self.model["bundle"], destination, self.session)
        self.assertEqual(partial.read_bytes(), b"CONSERVAR")
        self.assertEqual(self.session.calls, [])

    def test_cli_without_prepare_does_not_authenticate_or_download(self):
        with patch.object(bootstrap, "prepare_workspace", side_effect=AssertionError("No descargar")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(bootstrap.main(["--root", str(self.root), "--destino", str(self.destination)]), 0)
        self.assertIn("Sólo plan", output.getvalue())
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
