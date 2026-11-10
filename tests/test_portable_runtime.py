"""Contratos del cargador con archivos sintéticos; no carga joblib, E5 ni cloud."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.experiments import export_model_bundle as bundle
from src.inference import portable_model as portable
from src.ingestion import recent_batch as batch


class PortableRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-portable-", dir=self.cwd)
        self.root = Path(self.temporary.name).resolve()
        self.model_patch = patch.object(bundle, "MODEL_SHA", bundle.hashlib.sha256(b"clasificador-ficticio").hexdigest())
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(portable.ENVIRONMENT_VARIABLE, None)
        for name in portable.RUNTIME_FILES:
            self.write(name, b"# codigo de referencia\n")
        profile = self.write("config/company_profile.yaml", b"name: prueba\n")
        payload = [bundle.payload_bytes("classifier.joblib", b"clasificador-ficticio"),
                   bundle.payload_bytes("regression_cases.json", b'{"cases":[]}'),
                   bundle.payload_bytes("company_profile.yaml", profile.read_bytes()),
                   bundle.payload_bytes("labeling_policy.md", b"criterio ficticio")]
        payload += [bundle.payload_bytes("encoder/" + name, b"encoder ficticio") for name in bundle.ENCODER_FILES]
        identity = {"schema_version": 1,
                    "model": {"name": bundle.MODEL_NAME, "version": "1", "uri": f"models:/{bundle.MODEL_NAME}/1",
                              "run_id": "run-prueba", "sha256": bundle.MODEL_SHA},
                    "encoder": {"name": bundle.ENCODER, "revision": bundle.REVISION, "dimension": 384, "max_tokens": 512},
                    "representation": dict(portable.REPRESENTATION), "profile_version": "0.1.0", "labeling_policy_version": "0.2.0",
                    "reference_environment": {"python": "3.13-test", "packages": {p: "test" for p in portable.PACKAGES}},
                    "inference_code_sha256": {name: bundle.sha_file(self.root / name) for name in bundle.CODE_FILES},
                    "directories": list(bundle.DIRECTORIES),
                    "files": {item.name: {"bytes": item.size, "sha256": item.sha256} for item in payload}}
        manifest = dict(identity, bundle_id=bundle.hashlib.sha256(bundle.canonical(identity)).hexdigest())
        self.archive = bundle.create_archive({"archive": self.root / "data/processed/deployment/modelo-prueba.tar.gz",
                                              "manifest": manifest, "payloads": payload,
                                              "bytes": sum(item.size for item in payload)})
        self.descriptor = {"schema_version": 1, "model": manifest["model"], "encoder": manifest["encoder"],
                           "bundle": {"id": manifest["bundle_id"], "gcs_uri": "gs://bucket-prueba/modelos/uno.tar.gz",
                                      "generation": "12345", "sha256": bundle.sha_file(self.archive),
                                      "bytes": self.archive.stat().st_size, "format": "tar.gz"}}
        self.save_descriptor()

    def tearDown(self):
        assert self.root.parent == self.cwd and self.root.name.startswith("prueba-portable-")
        self.temporary.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def save_descriptor(self):
        self.write(portable.DESCRIPTOR, bundle.canonical(self.descriptor))

    def inspect(self):
        return portable.inspect_bundle(self.archive, self.root)

    def test_plan_is_read_only_and_does_not_load_or_require_mlflow_files(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(portable, "_load_local", side_effect=AssertionError("No cargar")) as load:
            plan = self.inspect()
        self.assertFalse((self.root / "mlflow_local").exists())
        self.assertFalse((self.root / "models").exists())
        self.assertEqual(plan["manifest"]["model"]["sha256"], bundle.MODEL_SHA)
        load.assert_not_called()
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_mismatched_archive_sha_prevents_deserialization(self):
        self.descriptor["bundle"]["sha256"] = "0" * 64
        self.save_descriptor()
        with patch.object(portable, "_load_local") as load:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                with portable.loaded_model(self.archive, self.root):
                    self.fail("No se debe cargar")
        load.assert_not_called()

    def test_mismatched_size_and_missing_generation_are_rejected(self):
        for key, value in (("bytes", 1), ("generation", "")):
            old = self.descriptor["bundle"][key]
            self.descriptor["bundle"][key] = value
            self.save_descriptor()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.inspect()
            self.descriptor["bundle"][key] = old

    def test_different_bundle_id_is_rejected(self):
        self.descriptor["bundle"]["id"] = "0" * 64
        self.save_descriptor()
        with self.assertRaisesRegex(ValueError, "identidad"):
            self.inspect()

    def test_different_model_or_encoder_reference_is_rejected(self):
        self.descriptor["model"]["run_id"] = "otro-run"
        self.save_descriptor()
        with self.assertRaisesRegex(ValueError, "referencia versionada"):
            self.inspect()

    def test_changed_core_inference_code_is_rejected(self):
        self.write(bundle.CODE_FILES[0], b"cambio no validado\n")
        with self.assertRaisesRegex(ValueError, "código de inferencia"):
            self.inspect()

    def test_changed_company_profile_is_rejected(self):
        self.write("config/company_profile.yaml", b"name: otro\n")
        with self.assertRaisesRegex(ValueError, "perfil de empresa"):
            self.inspect()

    def test_environment_mismatch_stops_before_loading_and_extraction(self):
        with patch.object(portable, "version", return_value="version-distinta"), patch.object(portable, "_load_local") as load:
            with self.assertRaisesRegex(RuntimeError, "versión"):
                with portable.loaded_model(self.archive, self.root):
                    pass
        load.assert_not_called()
        self.assertEqual(list(self.archive.parent.glob("runtime-modelo-*")), [])

    def test_local_loading_uses_whitelisted_files_and_cleans_up_on_success(self):
        paths = []

        def load(directory, manifest):
            paths.append(directory)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertTrue((directory / "encoder/2_Normalize").is_dir())
            self.assertEqual((directory / "classifier.joblib").read_bytes(), b"clasificador-ficticio")
            return "clasificador", "encoder", {"target": "Afinidad"}

        with patch.object(portable, "version", return_value="test"), patch.object(portable, "_load_local", side_effect=load):
            with portable.loaded_model(self.archive, self.root) as runtime:
                self.assertEqual(runtime.classifier, "clasificador")
                self.assertEqual(runtime.lineage, self.descriptor["bundle"])
            self.assertIsNone(runtime.encoder)
        self.assertFalse(paths[0].exists())
        self.assertTrue(self.archive.exists())

    def test_runtime_failure_still_releases_its_own_temporary_files(self):
        with patch.object(portable, "version", return_value="test"), \
                patch.object(portable, "_load_local", return_value=("clasificador", "encoder", {})):
            with self.assertRaisesRegex(RuntimeError, "fallo inventado"):
                with portable.loaded_model(self.archive, self.root):
                    raise RuntimeError("fallo inventado")
        self.assertEqual(list(self.archive.parent.glob("runtime-modelo-*")), [])

    def test_offline_environment_restores_previous_settings(self):
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        with self.assertRaises(RuntimeError):
            with portable.offline_environment():
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                raise RuntimeError("fallo")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
        self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)

    def test_missing_environment_variable_preserves_legacy_selection(self):
        self.assertIsNone(portable.configured_archive(self.root))
        self.assertIn("candidato", batch.plan_lote("14102026")["modelo"])

    def test_relative_archive_is_resolved_against_project_root(self):
        os.environ[portable.ENVIRONMENT_VARIABLE] = self.archive.relative_to(self.root).as_posix()
        self.assertEqual(portable.configured_archive(self.root), self.archive)

    def test_empty_setting_or_cloud_uri_never_falls_back_to_mlflow(self):
        for setting in ("", "   ", "gs://bucket/model.tar.gz"):
            os.environ[portable.ENVIRONMENT_VARIABLE] = setting
            with self.subTest(setting=setting), patch.object(batch, "_load_candidate") as legacy, self.assertRaises(ValueError):
                batch.predecir_lote(self.root / "data/raw/batches/lote", raiz=self.root)
            legacy.assert_not_called()

    def test_existing_coordinator_function_dispatches_to_portable_without_mlflow(self):
        os.environ[portable.ENVIRONMENT_VARIABLE] = str(self.archive)
        expected = self.root / "predicciones.csv"
        with patch.object(portable, "predict_batch", return_value=expected) as predict, \
                patch.object(batch, "_load_candidate", side_effect=AssertionError("No MLflow")):
            result = batch.predecir_lote(self.root / "data/raw/batches/lote", raiz=self.root)
        self.assertEqual(result, expected)
        predict.assert_called_once_with(self.root / "data/raw/batches/lote", root=self.root, archive=self.archive)
        self.assertIn("portable", batch.plan_lote("14102026")["modelo"])

    def test_invalid_portable_file_is_not_silently_replaced_by_legacy_mode(self):
        os.environ[portable.ENVIRONMENT_VARIABLE] = str(self.archive)
        with patch.object(portable, "predict_batch", side_effect=ValueError("paquete inválido")), \
                patch.object(batch, "_load_candidate") as legacy:
            with self.assertRaisesRegex(ValueError, "paquete inválido"):
                batch.predecir_lote(self.root / "data/raw/batches/lote", raiz=self.root)
        legacy.assert_not_called()

    def test_second_complete_prediction_is_not_generated(self):
        folder = self.root / "data/raw/batches/lote"
        existing = "data/processed/batches/lote/primero/registro_ejecucion.json"
        self.write(existing, b'{"estado":"completo"}')
        with patch.object(batch, "_load_capture", return_value=({}, [], {})), patch.object(portable, "loaded_model") as load:
            with self.assertRaises(FileExistsError):
                portable.predict_batch(folder, root=self.root, archive=self.archive)
        load.assert_not_called()

    def test_different_representation_is_rejected(self):
        plan = self.inspect()
        wrong = dict(plan["manifest"], representation={"chunk_tokens": 500})
        with patch.object(bundle, "verify_archive", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "representación"):
                self.inspect()

    def test_public_validation_cli_does_not_load_a_model(self):
        with patch.object(portable, "_load_local", side_effect=AssertionError("No cargar")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(portable.main(["--archivo", str(self.archive), "--root", str(self.root)]), 0)
        self.assertIn('"mlflow_requerido": false', output.getvalue())


if __name__ == "__main__":
    unittest.main()
