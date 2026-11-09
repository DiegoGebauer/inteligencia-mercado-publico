"""Pruebas sintéticas del checkpoint; sin API, cloud ni modelos."""
import copy
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.operations import state_checkpoint as checkpoint
from src.reporting import build_batch_history as history


PROJECT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-checkpoint-", dir=Path.cwd())
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "proyecto"
        self.root.mkdir()
        for name in checkpoint.CODE:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PROJECT / name).read_bytes())
        self.write("config/company_profile.yaml", b"name: consultora\n")
        self.write("config/cloud/model_bundle.json", b'{"schema_version":1}')
        self.write(".env", b"NO_EXPORTAR=secreto")
        self.write("mlflow_local/mlflow.db", b"no exportar")
        self.write("data/raw/quarantine/lote/captura.json", b"interrumpido")
        self.json("data/raw/batches/lote/listado.json", {"Listado": [{"CodigoExterno": "1-1-LE26"}]})
        self.json("data/raw/batches/lote/details/1-1-LE26.json", {"Listado": [{
            "CodigoExterno": "1-1-LE26", "Nombre": "Servicio de indicadores", "Fechas": {}}]})
        self.json("data/raw/batches/lote/captura.json", {
            "lote_id": "lote", "estado": "completo", "seleccionados": ["1-1-LE26"],
            "inicio_utc": "2026-10-16T03:00:00+00:00", "fin_utc": "2026-10-16T03:00:01+00:00",
            "consultas_api": 2, "plan": {"fecha_iso": "2026-10-14"},
            "listado": {"archivo": "listado.json", "sha256": self.digest("data/raw/batches/lote/listado.json")},
            "detalles": [{"codigo": "1-1-LE26", "estado": "correcto", "archivo": "details/1-1-LE26.json",
                          "sha256": self.digest("data/raw/batches/lote/details/1-1-LE26.json"),
                          "captura_utc": "2026-10-16T03:00:01+00:00"}]})
        snapshot = history.build_history(self.root)
        summary = json.loads((snapshot / "resumen.json").read_text(encoding="utf-8"))
        self.state = {"schema_version": 1, "fase": "completo", "lote": "lote",
                      "historial": snapshot.relative_to(self.root).as_posix(), "publicado_bigquery": True,
                      "publicacion": {"historial": summary["huella_entradas"], "job_id": "job-sintetico"}}
        self.json(checkpoint.STATE, self.state)

    def tearDown(self):
        assert self.base.parent == Path.cwd().resolve() and self.base.name.startswith("prueba-checkpoint-")
        self.temporary.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def json(self, name, value):
        return self.write(name, checkpoint.canonical(value))

    def digest(self, name):
        return checkpoint.sha((self.root / name).read_bytes())

    def archive(self):
        return checkpoint.create_checkpoint(self.root)

    def csv(self, name, row):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        self.write(name, text.getvalue().encode("utf-8-sig"))

    def add_prediction_and_review(self):
        capture = json.loads((self.root / "data/raw/batches/lote/captura.json").read_bytes())
        identity = {"ModeloURI": "models:/prueba/1", "ModeloRunId": "run1", "HashModeloOriginalSHA256": "modelo1",
                    "Encoder": "encoder-prueba", "RevisionEncoderLocal": "revision1", "VersionPerfil": "0.1.0", "VersionCriterio": "0.2.0"}
        row = {"CodigoExterno": "1-1-LE26", "LoteId": "lote", "FechaPrediccionUTC": "2026-10-16T04:00:00+00:00",
               "AfinidadPredicha": "0", "AfinidadEsperada": "0.4", "ProbAfinidad0": "0.8", "ProbAfinidad1": "0.05", "ProbAfinidad2": "0.1", "ProbAfinidad3": "0.05",
               "OrigenArchivo": "data/raw/batches/lote/details/1-1-LE26.json", "HashFuenteSHA256": capture["detalles"][0]["sha256"],
               "FechaCapturaUTC": capture["detalles"][0]["captura_utc"], "TextoLicitacion": "Servicio de indicadores",
               "HashTextoSHA256": checkpoint.sha(b"Servicio de indicadores"), **identity}
        csv_path = "data/processed/batches/lote/intento/predicciones.csv"
        self.csv(csv_path, row)
        self.json("data/processed/batches/lote/intento/registro_ejecucion.json", {
            "lote_id": "lote", "estado": "completo", "filas": 1,
            "hash_captura_sha256": self.digest("data/raw/batches/lote/captura.json"),
            "predicciones_sha256": self.digest(csv_path), "identidad_modelo": identity})
        criterion = "data/labeled/temporal_aligned/criterio.md"
        self.write(criterion, b"Criterio version 0.2.0")
        review = {key: row[key] for key in ("CodigoExterno", "LoteId", "AfinidadPredicha", "ModeloURI", "VersionCriterio", "HashModeloOriginalSHA256", "HashFuenteSHA256", "HashTextoSHA256", "OrigenArchivo")}
        review.update(RevisionId="revision1", AfinidadHumana="3", EstadoRevision="revision_humana_confirmada",
                      FechaRevisionLocal="2026-10-16", ConfirmacionUsuario="confirmo 3", OrigenPredicciones=csv_path,
                      HashPrediccionesSHA256=self.digest(csv_path), OrigenCriterio=criterion, HashCriterioSHA256=self.digest(criterion))
        self.csv("data/processed/reviews/lote/revision.csv", review)
        snapshot = history.build_history(self.root)
        summary = json.loads((snapshot / "resumen.json").read_bytes())
        self.state.update(historial=snapshot.relative_to(self.root).as_posix(),
                          publicacion={"historial": summary["huella_entradas"], "job_id": "job-con-prediccion"})
        self.json(checkpoint.STATE, self.state)

    def test_prediction_review_and_criterion_survive_recovery(self):
        self.add_prediction_and_review()
        path = self.archive()
        manifest = checkpoint.restore_checkpoint(path, self.base / "con_revision", code_root=self.root)
        self.assertEqual(manifest["conteos"]["predicciones"], 1)
        self.assertEqual(manifest["conteos"]["revisiones"], 1)
        self.assertIn("data/labeled/temporal_aligned/criterio.md", manifest["files"])
        view = self.base / "con_revision" / self.state["historial"] / "vista_actual.csv"
        with view.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual((row["AfinidadPredicha"], row["AfinidadHumana"]), ("0", "3"))

    def test_changed_review_criterion_is_rejected(self):
        self.add_prediction_and_review()
        self.write("data/labeled/temporal_aligned/criterio.md", b"alterado")
        with self.assertRaisesRegex(ValueError, "criterio"):
            checkpoint.build_plan(self.root)

    def test_plan_is_read_only_and_excludes_non_operational_data(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        plan = checkpoint.build_plan(self.root)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn(".env", plan["payload"])
        self.assertNotIn("mlflow_local/mlflow.db", plan["payload"])
        self.assertNotIn("data/raw/quarantine/lote/captura.json", plan["payload"])
        self.assertEqual(plan["manifest"]["conteos"]["capturas"], 1)

    def test_recovery_reproduces_history_and_cannot_overwrite(self):
        path = self.archive()
        manifest = checkpoint.verify_archive(path)
        destination = self.base / "recuperado"
        restored = checkpoint.restore_checkpoint(path, destination, code_root=self.root,
                                                 expected_sha256=checkpoint.file_sha(path))
        self.assertEqual(restored, manifest)
        for name in manifest["files"]:
            self.assertEqual((destination / name).read_bytes(), (self.root / name).read_bytes())
        self.assertEqual(json.loads((destination / checkpoint.STATE).read_text())["publicacion"]["job_id"], "job-sintetico")
        with self.assertRaises(FileExistsError):
            checkpoint.restore_checkpoint(path, self.root, code_root=self.root)

    def test_repeated_export_reuses_exact_archive(self):
        path = self.archive()
        original = (path.read_bytes(), path.stat().st_mtime_ns)
        self.assertEqual(self.archive(), path)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
        self.assertFalse((self.root / checkpoint.LOCK).exists())

    def test_active_or_stale_lock_blocks_plan_and_create(self):
        self.write(checkpoint.LOCK, b"otra ejecucion")
        with self.assertRaises(RuntimeError):
            checkpoint.build_plan(self.root)
        with self.assertRaises(FileExistsError):
            self.archive()
        self.assertEqual((self.root / checkpoint.LOCK).read_bytes(), b"otra ejecucion")

    def test_ambiguous_publication_is_not_exported_or_retried(self):
        self.json(checkpoint.STATE, dict(self.state, fase="publicacion_en_curso"))
        with self.assertRaisesRegex(ValueError, "fase completo"):
            self.archive()
        self.assertFalse((self.root / checkpoint.LOCK).exists())

    def test_changed_capture_is_rejected(self):
        self.write("data/raw/batches/lote/details/1-1-LE26.json", b"{}")
        with self.assertRaises(ValueError):
            checkpoint.build_plan(self.root)

    def test_missing_manifest_is_rejected_without_deletion(self):
        path = self.write("data/raw/batches/otro/listado.json", b"{}")
        with self.assertRaisesRegex(ValueError, "sin manifiesto"):
            checkpoint.build_plan(self.root)
        self.assertTrue(path.exists())

    def test_loose_prediction_is_rejected(self):
        self.write("data/processed/batches/lote/intento/predicciones.csv", b"CodigoExterno\n")
        with self.assertRaisesRegex(ValueError, "sin registro"):
            checkpoint.build_plan(self.root)

    def test_unlisted_file_inside_operational_tree_is_rejected(self):
        self.write("data/raw/batches/lote/credencial.txt", b"no exportar")
        with self.assertRaisesRegex(ValueError, "lista permitida"):
            checkpoint.build_plan(self.root)

    def test_wrong_publication_receipt_is_rejected(self):
        state = copy.deepcopy(self.state)
        state["publicacion"]["historial"] = "otra huella"
        self.json(checkpoint.STATE, state)
        with self.assertRaisesRegex(ValueError, "comprobante"):
            checkpoint.build_plan(self.root)

    def test_wrong_expected_archive_hash_leaves_no_destination(self):
        path = self.archive()
        with self.assertRaisesRegex(ValueError, "SHA256 esperado"):
            checkpoint.restore_checkpoint(path, self.base / "fallido", code_root=self.root, expected_sha256="0" * 64)
        self.assertFalse((self.base / "fallido").exists())

    def test_different_code_prevents_recovery(self):
        path = self.archive()
        self.write(checkpoint.CODE[1], b"cambio de codigo")
        with self.assertRaisesRegex(ValueError, "código difiere"):
            checkpoint.restore_checkpoint(path, self.base / "fallido", code_root=self.root)
        self.assertFalse((self.base / "fallido").exists())

    def rewrite_archive(self, source, *, extra=None, transform=None):
        output = self.base / "alterado.tar.gz"
        with tarfile.open(source, "r:gz") as incoming, tarfile.open(output, "w:gz") as outgoing:
            for member in incoming:
                data = incoming.extractfile(member).read()
                if transform:
                    data = transform(member.name, data)
                new_member = tarfile.TarInfo(member.name)
                new_member.size = len(data)
                outgoing.addfile(new_member, io.BytesIO(data))
            if extra:
                outgoing.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
        return output

    def test_payload_tampering_is_detected(self):
        path = self.rewrite_archive(self.archive(), transform=lambda name, data: b"X" * len(data) if name == checkpoint.STATE else data)
        with self.assertRaisesRegex(ValueError, "Contenido alterado"):
            checkpoint.verify_archive(path)

    def test_archive_links_are_rejected(self):
        member = tarfile.TarInfo("data/raw/batches/lote/details/2-2-LE26.json")
        member.type, member.linkname = tarfile.SYMTYPE, "../../../../.env"
        path = self.rewrite_archive(self.archive(), extra=member)
        with self.assertRaises(ValueError):
            checkpoint.verify_archive(path)

    def test_unexpected_archive_file_is_rejected(self):
        member = tarfile.TarInfo(".env")
        member.size = 1
        path = self.rewrite_archive(self.archive(), extra=member)
        with self.assertRaisesRegex(ValueError, "lista permitida"):
            checkpoint.verify_archive(path)

    def test_corrupt_existing_archive_is_not_replaced(self):
        path = self.archive()
        path.write_bytes(b"archivo corrupto")
        with self.assertRaises(tarfile.ReadError):
            self.archive()
        self.assertEqual(path.read_bytes(), b"archivo corrupto")

    def test_traversal_and_secret_paths_are_rejected(self):
        for name in ("../.env", "/data/raw/batches/lote/listado.json", ".env", "data/raw/batches/../secreto/listado.json",
                     "data/raw/batches/CON/listado.json", "data/raw/batches/lote/./listado.json", "data\\raw\\batches\\lote\\listado.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                checkpoint.safe_name(name)

    def test_modified_source_during_export_leaves_no_valid_archive(self):
        real_build = checkpoint.build_plan
        calls = 0

        def changing_plan(root, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.write("config/company_profile.yaml", b"name: cambio\n")
            return real_build(root, **kwargs)

        with patch.object(checkpoint, "build_plan", side_effect=changing_plan):
            with self.assertRaisesRegex(RuntimeError, "fuentes cambiaron"):
                self.archive()
        self.assertEqual(list((self.root / "data/processed/checkpoints").iterdir()), [])
        self.assertFalse((self.root / checkpoint.LOCK).exists())


if __name__ == "__main__":
    unittest.main()
