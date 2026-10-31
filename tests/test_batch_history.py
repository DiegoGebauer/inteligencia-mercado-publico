"""Prueba el historial con archivos sintéticos, sin API ni modelos."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "src/reporting/build_batch_history.py"
spec = importlib.util.spec_from_file_location("history_under_test", MODULE)
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
CODE = "1-1-LE26"


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.base = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-historial-", dir=self.base)
        self.root = Path(self.temporary.name).resolve()
        self.batch("lote1")
        self.predict("lote1", "intento1")
        self.review()

    def tearDown(self):
        assert self.root.parent == self.base and self.root.name.startswith("prueba-historial-")
        self.temporary.cleanup()

    def write_json(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_json(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def sha(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def write_csv(self, relative, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def rows(self, path):
        with Path(path).open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    def batch(self, batch_id, *, timestamp="2026-10-16T02:00:00+00:00", status="Publicada", reuse=None):
        relative = f"data/raw/batches/{batch_id}"
        self.write_json(f"{relative}/listado.json", {"Listado": [{"CodigoExterno": CODE}]})
        self.write_json(f"{relative}/details/{CODE}.json", {"Listado": [{
            "CodigoExterno": CODE, "Nombre": "Estudio de indicadores", "Estado": status,
            "Comprador": {"NombreOrganismo": "Organismo de prueba", "RegionUnidad": "Región de prueba", "Email": "no-exportar@example.test"},
            "Fechas": {"FechaPublicacion": "2026-10-14T10:00:00", "FechaCierre": "2026-10-28T15:00:00"},
        }]})
        manifest = {
            "lote_id": batch_id, "estado": "completo", "seleccionados": [CODE],
            "inicio_utc": timestamp, "fin_utc": timestamp, "consultas_api": 2,
            "plan": {"fecha_iso": "2026-10-14"},
            "listado": {"archivo": "listado.json", "sha256": self.sha(f"{relative}/listado.json")},
            "detalles": [{"codigo": CODE, "estado": "correcto", "captura_utc": timestamp,
                          "archivo": f"details/{CODE}.json", "sha256": self.sha(f"{relative}/details/{CODE}.json")}],
        }
        if reuse:
            manifest["origen"] = {"lote_id": reuse, "captura_sha256": self.sha(f"data/raw/batches/{reuse}/captura.json")}
            manifest["detalles"][0]["reutilizado_de_lote"] = reuse
        self.write_json(f"{relative}/captura.json", manifest)

    def predict(self, batch_id, attempt, timestamp="2026-10-16T03:00:00+00:00"):
        capture = self.read_json(f"data/raw/batches/{batch_id}/captura.json")
        identity = {"ModeloURI": "models:/prueba/1", "ModeloRunId": "run1", "HashModeloOriginalSHA256": "modelo1",
                    "Encoder": "encoder-prueba", "RevisionEncoderLocal": "revision1", "VersionPerfil": "0.1.0", "VersionCriterio": "0.2.0"}
        row = {"CodigoExterno": CODE, "LoteId": batch_id, "FechaPrediccionUTC": timestamp,
               "AfinidadPredicha": "0", "AfinidadEsperada": "0.4", "ProbAfinidad0": "0.8", "ProbAfinidad1": "0.05", "ProbAfinidad2": "0.1", "ProbAfinidad3": "0.05",
               "OrigenArchivo": f"data/raw/batches/{batch_id}/details/{CODE}.json", "HashFuenteSHA256": capture["detalles"][0]["sha256"],
               "FechaCapturaUTC": capture["detalles"][0]["captura_utc"], "TextoLicitacion": "Servicio de indicadores",
               "HashTextoSHA256": hashlib.sha256(b"Servicio de indicadores").hexdigest(), **identity}
        csv_path = f"data/processed/batches/{batch_id}/{attempt}/predicciones.csv"
        self.write_csv(csv_path, [row])
        self.write_json(f"data/processed/batches/{batch_id}/{attempt}/registro_ejecucion.json", {
            "lote_id": batch_id, "estado": "completo", "filas": 1,
            "hash_captura_sha256": self.sha(f"data/raw/batches/{batch_id}/captura.json"),
            "predicciones_sha256": self.sha(csv_path), "identidad_modelo": identity,
        })

    def review(self, *, filename="revision_001.csv", human="3", confirmation="dejemoslo como 3"):
        csv_path = "data/processed/batches/lote1/intento1/predicciones.csv"
        row = self.rows(self.root / csv_path)[0]
        criterion = "data/labeled/temporal_aligned/criterio.md"
        path = self.root / criterion
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Criterio de prueba 0.2.0", encoding="utf-8")
        review = {key: row[key] for key in ["CodigoExterno", "LoteId", "AfinidadPredicha", "ModeloURI", "VersionCriterio", "HashModeloOriginalSHA256", "HashFuenteSHA256", "HashTextoSHA256", "OrigenArchivo"]}
        review.update(RevisionId="revision_001", AfinidadHumana=human, EstadoRevision="revision_asistida_confirmada_por_usuario",
                      FechaRevisionLocal="2026-10-16", ZonaHoraria="America/Santiago", Motivo="Componente analítico", ConfirmacionUsuario=confirmation,
                      BasesRevisadas="False", IncorporadoEntrenamiento="False", OrigenPredicciones=csv_path,
                      HashPrediccionesSHA256=self.sha(csv_path), OrigenCriterio=criterion, HashCriterioSHA256=self.sha(criterion))
        self.write_csv(f"data/processed/reviews/lote1/{filename}", [review])

    def test_same_inputs_reuse_snapshot_and_leave_sources_intact(self):
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        first = history.build_history(self.root)
        bytes_before = {path.name: path.read_bytes() for path in first.iterdir()}
        second = history.build_history(self.root)
        self.assertEqual(first, second)
        self.assertEqual(len(list(first.parent.iterdir())), 1)
        self.assertEqual(bytes_before, {path.name: path.read_bytes() for path in second.iterdir()})
        for name, data in before.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_recovery_copies_are_not_new_captures(self):
        self.batch("lote2", timestamp="2026-10-15T23:00:00-03:00", reuse="lote1")
        folder = history.build_history(self.root)
        summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["conteos"]["capturas"], 1)
        self.assertEqual(summary["conteos"]["lote_captura"], 2)
        self.assertEqual(summary["capturas_reutilizadas_sin_duplicar"], 1)

    def test_latest_view_keeps_prediction_and_human_revision_separate(self):
        folder = history.build_history(self.root)
        row = self.rows(folder / "vista_actual.csv")[0]
        self.assertEqual(row["AfinidadPredicha"], "0")
        self.assertEqual(row["AfinidadHumana"], "3")
        self.assertEqual(row["DifierePrediccionRevision"], "True")
        self.assertNotIn("Email", self.rows(folder / "capturas.csv")[0])

    def test_later_capture_is_preserved_without_carrying_old_revision(self):
        first = history.build_history(self.root)
        self.batch("lote2", timestamp="2026-10-17T02:00:00+00:00", status="Cerrada")
        second = history.build_history(self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.rows(second / "capturas.csv")), 2)
        row = self.rows(second / "vista_actual.csv")[0]
        self.assertEqual(row["EstadoCapturado"], "Cerrada")
        self.assertEqual(row["AfinidadHumana"], "")
        self.assertEqual(row["AfinidadPredicha"], "")
        self.assertEqual(row["RevisionesHistoricas"], "1")
        self.assertTrue((first / "resumen.json").is_file())

    def test_new_prediction_does_not_inherit_another_prediction_review(self):
        self.predict("lote1", "intento2", "2026-10-17T03:00:00+00:00")
        folder = history.build_history(self.root)
        self.assertEqual(len(self.rows(folder / "predicciones.csv")), 2)
        row = self.rows(folder / "vista_actual.csv")[0]
        self.assertEqual(row["AfinidadHumana"], "")
        self.assertEqual(row["EstadoRevision"], "pendiente_revision_humana")

    def test_changed_source_capture_stops_before_writing_history(self):
        (self.root / f"data/raw/batches/lote1/details/{CODE}.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cambió la captura"):
            history.build_history(self.root)
        self.assertFalse((self.root / "data/processed/history").exists())

    def test_changed_prediction_csv_is_rejected(self):
        path = self.root / "data/processed/batches/lote1/intento1/predicciones.csv"
        with path.open("a", encoding="utf-8") as file:
            file.write("cambio")
        with self.assertRaisesRegex(ValueError, "Cambió el CSV"):
            history.build_history(self.root)

    def test_corrupt_snapshot_is_not_silently_overwritten(self):
        folder = history.build_history(self.root)
        path = folder / "vista_actual.csv"
        path.write_text("alterado", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Cambió una salida"):
            history.build_history(self.root)
        self.assertEqual(path.read_text(encoding="utf-8"), "alterado")

    def test_identical_review_copy_is_deduplicated(self):
        self.review(filename="copia.csv")
        folder = history.build_history(self.root)
        self.assertEqual(len(self.rows(folder / "revisiones.csv")), 1)

    def test_conflicting_same_revision_id_is_rejected(self):
        self.review(filename="conflicto.csv", human="2")
        with self.assertRaisesRegex(ValueError, "valores distintos"):
            history.build_history(self.root)

    def test_label_without_explicit_confirmation_is_rejected(self):
        self.review(confirmation="")
        with self.assertRaisesRegex(ValueError, "confirmación explícita"):
            history.build_history(self.root)

    def test_path_escape_in_capture_is_rejected(self):
        relative = "data/raw/batches/lote1/captura.json"
        manifest = self.read_json(relative)
        manifest["detalles"][0]["archivo"] = "../../secreto.json"
        self.write_json(relative, manifest)
        with self.assertRaisesRegex(ValueError, "ruta inesperada"):
            history.build_history(self.root)

    def test_failed_predictions_are_counted_without_inventing_results(self):
        self.write_json("data/processed/batches/lote1/fallido/registro_ejecucion.json", {"lote_id": "lote1", "estado": "fallido"})
        folder = history.build_history(self.root)
        summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["intentos_prediccion_incompletos"], 1)
        self.assertEqual(summary["conteos"]["predicciones"], 1)

    def test_empty_project_has_header_only_tables(self):
        empty = self.root / "proyecto_vacio"
        empty.mkdir()
        folder = history.build_history(empty)
        for name in history.TABLE_FIELDS:
            self.assertEqual(self.rows(folder / name), [])

    def test_output_cannot_be_placed_among_training_inputs(self):
        with self.assertRaisesRegex(ValueError, "data/processed/history"):
            history.build_history(self.root, self.root / "data/labeled")


if __name__ == "__main__":
    unittest.main()
