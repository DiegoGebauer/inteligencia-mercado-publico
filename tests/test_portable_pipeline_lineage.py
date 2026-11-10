"""Regresión del montaje de capturas recuperadas; sin inferencia, API o cloud."""
import json
from pathlib import Path
import tempfile
import unittest

from src.experiments import verify_portable_pipeline as verification
from src.reporting import build_batch_history as history


class PortablePipelineLineageTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-origen-portable-", dir=self.cwd)
        self.base = Path(self.temporary.name).resolve()
        self.root, self.stage = self.base / "proyecto", self.base / "temporal"
        self.root.mkdir()
        self.stage.mkdir()
        self.make_lot("original", complete=False)
        self.make_lot("recuperado", parent="original")

    def tearDown(self):
        assert self.base.parent == self.cwd and self.base.name.startswith("prueba-origen-portable-")
        self.temporary.cleanup()

    def write(self, name, value):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def make_lot(self, identifier, parent=None, complete=True):
        base = f"data/raw/batches/{identifier}"
        listed = self.write(base + "/listado.json", {"Listado": [{"CodigoExterno": "1-1-LE26"}, {"CodigoExterno": "2-2-LE26"}]})
        details = []
        for code in ("1-1-LE26", "2-2-LE26"):
            if not complete and code == "2-2-LE26":
                details.append({"codigo": code, "estado": "fallido", "motivo": "conexion_o_timeout"})
                continue
            path = self.write(f"{base}/details/{code}.json", {"Listado": [{"CodigoExterno": code, "Nombre": "Servicio de datos"}]})
            details.append({"codigo": code, "estado": "correcto", "archivo": f"details/{code}.json",
                            "sha256": history._sha(path.read_bytes()), "captura_utc": "2026-10-16T03:00:00+00:00"})
        manifest = {"lote_id": identifier, "estado": "completo" if complete else "incompleto",
                    "seleccionados": ["1-1-LE26", "2-2-LE26"], "detalles": details,
                    "listado": {"archivo": "listado.json", "sha256": history._sha(listed.read_bytes())},
                    "plan": {"fecha_iso": "2026-10-14"}, "inicio_utc": "2026-10-16T03:00:00+00:00",
                    "fin_utc": "2026-10-16T03:01:00+00:00", "consultas_api": 2}
        if parent:
            path = self.root / f"data/raw/batches/{parent}/captura.json"
            manifest["origen"] = {"lote_id": parent, "captura_sha256": history._sha(path.read_bytes())}
        self.write(base + "/captura.json", manifest)

    def change_child(self, origin):
        name = "data/raw/batches/recuperado/captura.json"
        manifest = json.loads((self.root / name).read_text(encoding="utf-8"))
        manifest["origen"] = origin
        self.write(name, manifest)

    def validate_stage(self):
        sources = history._Sources(self.stage)
        return history._captures(sources, history._inventory(self.stage)["capturas"])

    def test_parentless_lot_copies_only_its_required_inputs(self):
        secret = self.root / "data/raw/batches/original/no_copiar.env"
        secret.write_bytes(b"SECRETO")
        protected = verification.copy_reference_captures(self.root, self.stage, "original")
        self.assertEqual(len(protected), 3)
        self.assertFalse((self.stage / secret.relative_to(self.root)).exists())
        self.assertEqual(len(self.validate_stage()[4]), 1)

    def test_recovered_lot_preserves_origin_bytes_and_source_files(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        protected = verification.copy_reference_captures(self.root, self.stage, "recuperado")
        lots, captures, _, _, manifests = self.validate_stage()
        self.assertEqual(set(manifests), {"original", "recuperado"})
        self.assertEqual(len(lots), 2)
        self.assertEqual(len(captures), 2)
        for path, digest in protected.items():
            self.assertEqual(history._sha((self.stage / path.relative_to(self.root)).read_bytes()), digest)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_multilevel_recovery_copies_entire_ancestry(self):
        self.make_lot("final", parent="recuperado")
        verification.copy_reference_captures(self.root, self.stage, "final")
        _, captures, links, _, manifests = self.validate_stage()
        self.assertEqual(set(manifests), {"original", "recuperado", "final"})
        self.assertEqual(len(captures), 2)
        self.assertEqual(len(links), 5)

    def test_previous_single_folder_copy_reproduces_failure(self):
        inputs = verification.reference_capture_inputs(self.root, "recuperado")
        for name, data in inputs.items():
            if name.startswith("data/raw/batches/recuperado/"):
                target = self.stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "origen de la recuperación"):
            self.validate_stage()

    def test_changed_parent_manifest_fails_before_copying(self):
        path = self.root / "data/raw/batches/original/captura.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifiesto de origen"):
            verification.copy_reference_captures(self.root, self.stage, "recuperado")
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_missing_parent_fails_before_copying(self):
        self.change_child({"lote_id": "ausente", "captura_sha256": "0" * 64})
        with self.assertRaises(FileNotFoundError):
            verification.copy_reference_captures(self.root, self.stage, "recuperado")
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_parent_cannot_escape_capture_directory(self):
        self.change_child({"lote_id": "../../fuera", "captura_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "Identificador"):
            verification.copy_reference_captures(self.root, self.stage, "recuperado")
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_cycle_is_rejected_without_copying_or_looping(self):
        self.change_child({"lote_id": "recuperado", "captura_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "ciclo"):
            verification.copy_reference_captures(self.root, self.stage, "recuperado")
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_corrupt_parent_detail_is_not_accepted(self):
        path = self.root / "data/raw/batches/original/details/1-1-LE26.json"
        path.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "Cambió la captura"):
            verification.copy_reference_captures(self.root, self.stage, "recuperado")
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_existing_destination_file_is_not_overwritten(self):
        path = self.stage / "data/raw/batches/original/captura.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"NO REEMPLAZAR")
        with self.assertRaises(FileExistsError):
            verification.copy_reference_captures(self.root, self.stage, "recuperado")
        self.assertEqual(path.read_bytes(), b"NO REEMPLAZAR")


if __name__ == "__main__":
    unittest.main()
