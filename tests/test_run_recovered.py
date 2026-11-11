"""Lanzador con subproceso simulado y archivos inventados; sin API, ML o cloud."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.inference import portable_model as portable
from src.operations import run_recovered as recovered
from src.operations import state_checkpoint as checkpoint
from src.reporting import build_batch_history as history


PROJECT = Path(__file__).resolve().parents[1]


class RecoveredRunTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="prueba-corrida-recuperada-", dir=self.cwd)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "proyecto"
        self.directory = self.root / "data/processed/arranque_cloud_prueba_01"
        self.workspace = self.directory / "workspace"
        self.workspace.mkdir(parents=True)
        self.model = self.directory / "descargas/modelo.tar.gz"
        self.model.parent.mkdir()
        self.model.write_bytes(b"modelo de prueba")
        for name in checkpoint.CODE:
            self.write(self.workspace, name, (PROJECT / name).read_bytes())
        self.write(self.workspace, "config/company_profile.yaml", b"name: prueba\n")
        self.model_ref = {"id": "f" * 64, "sha256": "e" * 64, "generation": "123"}
        self.json(self.workspace, portable.DESCRIPTOR, {"schema_version": 1, "bundle": self.model_ref})
        self.make_batch("lote1", "1-1-LE26")
        snapshot = history.build_history(self.workspace)
        self.state = {"schema_version": 1, "fase": "completo", "lote": "lote1",
                      "historial": snapshot.relative_to(self.workspace).as_posix(), "publicado_bigquery": False}
        self.json(self.workspace, checkpoint.STATE, self.state)
        self.report = {"schema_version": 1, "estado": "preparado", "fase": "listo",
                       "workspace": str(self.workspace), "modelo_local": str(self.model),
                       "referencias": {"modelo": self.model_ref},
                       "codigo_y_config_sha256": {name: checkpoint.sha((self.workspace / name).read_bytes()) for name in checkpoint.CODE}}
        self.json(self.directory, "bootstrap.json", self.report)
        self.model_mock = patch.object(portable, "inspect_bundle", return_value={"descriptor": {"bundle": self.model_ref}})
        self.model_mock.start()
        self.addCleanup(self.model_mock.stop)
        self.ticket_mock = patch.object(recovered, "read_ticket_file", return_value="TICKET_SINTETICO_NO_REGISTRAR")
        self.ticket = self.ticket_mock.start()
        self.addCleanup(self.ticket_mock.stop)
        self.env = patch.dict(os.environ, {recovered.TICKET: ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def tearDown(self):
        assert self.base.parent == self.cwd and self.base.name.startswith("prueba-corrida-recuperada-")
        self.temporary.cleanup()

    def write(self, root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def json(self, root, name, value):
        return self.write(root, name, checkpoint.canonical(value))

    def make_batch(self, lot_id, code):
        base = f"data/raw/batches/{lot_id}"
        listed = self.json(self.workspace, base + "/listado.json", {"Listado": [{"CodigoExterno": code}]})
        detail = self.json(self.workspace, base + f"/details/{code}.json", {"Listado": [{"CodigoExterno": code, "Nombre": "Indicadores"}]})
        self.json(self.workspace, base + "/captura.json", {
            "lote_id": lot_id, "estado": "completo", "seleccionados": [code],
            "inicio_utc": "2000-01-01T03:00:00+00:00", "fin_utc": "2000-01-01T03:01:00+00:00", "consultas_api": 2,
            "plan": {"fecha_iso": "2000-01-01"}, "listado": {"archivo": "listado.json", "sha256": checkpoint.sha(listed.read_bytes())},
            "detalles": [{"codigo": code, "estado": "correcto", "archivo": f"details/{code}.json",
                          "sha256": checkpoint.sha(detail.read_bytes()), "captura_utc": "2000-01-01T03:01:00+00:00"}]})

    def plan(self, **kwargs):
        return recovered.build_plan(self.root, desde="2000-01-01", hasta="2000-01-02", **kwargs)

    def fake_success(self, command, **kwargs):
        self.make_batch("lote2", "2-2-LE26")
        snapshot = history.build_history(self.workspace)
        summary = json.loads((snapshot / "resumen.json").read_bytes())
        state = dict(self.state, lote="lote2", historial=snapshot.relative_to(self.workspace).as_posix(),
                     publicado_bigquery=True, fin_utc="2000-01-02T03:05:00+00:00",
                     publicacion={"historial": summary["huella_entradas"], "job_id": "job-sintetico"})
        self.json(self.workspace, checkpoint.STATE, state)
        return SimpleNamespace(returncode=0)

    def execute(self, result=None):
        with patch("sys.stdout", new_callable=io.StringIO):
            return recovered.execute_plan(self.plan(), runner=result or Mock(side_effect=self.fake_success))

    def test_plan_does_not_load_ticket_or_change_files(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        plan = self.plan()
        self.assertEqual(plan["before"]["conteos"]["capturas"], 1)
        self.ticket.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_show_plan_uses_recovered_cwd_and_never_enables_mutations(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0))
        with patch("sys.stdout", new_callable=io.StringIO):
            recovered.show_plan(self.plan(), runner=runner)
        command = runner.call_args.args[0]
        self.assertNotIn("--ejecutar", command)
        self.assertNotIn("--publicar-bigquery", command)
        self.assertEqual(runner.call_args.kwargs["cwd"], self.workspace)
        self.assertEqual(runner.call_args.kwargs["env"][portable.ENVIRONMENT_VARIABLE], str(self.model))
        self.ticket.assert_not_called()

    def test_success_confirms_publication_and_creates_new_local_checkpoint(self):
        runner = Mock(side_effect=self.fake_success)
        result = self.execute(runner)
        self.assertEqual(result["estado"], "completo")
        self.assertEqual(result["conteos"]["capturas"], 2)
        self.assertEqual(result["job_id"], "job-sintetico")
        self.assertFalse(result["checkpoint_publicado_gcs"])
        self.assertEqual(checkpoint.file_sha(result["checkpoint_local"]), result["sha256"])
        args, kwargs = runner.call_args.args[0], runner.call_args.kwargs
        self.assertIn("--ejecutar", args)
        self.assertIn("--publicar-bigquery", args)
        self.assertEqual(args[args.index("--max-segmentos") + 1], "1")
        self.assertEqual(args[args.index("--max-detalles") + 1], "5")
        self.assertEqual(kwargs["env"][recovered.TICKET], "TICKET_SINTETICO_NO_REGISTRAR")
        self.assertEqual(os.environ[recovered.TICKET], "")
        self.assertFalse((self.workspace / ".env").exists())
        reports = list((self.directory / "corridas").glob("*.json"))
        self.assertEqual(len(reports), 1)
        self.assertNotIn("TICKET_SINTETICO_NO_REGISTRAR", reports[0].read_text())

    def test_missing_ticket_aborts_before_starting_or_writing_a_run(self):
        self.ticket.return_value = None
        runner = Mock()
        with self.assertRaisesRegex(RuntimeError, "Falta MERCADO_PUBLICO_TICKET"):
            self.execute(runner)
        runner.assert_not_called()
        self.assertFalse((self.directory / "corridas").exists())

    def test_child_failure_is_preserved_and_blocks_blind_retry(self):
        with self.assertRaisesRegex(RuntimeError, "corrida no terminó"):
            self.execute(Mock(return_value=SimpleNamespace(returncode=1)))
        self.assertFalse((self.directory / "corrida_recuperada.lock").exists())
        with self.assertRaisesRegex(RuntimeError, "sin finalizar"):
            self.plan()

    def test_success_exit_with_unchanged_state_is_not_confirmed(self):
        with self.assertRaisesRegex(ValueError, "finalización nueva"):
            self.execute(Mock(return_value=SimpleNamespace(returncode=0)))

    def test_local_only_run_does_not_claim_bigquery_publication(self):
        def local_only(*args, **kwargs):
            self.json(self.workspace, checkpoint.STATE, dict(self.state, fin_utc="2000-01-02T03:05:00+00:00"))
            return SimpleNamespace(returncode=0)
        with self.assertRaisesRegex(ValueError, "finalización nueva"):
            self.execute(Mock(side_effect=local_only))

    def test_changed_workspace_code_is_rejected(self):
        self.write(self.workspace, "src/run_incremental.py", b"alterado")
        with self.assertRaisesRegex(ValueError, "Cambió el código"):
            self.plan()

    def test_missing_or_incomplete_bootstrap_is_rejected(self):
        self.json(self.directory, "bootstrap.json", dict(self.report, estado="fallido"))
        with self.assertRaisesRegex(ValueError, "no está preparado"):
            self.plan()

    def test_more_than_five_details_is_not_accepted(self):
        with self.assertRaises(ValueError):
            self.plan(limite=6)

    def test_workspace_cannot_be_redirected_to_main_project(self):
        self.json(self.directory, "bootstrap.json", dict(self.report, workspace=str(self.root)))
        with self.assertRaisesRegex(ValueError, "rutas del arranque"):
            self.plan()

    def test_existing_cycle_lock_is_preserved(self):
        path = self.directory / "corrida_recuperada.lock"
        path.write_bytes(b"CONSERVAR")
        with self.assertRaises(FileExistsError):
            with recovered.single_cycle(self.directory):
                self.fail("No entrar")
        self.assertEqual(path.read_bytes(), b"CONSERVAR")

    def test_environment_ticket_is_used_without_reading_env_file(self):
        with patch.dict(os.environ, {recovered.TICKET: "OTRO_TICKET_SINTETICO"}):
            result = recovered.child_environment(self.root, self.model, need_ticket=True)
        self.assertEqual(result[recovered.TICKET], "OTRO_TICKET_SINTETICO")
        self.ticket.assert_not_called()

    def test_checkpoint_failure_keeps_confirmed_job_in_a_failed_report(self):
        with patch.object(checkpoint, "create_checkpoint", side_effect=OSError("fallo sintetico")):
            with self.assertRaises(OSError):
                self.execute()
        record = json.loads(next((self.directory / "corridas").glob("*.json")).read_text())
        self.assertEqual(record["job_id"], "job-sintetico")
        self.assertEqual(record["estado"], "fallido")


if __name__ == "__main__":
    unittest.main()
