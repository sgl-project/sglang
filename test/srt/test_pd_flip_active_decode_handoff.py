import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestPDFlipActiveDecodeHandoff(unittest.TestCase):
    def test_target_prepare_declares_adopt_on_success_contract(self):
        io_struct = (REPO_ROOT / "python/sglang/srt/managers/io_struct.py").read_text()

        self.assertIn("class PDFlipMigrationTargetPrepareReq", io_struct)
        self.assertIn("adopt_on_success: bool = False", io_struct)

    def test_scheduler_adopts_target_request_after_success(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()

        self.assertIn('"adopt_on_success": recv_req.adopt_on_success', scheduler)
        self.assertIn('session.get("adopt_on_success", False)', scheduler)
        self.assertIn("def _pd_flip_adopt_target_request", scheduler)
        self.assertIn("req.init_next_round_input(self.tree_cache)", scheduler)
        self.assertIn("self.waiting_queue.append(req)", scheduler)
        self.assertIn('entry["request_adopted"] = True', scheduler)

    def test_migration_manifest_preserves_output_routing_fields(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()

        for field in [
            '"http_worker_ipc": getattr(req, "http_worker_ipc", None)',
            '"stream": bool(getattr(req, "stream", False))',
            '"logprob_start_len": getattr(req, "logprob_start_len", -1)',
            '"time_stats": self._pd_flip_serialize_time_stats(',
            "http_worker_ipc=manifest.get(\"http_worker_ipc\")",
            "stream=bool(manifest.get(\"stream\", False))",
            "time_stats=self._pd_flip_deserialize_time_stats(",
            'req.logprob_start_len = int(manifest.get("logprob_start_len", -1))',
        ]:
            self.assertIn(field, scheduler)

    def test_source_release_uses_internal_migration_finish_reason(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()

        self.assertIn(
            "Request migrated during PD role flip; output continues on target decode.",
            scheduler,
        )
        self.assertNotIn(
            'FINISH_ABORT(\n                    "Request migrated during PD role flip.",\n                    HTTPStatus.SERVICE_UNAVAILABLE',
            scheduler,
        )
        self.assertIn("pd_flip_migrated_to_target", scheduler)


if __name__ == "__main__":
    unittest.main()
