import ast
import pathlib
import types
import typing
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_scheduler_method(name):
    path = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    namespace = {"Dict": typing.Dict, "Any": typing.Any, "Req": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestPDFlipActiveDecodeHandoff(unittest.TestCase):
    def test_target_prepare_declares_adopt_on_success_contract(self):
        io_struct = (REPO_ROOT / "python/sglang/srt/managers/io_struct.py").read_text()

        self.assertIn("class PDFlipMigrationTargetPrepareReq", io_struct)
        self.assertIn("adopt_on_success: bool = False", io_struct)
        self.assertIn("prepare_only: bool = False", io_struct)
        self.assertIn("adopt_on_commit: bool = True", io_struct)
        self.assertIn("class PDFlipMigrationTargetCommitReq", io_struct)
        self.assertIn("class PDFlipMigrationTargetAbortReq", io_struct)

    def test_scheduler_adopts_target_request_after_success(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
        decode = (REPO_ROOT / "python/sglang/srt/disaggregation/decode.py").read_text()

        self.assertIn('"adopt_on_success": recv_req.adopt_on_success', scheduler)
        self.assertIn('session.get("adopt_on_success", False)', scheduler)
        self.assertIn("def _pd_flip_adopt_target_request", scheduler)
        self.assertIn("req.pd_flip_prebuilt_kv_ready = True", scheduler)
        self.assertIn(
            "self._pd_flip_prepare_target_request_for_adoption(req)", scheduler
        )
        self.assertIn("pd_flip_prebuilt_kv_ready", decode)
        self.assertIn("self.waiting_queue.append(req)", scheduler)
        self.assertIn('entry["request_adopted"] = True', scheduler)

    def test_pd_flip_prebuilt_handoff_does_not_rematch_radix_prefix(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
        adopt_start = scheduler.index("def _pd_flip_adopt_target_request")
        adopt_end = scheduler.index("def _pd_flip_build_migration_manifest", adopt_start)
        adopt_body = scheduler[adopt_start:adopt_end]

        self.assertNotIn("req.init_next_round_input(self.tree_cache)", adopt_body)
        self.assertNotIn("req.init_next_round_input(", adopt_body)

        decode = (REPO_ROOT / "python/sglang/srt/disaggregation/decode.py").read_text()
        batch_start = decode.index("def get_new_prebuilt_batch")
        batch_end = decode.index("def process_decode_queue", batch_start)
        batch_body = decode[batch_start:batch_end]
        self.assertIn("pd_flip_prebuilt_kv_ready", batch_body)

        activate_start = scheduler.index("def activate_pd_flip_migration_target")
        activate_end = scheduler.index("def abort_pd_flip_migration_target", activate_start)
        activate_body = scheduler[activate_start:activate_end]
        self.assertIn(
            "self._pd_flip_prepare_target_request_for_adoption(req)",
            activate_body,
        )

    def test_adopt_preserves_received_kv_ownership_until_prebuilt_admission(self):
        method = _load_scheduler_method("_pd_flip_adopt_target_request")
        prepare = _load_scheduler_method(
            "_pd_flip_prepare_target_request_for_adoption"
        )

        class ReceivedReq:
            def __init__(self):
                self.cache_protected_len = 0
                self.prefix_indices = []
                self.last_node = None
                self.time_stats = None

            def init_next_round_input(self, _tree_cache):
                raise AssertionError("adopt must not rematch the radix tree")

        req = ReceivedReq()
        entry = {"decode_req": types.SimpleNamespace(req=req)}
        root_node = object()
        scheduler = types.SimpleNamespace(
            waiting_queue=[],
            tree_cache=types.SimpleNamespace(root_node=root_node),
            _pd_flip_note_timing=lambda *_args: None,
        )
        scheduler._pd_flip_prepare_target_request_for_adoption = (
            lambda target_req: prepare(scheduler, target_req)
        )

        method(scheduler, entry)

        self.assertIs(scheduler.waiting_queue[0], req)
        self.assertTrue(req.pd_flip_prebuilt_kv_ready)
        self.assertEqual(req.cache_protected_len, 0)
        self.assertEqual(req.prefix_indices, [])
        self.assertIs(req.last_node, root_node)

    def test_scheduler_declares_two_phase_target_commit_and_abort(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
        tokenizer_control = (
            REPO_ROOT / "python/sglang/srt/managers/tokenizer_control_mixin.py"
        ).read_text()
        http_server = (
            REPO_ROOT / "python/sglang/srt/entrypoints/http_server.py"
        ).read_text()

        self.assertIn('"prepare_only": recv_req.prepare_only', scheduler)
        self.assertIn('"adopt_on_commit": recv_req.adopt_on_commit', scheduler)
        self.assertIn('"transferred_held"', scheduler)
        self.assertIn("def commit_pd_flip_migration_target", scheduler)
        self.assertIn("def abort_pd_flip_migration_target", scheduler)
        self.assertIn("def _pd_flip_abort_target_session", scheduler)
        self.assertIn('"final_owner" in entries[rid]', scheduler)
        self.assertIn('entry.pop("final_owner", None)', scheduler)
        self.assertIn("PDFlipMigrationTargetCommitReq", tokenizer_control)
        self.assertIn("PDFlipMigrationTargetAbortReq", tokenizer_control)
        self.assertIn("/pd_flip/migration/target/commit", http_server)
        self.assertIn("/pd_flip/migration/target/abort", http_server)

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
        release_start = scheduler.index("def _pd_flip_release_source_requests")
        release_end = scheduler.index(
            "def _pd_flip_prepare_target_entries", release_start
        )
        release_body = scheduler[release_start:release_end]

        self.assertNotIn("FINISH_ABORT", release_body)
        self.assertNotIn(
            'FINISH_ABORT(\n                    "Request migrated during PD role flip.",\n                    HTTPStatus.SERVICE_UNAVAILABLE',
            scheduler,
        )
        self.assertIn("pd_flip_migrated_to_target", release_body)
        self.assertIn("pd_flip_waiting_for_relay_output", release_body)

    def test_pd_flip_output_relay_contract_is_declared(self):
        io_struct = (REPO_ROOT / "python/sglang/srt/managers/io_struct.py").read_text()
        tokenizer_control = (
            REPO_ROOT / "python/sglang/srt/managers/tokenizer_control_mixin.py"
        ).read_text()
        tokenizer_manager = (
            REPO_ROOT / "python/sglang/srt/managers/tokenizer_manager.py"
        ).read_text()
        http_server = (
            REPO_ROOT / "python/sglang/srt/entrypoints/http_server.py"
        ).read_text()

        self.assertIn("PDFlipMigrationOutputRelayReq", io_struct)
        self.assertIn("pd_flip_output_relay_targets", tokenizer_control)
        self.assertIn("pd_flip_output_relay_targets", tokenizer_manager)
        self.assertIn("relay_pd_flip_migration_output", tokenizer_manager)
        self.assertIn("/pd_flip/migration/output/relay", http_server)

    def test_pd_flip_delta_migration_contract_is_declared(self):
        io_struct = (REPO_ROOT / "python/sglang/srt/managers/io_struct.py").read_text()
        tokenizer_control = (
            REPO_ROOT / "python/sglang/srt/managers/tokenizer_control_mixin.py"
        ).read_text()
        http_server = (
            REPO_ROOT / "python/sglang/srt/entrypoints/http_server.py"
        ).read_text()
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
        controller = (
            REPO_ROOT / "scripts/playground/disaggregation/pd_flip_controller.py"
        ).read_text()

        self.assertIn("class PDFlipMigrationSourceDeltaReq", io_struct)
        self.assertIn("class PDFlipMigrationTargetDeltaPrepareReq", io_struct)
        self.assertIn("start_pd_flip_migration_source_delta", tokenizer_control)
        self.assertIn("prepare_pd_flip_migration_target_delta", tokenizer_control)
        self.assertIn("/pd_flip/migration/source/delta", http_server)
        self.assertIn("/pd_flip/migration/target/delta/prepare", http_server)
        self.assertIn("def start_pd_flip_migration_source_delta", scheduler)
        self.assertIn("def prepare_pd_flip_migration_target_delta", scheduler)
        self.assertIn("_sync_two_phase_delta_before_commit", controller)


if __name__ == "__main__":
    unittest.main()
