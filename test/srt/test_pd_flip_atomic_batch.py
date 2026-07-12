import ast
import copy
import os
import pathlib
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

if os.name == "nt" and "resource" not in sys.modules:
    resource_stub = types.ModuleType("resource")
    resource_stub.RLIMIT_NOFILE = 0
    resource_stub.RLIMIT_STACK = 1
    resource_stub.getrlimit = lambda _kind: (1024, 1024)
    resource_stub.setrlimit = lambda _kind, _limits: None
    sys.modules["resource"] = resource_stub

try:
    from sglang.srt.managers.io_struct import (
        PDFlipMigrationAbortReq,
        PDFlipMigrationOutputRelayReq,
        PDFlipMigrationSourceDeltaReq,
        PDFlipMigrationSourceFinishReq,
        PDFlipMigrationTargetCommitReq,
    )
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc
    for module_name in list(sys.modules):
        if module_name == "sglang" or module_name.startswith("sglang."):
            sys.modules.pop(module_name, None)

    class _Output:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    scheduler_tree = ast.parse(
        (
            pathlib.Path(__file__).resolve().parents[2]
            / "python/sglang/srt/managers/scheduler.py"
        ).read_text(encoding="utf-8")
    )
    scheduler_class = next(
        node
        for node in scheduler_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    fallback_method_names = {
        "commit_pd_flip_migration_target",
        "activate_pd_flip_migration_target",
        "start_pd_flip_migration_source_delta",
        "_pd_flip_request_batch_quiesce",
        "_pd_flip_maybe_enter_batch_quiesce",
        "_pd_flip_resume_batch_after_cutover",
    }
    method_nodes = [
        copy.deepcopy(node)
        for node in scheduler_class.body
        if isinstance(node, ast.FunctionDef) and node.name in fallback_method_names
    ]
    fallback_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *method_nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(fallback_module)
    fallback_namespace = {"PDFlipMigrationReqOutput": _Output}
    exec(compile(fallback_module, "<scheduler-ast>", "exec"), fallback_namespace)

    class Scheduler:
        pass

    for method_name in fallback_method_names:
        setattr(Scheduler, method_name, fallback_namespace[method_name])
    TokenizerManager = None
    PDFlipMigrationTargetCommitReq = types.SimpleNamespace
    PDFlipMigrationSourceDeltaReq = types.SimpleNamespace


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class FakeReq:
    def __init__(self, rid, *, fail_init=False):
        self.rid = rid
        self.to_finish = None
        self.req_pool_idx = None
        self.pd_flip_defer_kv_release = False
        self.pd_flip_force_kv_release = False
        self.pd_flip_kv_release_deferred = False
        self.pd_flip_deferred_kv_release_is_insert = False
        self.time_stats = types.SimpleNamespace(set_wait_queue_entry_time=lambda: None)
        self._fail_init = fail_init

    def finished(self):
        return self.to_finish is not None

    def init_next_round_input(self, tree_cache):
        if self._fail_init:
            raise RuntimeError("init failed")


def target_scheduler(phases, *, fail_init_rid=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = object()
    scheduler.waiting_queue = []
    scheduler.enable_decode_hicache = False
    scheduler._pd_flip_target_pump_transfer = lambda session: None
    scheduler._pd_flip_target_commit_hicache_restore = lambda decode_req: None
    scheduler._pd_flip_migration_status_dict = lambda: {}
    scheduler._pd_flip_note_timing = lambda container, name: None
    scheduler._pd_flip_release_target_request = lambda entry: entry.update(
        request_released=True
    )
    scheduler._pd_flip_free_target_metadata = lambda entry: None

    def abort_target(session, reason):
        for entry in session["target_entries"].values():
            entry["phase"] = "aborted"
            entry["held"] = False
        session["state"] = "target_aborted"
        session["last_error"] = reason

    scheduler._pd_flip_abort_target_session = abort_target
    entries = {}
    for rid, phase in phases.items():
        req = FakeReq(rid, fail_init=rid == fail_init_rid)
        entries[rid] = {
            "phase": phase,
            "held": phase == "transferred_held",
            "decode_req": types.SimpleNamespace(req=req, kv_receiver=None),
            "timing_debug": {},
        }
    scheduler.pd_flip_migration_session = {
        "session_id": "s",
        "role": "target",
        "state": "target_transferred_held",
        "target_entries": entries,
        "manifests": [{"rid": rid} for rid in entries],
        "held_reqs": len(entries),
        "timing_debug": {},
    }
    return scheduler


class TestAtomicTargetHandoff(unittest.TestCase):
    def test_target_commit_does_not_schedule_before_activate(self):
        scheduler = target_scheduler(
            {"r0": "transferred_held", "r1": "transferred_held"}
        )

        out = Scheduler.commit_pd_flip_migration_target(
            scheduler, PDFlipMigrationTargetCommitReq(session_id="s", rids=["r0", "r1"])
        )

        self.assertTrue(out.success)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(
            scheduler.pd_flip_migration_session["state"], "ready_to_activate"
        )
        self.assertEqual(
            [
                entry["phase"]
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            ],
            ["ready_to_activate", "ready_to_activate"],
        )

    def test_one_failed_entry_aborts_entire_batch(self):
        scheduler = target_scheduler({"r0": "transferred_held", "r1": "failed"})

        out = Scheduler.commit_pd_flip_migration_target(
            scheduler, PDFlipMigrationTargetCommitReq(session_id="s", rids=["r0", "r1"])
        )

        self.assertFalse(out.success)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertTrue(
            all(
                entry["phase"] == "aborted"
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            )
        )

    def test_commit_exception_aborts_without_partial_phase(self):
        scheduler = target_scheduler(
            {"r0": "transferred_held", "r1": "transferred_held"}
        )
        calls = 0

        def fail_second(_decode_req):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("commit failed")

        scheduler._pd_flip_target_commit_hicache_restore = fail_second

        out = Scheduler.commit_pd_flip_migration_target(
            scheduler, PDFlipMigrationTargetCommitReq(session_id="s", rids=["r0", "r1"])
        )

        self.assertFalse(out.success)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(
            {
                entry["phase"]
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            },
            {"aborted"},
        )

    def test_activate_is_atomic_and_skips_drop_entries(self):
        scheduler = target_scheduler(
            {"r0": "ready_to_activate", "r1": "ready_to_activate"}
        )
        scheduler.pd_flip_migration_session["target_entries"]["r1"][
            "drop_on_commit"
        ] = True

        out = Scheduler.activate_pd_flip_migration_target(
            scheduler,
            types.SimpleNamespace(session_id="s", rids=["r0", "r1"]),
        )

        self.assertTrue(out.success)
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["r0"])
        self.assertEqual(
            {
                entry["phase"]
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            },
            {"active"},
        )

    def test_activate_exception_keeps_queue_and_phases_unchanged(self):
        scheduler = target_scheduler(
            {"r0": "ready_to_activate", "r1": "ready_to_activate"},
            fail_init_rid="r1",
        )

        out = Scheduler.activate_pd_flip_migration_target(
            scheduler,
            types.SimpleNamespace(session_id="s", rids=["r0", "r1"]),
        )

        self.assertFalse(out.success)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(
            {
                entry["phase"]
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            },
            {"ready_to_activate"},
        )


class TestSourceQuiesce(unittest.TestCase):
    def test_delta_waits_for_whole_batch_quiesce_before_capturing_c1(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_migration_session = {
            "session_id": "s",
            "role": "source",
            "source_entries": {"r0": {"source_queue": "running"}},
        }
        scheduler._pd_flip_source_pump_transfer = lambda session: None
        scheduler._pd_flip_migration_status_dict = lambda: {}
        scheduler.pd_flip_batch_quiesced = False

        out = Scheduler.start_pd_flip_migration_source_delta(
            scheduler, PDFlipMigrationSourceDeltaReq(session_id="s", rids=["r0"])
        )

        self.assertFalse(out.success)
        self.assertTrue(scheduler.pd_flip_quiesce_requested)
        self.assertNotIn("delta_generation", scheduler.pd_flip_migration_session)

        scheduler.pd_flip_batch_quiesced = True
        scheduler._pd_flip_build_delta_manifest = lambda entry, generation: {
            "rid": "r0",
            "delta_noop": True,
        }
        scheduler._pd_flip_mark_source_delta_applied = lambda entry, manifest: None
        out = Scheduler.start_pd_flip_migration_source_delta(
            scheduler, PDFlipMigrationSourceDeltaReq(session_id="s", rids=["r0"])
        )
        self.assertTrue(out.success)
        self.assertEqual(out.manifests, [{"rid": "r0", "delta_noop": True}])
        generation = scheduler.pd_flip_migration_session["delta_generation"]

        repeated = Scheduler.start_pd_flip_migration_source_delta(
            scheduler, PDFlipMigrationSourceDeltaReq(session_id="s", rids=["r0"])
        )
        self.assertTrue(repeated.success)
        self.assertEqual(repeated.manifests, out.manifests)
        self.assertEqual(
            scheduler.pd_flip_migration_session["delta_generation"], generation
        )

    def test_pending_overlap_result_blocks_quiesce(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_quiesce_requested = True
        scheduler.pd_flip_batch_quiesced = False
        scheduler.result_queue = [object()]

        self.assertFalse(Scheduler._pd_flip_maybe_enter_batch_quiesce(scheduler))
        self.assertFalse(scheduler.pd_flip_batch_quiesced)

        scheduler.result_queue.clear()
        self.assertTrue(Scheduler._pd_flip_maybe_enter_batch_quiesce(scheduler))
        self.assertTrue(scheduler.pd_flip_batch_quiesced)


class TestDecodeLoopAST(unittest.TestCase):
    @staticmethod
    def _module(path):
        return ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))

    def test_activate_request_route_control_and_dispatch_are_connected(self):
        io_tree = self._module("python/sglang/srt/managers/io_struct.py")
        activate_cls = next(
            node
            for node in io_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PDFlipMigrationTargetActivateReq"
        )
        fields = {
            node.target.id
            for node in activate_cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertEqual(fields, {"session_id", "rids"})

        http_tree = self._module("python/sglang/srt/entrypoints/http_server.py")
        route = next(
            node
            for node in http_tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "activate_pd_flip_migration_target"
        )
        self.assertTrue(
            any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "post"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and decorator.args[0].value == "/pd_flip/migration/target/activate"
                for decorator in route.decorator_list
            )
        )

        control_tree = self._module(
            "python/sglang/srt/managers/tokenizer_control_mixin.py"
        )
        control = next(
            node
            for node in ast.walk(control_tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "activate_pd_flip_migration_target"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "pd_flip_migration_communicator"
                for node in ast.walk(control)
            )
        )

        scheduler_tree = self._module("python/sglang/srt/managers/scheduler.py")
        self.assertTrue(
            any(
                isinstance(node, ast.Tuple)
                and len(node.elts) == 2
                and isinstance(node.elts[0], ast.Name)
                and node.elts[0].id == "PDFlipMigrationTargetActivateReq"
                and isinstance(node.elts[1], ast.Attribute)
                and node.elts[1].attr == "activate_pd_flip_migration_target"
                for node in ast.walk(scheduler_tree)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef)
                and node.name == "activate_pd_flip_migration_target"
                for node in ast.walk(scheduler_tree)
            )
        )

    def test_decode_loops_check_quiesce_before_getting_new_batch(self):
        tree = self._module("python/sglang/srt/disaggregation/decode.py")
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in (
            "event_loop_normal_disagg_decode",
            "event_loop_overlap_disagg_decode",
        ):
            calls = [
                node
                for node in ast.walk(methods[name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]
            quiesce_lines = [
                node.lineno
                for node in calls
                if node.func.attr == "_pd_flip_maybe_enter_batch_quiesce"
            ]
            next_lines = [
                node.lineno
                for node in calls
                if node.func.attr == "get_next_disagg_decode_batch_to_run"
            ]
            self.assertTrue(quiesce_lines, name)
            self.assertLess(min(quiesce_lines), min(next_lines), name)
            if name == "event_loop_overlap_disagg_decode":
                result_lines = [
                    node.lineno
                    for node in calls
                    if node.func.attr == "process_batch_result"
                ]
                self.assertTrue(result_lines)
                self.assertLess(min(result_lines), min(quiesce_lines))


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, str(RUNTIME_IMPORT_ERROR))
class TestSourceFinishAndAbort(unittest.TestCase):
    def test_source_finish_filters_only_selected_and_resumes(self):
        scheduler = Scheduler.__new__(Scheduler)
        selected, other = FakeReq("selected"), FakeReq("other")

        class Batch:
            reqs = [selected, other]

            def filter_batch(self):
                self.reqs = [req for req in self.reqs if not req.finished()]

        scheduler.running_batch = Batch()
        scheduler.tree_cache = object()
        scheduler._pd_flip_source_pump_transfer = lambda session: None
        scheduler._pd_flip_free_source_metadata = lambda entry: None
        scheduler.pd_flip_migration_session = {
            "session_id": "s",
            "role": "source",
            "dry_run": False,
            "manifests": [{"rid": "selected"}],
            "transferred_rids": {"selected"},
            "source_entries": {
                "selected": {
                    "req": selected,
                    "source_queue": "running",
                    "committed_len": 0,
                    "timing_debug": {},
                }
            },
        }
        scheduler.pd_flip_quiesce_requested = True
        scheduler.pd_flip_batch_quiesced = True

        out = Scheduler.finish_pd_flip_migration_source(
            scheduler,
            PDFlipMigrationSourceFinishReq(session_id="s", released_rids=["selected"]),
        )

        self.assertTrue(out.success)
        self.assertEqual([req.rid for req in scheduler.running_batch.reqs], ["other"])
        self.assertFalse(scheduler.pd_flip_quiesce_requested)

    def test_source_abort_does_not_filter_running_batch_and_resumes(self):
        scheduler = Scheduler.__new__(Scheduler)
        selected, other = FakeReq("selected"), FakeReq("other")
        batch = types.SimpleNamespace(reqs=[selected, other])
        scheduler.running_batch = batch
        scheduler.pd_flip_quiesce_requested = True
        scheduler.pd_flip_batch_quiesced = True
        scheduler.pd_flip_migration_session = {
            "role": "source",
            "source_entries": {},
            "manifests": [],
            "timing_debug": {},
        }
        scheduler._pd_flip_restore_waiting_source_requests = lambda session: None

        Scheduler.abort_pd_flip_migration(
            scheduler, PDFlipMigrationAbortReq(session_id="s", reason="abort")
        )

        self.assertEqual(batch.reqs, [selected, other])
        self.assertFalse(scheduler.pd_flip_quiesce_requested)


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, str(RUNTIME_IMPORT_ERROR))
class TestMonotonicRelaySequence(unittest.IsolatedAsyncioTestCase):
    async def test_relay_drops_duplicate_and_older_output_sequences(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.rid_to_state = {"r0": object()}
        manager.pd_flip_last_relay_seq_by_rid = {}
        manager._pd_flip_batch_output_from_payload = lambda rid, payload: object()
        manager._handle_batch_output = AsyncMock()

        for seq in (1, 1, 0, 2):
            result = await TokenizerManager.relay_pd_flip_migration_output(
                manager,
                PDFlipMigrationOutputRelayReq(
                    rid="r0", output={"rid": "r0", "output_seq": seq}
                ),
            )
            self.assertTrue(result["success"])

        self.assertEqual(manager._handle_batch_output.await_count, 2)
        self.assertEqual(manager.pd_flip_last_relay_seq_by_rid["r0"], 2)

    async def test_target_increments_sequence_before_relay(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.pd_flip_output_relay_targets = {"r0": "http://source"}
        manager.pd_flip_output_seq_by_rid = {"r0": 7}
        manager._pd_flip_batch_output_payload = lambda output, index, rid: {"rid": rid}
        manager._pd_flip_post_relay_output = lambda source, rid, payload: {
            "success": True,
            "output_seq": payload["output_seq"],
        }
        output = types.SimpleNamespace(finished_reasons=[None])

        with patch(
            "asyncio.to_thread", new=AsyncMock(side_effect=lambda fn, *args: fn(*args))
        ):
            relayed = await TokenizerManager._pd_flip_maybe_relay_output(
                manager, output, 0, "r0"
            )

        self.assertTrue(relayed)
        self.assertEqual(manager.pd_flip_output_seq_by_rid["r0"], 8)


if __name__ == "__main__":
    unittest.main()
