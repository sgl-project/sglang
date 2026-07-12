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
        self.init_calls = 0

    def finished(self):
        return self.to_finish is not None

    def init_next_round_input(self, tree_cache):
        self.init_calls += 1
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

    def test_explicit_empty_commit_batch_is_rejected(self):
        scheduler = target_scheduler(
            {"r0": "transferred_held", "r1": "transferred_held"}
        )

        out = Scheduler.commit_pd_flip_migration_target(
            scheduler, PDFlipMigrationTargetCommitReq(session_id="s", rids=[])
        )

        self.assertFalse(out.success)
        self.assertEqual(scheduler.waiting_queue, [])

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

    def test_activate_defers_request_initialization_to_decode_admission(self):
        scheduler = target_scheduler(
            {"r0": "ready_to_activate", "r1": "ready_to_activate"},
            fail_init_rid="r1",
        )

        out = Scheduler.activate_pd_flip_migration_target(
            scheduler,
            types.SimpleNamespace(session_id="s", rids=["r0", "r1"]),
        )

        self.assertTrue(out.success)
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["r0", "r1"])
        self.assertEqual(
            [
                entry["decode_req"].req.init_calls
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            ],
            [0, 0],
        )
        self.assertEqual(
            {
                entry["phase"]
                for entry in scheduler.pd_flip_migration_session[
                    "target_entries"
                ].values()
            },
            {"active"},
        )

    def test_explicit_empty_activate_batch_is_rejected(self):
        scheduler = target_scheduler(
            {"r0": "ready_to_activate", "r1": "ready_to_activate"}
        )

        out = Scheduler.activate_pd_flip_migration_target(
            scheduler, types.SimpleNamespace(session_id="s", rids=[])
        )

        self.assertFalse(out.success)
        self.assertEqual(scheduler.waiting_queue, [])


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

    def test_quiesce_request_freezes_session_and_rids(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_migration_session = {
            "session_id": "s",
            "role": "source",
            "source_entries": {
                "r0": {"source_queue": "running"},
                "r1": {"source_queue": "running"},
            },
        }
        scheduler._pd_flip_source_pump_transfer = lambda session: None
        scheduler._pd_flip_migration_status_dict = lambda: {}
        scheduler.pd_flip_batch_quiesced = False

        first = Scheduler.start_pd_flip_migration_source_delta(
            scheduler, PDFlipMigrationSourceDeltaReq(session_id="s", rids=["r0"])
        )
        different = Scheduler.start_pd_flip_migration_source_delta(
            scheduler, PDFlipMigrationSourceDeltaReq(session_id="s", rids=["r1"])
        )

        self.assertFalse(first.success)
        self.assertFalse(different.success)
        self.assertEqual(scheduler.pd_flip_quiesce_rids, ("r0",))
        self.assertEqual(scheduler.pd_flip_quiesce_session_id, "s")

    def test_explicit_empty_delta_batch_is_rejected(self):
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
            scheduler, PDFlipMigrationSourceDeltaReq(session_id="s", rids=[])
        )

        self.assertFalse(out.success)
        self.assertFalse(getattr(scheduler, "pd_flip_quiesce_requested", False))

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
            wrapper_impl_calls = [
                node.func.attr
                for node in ast.walk(methods[name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.endswith("_impl")
            ]
            self.assertEqual(len(wrapper_impl_calls), 1, name)
            impl_name = wrapper_impl_calls[0]
            self.assertIn(impl_name, methods, name)
            calls = [
                node
                for node in ast.walk(methods[impl_name])
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
            paused_lines = [
                node.lineno
                for node in ast.walk(methods[impl_name])
                if isinstance(node, ast.Attribute) and node.attr == "_engine_paused"
            ]
            self.assertTrue(paused_lines)
            self.assertLess(min(quiesce_lines), min(paused_lines), name)
            if name == "event_loop_overlap_disagg_decode":
                result_lines = [
                    node.lineno
                    for node in calls
                    if node.func.attr == "process_batch_result"
                ]
                self.assertTrue(result_lines)
                self.assertLess(min(result_lines), min(quiesce_lines))

    def test_output_sequence_and_session_cross_every_output_boundary(self):
        io_tree = self._module("python/sglang/srt/managers/io_struct.py")
        output_classes = {
            node.name: node
            for node in io_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"BatchTokenIDOutput", "BatchStrOutput"}
        }
        for name in ("BatchTokenIDOutput", "BatchStrOutput"):
            fields = {
                node.target.id
                for node in output_classes[name].body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            }
            self.assertIn("pd_flip_output_seqs", fields, name)
            self.assertIn("pd_flip_session_ids", fields, name)

        relay_cls = next(
            node
            for node in io_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PDFlipMigrationOutputRelayReq"
        )
        relay_fields = {
            node.target.id
            for node in relay_cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertTrue({"session_id", "output_seq"} <= relay_fields)

        streamer = self._module(
            "python/sglang/srt/managers/scheduler_components/output_streamer.py"
        )
        accept = next(
            node
            for node in ast.walk(streamer)
            if isinstance(node, ast.FunctionDef) and node.name == "accept"
        )
        assigned_attrs = {
            node.attr for node in ast.walk(accept) if isinstance(node, ast.Attribute)
        }
        self.assertIn("pd_flip_last_emitted_output_seq", assigned_attrs)

        detokenizer = self._module("python/sglang/srt/managers/detokenizer_manager.py")
        handle = next(
            node
            for node in ast.walk(detokenizer)
            if isinstance(node, ast.FunctionDef)
            and node.name == "handle_batch_token_id_out"
        )
        copied_keywords = {
            keyword.arg
            for node in ast.walk(handle)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        }
        self.assertIn("pd_flip_output_seqs", copied_keywords)
        self.assertIn("pd_flip_session_ids", copied_keywords)

    def test_manifest_target_and_relay_share_session_sequence_baseline(self):
        scheduler = self._module("python/sglang/srt/managers/scheduler.py")
        methods = {
            node.name: node
            for node in ast.walk(scheduler)
            if isinstance(node, ast.FunctionDef)
        }
        manifest_constants = {
            node.value
            for node in ast.walk(methods["_pd_flip_build_migration_manifest"])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("last_emitted_output_seq", manifest_constants)
        self.assertIn("pd_flip_session_id", manifest_constants)

        for method_name in (
            "_pd_flip_manifest_to_req",
            "_pd_flip_apply_delta_manifest_to_target",
        ):
            assigned = {
                node.attr
                for node in ast.walk(methods[method_name])
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
            }
            self.assertIn("pd_flip_last_emitted_output_seq", assigned, method_name)
            self.assertIn("pd_flip_migration_session_id", assigned, method_name)

        control = self._module("python/sglang/srt/managers/tokenizer_control_mixin.py")
        prepare = next(
            node
            for node in ast.walk(control)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "prepare_pd_flip_migration_target"
        )
        prepare_names = {
            node.id for node in ast.walk(prepare) if isinstance(node, ast.Name)
        }
        self.assertIn("relay_key", prepare_names)
        prepare_attrs = {
            node.attr for node in ast.walk(prepare) if isinstance(node, ast.Attribute)
        }
        self.assertIn("pd_flip_output_relay_baseline", prepare_attrs)

        manager = self._module("python/sglang/srt/managers/tokenizer_manager.py")
        handle = next(
            node
            for node in ast.walk(manager)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_handle_batch_output"
        )
        calls = [
            node
            for node in ast.walk(handle)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        drop_lines = [
            node.lineno
            for node in calls
            if node.func.attr == "_pd_flip_drop_or_record_output_seq"
        ]
        state_lookup_lines = [
            node.lineno
            for node in calls
            if node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "rid_to_state"
        ]
        self.assertLess(min(drop_lines), min(state_lookup_lines))


class TestOmittedSessionCleanup(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _compile_methods(path, class_name, method_names):
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        selected = [
            copy.deepcopy(node)
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in method_names
        ]
        if {node.name for node in selected} != set(method_names):
            raise AssertionError("requested production control methods are missing")
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *selected,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, "<control-ast>", "exec"), namespace)
        return namespace

    def _manager(self, response_session="s1"):
        control_names = {
            "_pd_flip_resolve_effective_session_id",
            "abort_pd_flip_migration_target",
            "finish_pd_flip_migration_source",
            "abort_pd_flip_migration",
        }
        methods = self._compile_methods(
            "python/sglang/srt/managers/tokenizer_control_mixin.py",
            "TokenizerControlMixin",
            control_names,
        )
        cleanup = self._compile_methods(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            {"_pd_flip_clear_session_relay_state"},
        )

        class Manager:
            pass

        for name, method in {**methods, **cleanup}.items():
            setattr(Manager, name, method)
        manager = Manager()
        manager.auto_create_handle_loop = lambda: None
        manager.pd_flip_migration_communicator = AsyncMock(
            return_value=[
                types.SimpleNamespace(
                    success=True, status={"session_id": response_session}
                )
            ]
        )
        manager.pd_flip_output_relay_targets = {
            ("s1", "r0"): "u1",
            ("s2", "r0"): "u2",
        }
        manager.pd_flip_output_relay_baseline = {
            ("s1", "r0"): 3,
            ("s2", "r0"): 4,
        }
        manager.pd_flip_last_relay_seq_by_key = {
            ("s1", "r0"): 3,
            ("s2", "r0"): 4,
        }
        manager.pd_flip_relay_session_by_rid = {"r0": "s1", "r2": "s2"}
        manager.rid_to_state = {}
        return manager

    async def test_target_abort_uses_resolved_session_without_touching_other_session(
        self,
    ):
        manager = self._manager()

        await manager.abort_pd_flip_migration_target(
            types.SimpleNamespace(session_id=None)
        )

        self.assertNotIn(("s1", "r0"), manager.pd_flip_output_relay_targets)
        self.assertIn(("s2", "r0"), manager.pd_flip_output_relay_targets)

    async def test_generic_abort_cleans_resolved_binding_and_last_seen_only(self):
        manager = self._manager()

        await manager.abort_pd_flip_migration(types.SimpleNamespace(session_id=None))

        self.assertNotIn(("s1", "r0"), manager.pd_flip_last_relay_seq_by_key)
        self.assertIn(("s2", "r0"), manager.pd_flip_last_relay_seq_by_key)
        self.assertNotEqual(manager.pd_flip_relay_session_by_rid.get("r0"), "s1")
        self.assertEqual(manager.pd_flip_relay_session_by_rid["r2"], "s2")

    async def test_source_finish_cleans_resolved_inactive_binding_only(self):
        manager = self._manager()

        await manager.finish_pd_flip_migration_source(
            types.SimpleNamespace(session_id=None)
        )

        self.assertNotIn(("s1", "r0"), manager.pd_flip_last_relay_seq_by_key)
        self.assertIn(("s2", "r0"), manager.pd_flip_last_relay_seq_by_key)

    async def test_ambiguous_response_sessions_do_not_trigger_cleanup(self):
        manager = self._manager()
        manager.pd_flip_migration_communicator = AsyncMock(
            return_value=[
                types.SimpleNamespace(success=True, status={"session_id": "s1"}),
                types.SimpleNamespace(success=True, status={"session_id": "s2"}),
            ]
        )

        await manager.abort_pd_flip_migration(types.SimpleNamespace(session_id=None))

        self.assertIn(("s1", "r0"), manager.pd_flip_last_relay_seq_by_key)
        self.assertIn(("s2", "r0"), manager.pd_flip_last_relay_seq_by_key)

    async def test_missing_response_session_does_not_trigger_cleanup(self):
        manager = self._manager()
        manager.pd_flip_migration_communicator = AsyncMock(return_value=[])

        await manager.abort_pd_flip_migration(types.SimpleNamespace(session_id=None))

        self.assertIn(("s1", "r0"), manager.pd_flip_last_relay_seq_by_key)
        self.assertIn(("s2", "r0"), manager.pd_flip_last_relay_seq_by_key)


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
    def test_relay_post_requires_and_sends_admin_bearer(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = types.SimpleNamespace(admin_api_key="secret")
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return b'{"success": true}'

        def urlopen(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            return Response()

        with patch("urllib.request.urlopen", side_effect=urlopen):
            result = manager._pd_flip_post_relay_output(
                "http://source", "s", "r0", 1, {"rid": "r0"}
            )

        self.assertTrue(result["success"])
        self.assertEqual(captured["authorization"], "Bearer secret")

        manager.server_args.admin_api_key = None
        with patch("urllib.request.urlopen") as send:
            result = manager._pd_flip_post_relay_output(
                "http://source", "s", "r0", 1, {"rid": "r0"}
            )
        self.assertFalse(result["success"])
        send.assert_not_called()

    async def test_relay_drops_duplicate_and_older_output_sequences(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.pd_flip_last_relay_seq_by_key = {("s", "r0"): 0}

        dropped = []
        for seq in (1, 1, 0, 2):
            output = types.SimpleNamespace(
                pd_flip_session_ids=["s"], pd_flip_output_seqs=[seq]
            )
            dropped.append(
                TokenizerManager._pd_flip_drop_or_record_output_seq(
                    manager, output, 0, "r0"
                )
            )

        self.assertEqual(dropped, [False, True, True, False])
        self.assertEqual(manager.pd_flip_last_relay_seq_by_key[("s", "r0")], 2)
        manager.pd_flip_last_relay_seq_by_key[("s2", "r0")] = 0
        self.assertFalse(
            TokenizerManager._pd_flip_drop_or_record_output_seq(
                manager,
                types.SimpleNamespace(
                    pd_flip_session_ids=["s2"], pd_flip_output_seqs=[1]
                ),
                0,
                "r0",
            )
        )

    async def test_target_relays_scheduler_sequence_without_reincrementing(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.pd_flip_output_relay_targets = {("s", "r0"): "http://source"}
        manager.pd_flip_output_relay_baseline = {("s", "r0"): 7}
        manager._pd_flip_batch_output_payload = lambda output, index, rid: {"rid": rid}
        captured = {}

        def post(source, session_id, rid, output_seq, payload):
            captured.update(
                source=source,
                session_id=session_id,
                rid=rid,
                output_seq=output_seq,
                payload=payload,
            )
            return {"success": True}

        manager._pd_flip_post_relay_output = post
        output = types.SimpleNamespace(
            finished_reasons=[None],
            pd_flip_session_ids=["s"],
            pd_flip_output_seqs=[8],
        )

        with patch(
            "asyncio.to_thread", new=AsyncMock(side_effect=lambda fn, *args: fn(*args))
        ):
            relayed = await TokenizerManager._pd_flip_maybe_relay_output(
                manager, output, 0, "r0"
            )

        self.assertTrue(relayed)
        self.assertEqual(captured["output_seq"], 8)
        self.assertEqual(manager.pd_flip_output_relay_baseline[("s", "r0")], 8)

    async def test_failed_relay_stays_in_ordered_outbox_until_ack(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.pd_flip_output_relay_targets = {("s", "r0"): "http://source"}
        manager.pd_flip_output_relay_baseline = {("s", "r0"): 7}
        manager.pd_flip_output_relay_outbox = {}
        manager.pd_flip_output_relay_retry_tasks = {}
        manager._pd_flip_batch_output_payload = lambda output, index, rid: {"rid": rid}
        results = iter(({"success": False}, {"success": True}))
        manager._pd_flip_post_relay_output = lambda *args: next(results)
        manager._pd_flip_ensure_relay_retry = lambda key: None
        output = types.SimpleNamespace(
            finished_reasons=["stop"],
            pd_flip_session_ids=["s"],
            pd_flip_output_seqs=[8],
        )

        with patch(
            "asyncio.to_thread", new=AsyncMock(side_effect=lambda fn, *args: fn(*args))
        ):
            self.assertTrue(
                await manager._pd_flip_maybe_relay_output(output, 0, "r0")
            )
            self.assertEqual(manager.pd_flip_output_relay_baseline[("s", "r0")], 7)
            self.assertIn(8, manager.pd_flip_output_relay_outbox[("s", "r0")])
            await manager._pd_flip_flush_relay_key(("s", "r0"))

        self.assertNotIn(("s", "r0"), manager.pd_flip_output_relay_outbox)
        self.assertNotIn(("s", "r0"), manager.pd_flip_output_relay_targets)


if __name__ == "__main__":
    unittest.main()
