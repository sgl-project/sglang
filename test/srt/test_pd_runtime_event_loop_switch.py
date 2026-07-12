import ast
import copy
import enum
import pathlib
import types
import unittest
from collections import deque


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
PREFILL_PATH = REPO_ROOT / "python/sglang/srt/disaggregation/prefill.py"
DECODE_PATH = REPO_ROOT / "python/sglang/srt/disaggregation/decode.py"
PP_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler_pp_mixin.py"
CONTROLLER_PATH = REPO_ROOT / "scripts/playground/disaggregation/pd_flip_controller.py"


class DisaggregationMode(enum.Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def _find_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _load_method(path, class_name, method_name, namespace=None):
    method = copy.deepcopy(_find_method(path, class_name, method_name))
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def _load_top_level_function(path, function_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[function_name]


def _attribute_assignment_value(statements, attribute):
    for statement in statements:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if any(
            isinstance(target, ast.Attribute) and target.attr == attribute
            for target in targets
        ):
            return statement.value
    return None


class TestPDRuntimeEventLoopSwitch(unittest.TestCase):
    def test_dispatch_loop_redispatches_after_decode_loop_changes_role(self):
        calls = []

        def dispatch_event_loop(scheduler):
            if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                calls.append("decode")
                scheduler.disaggregation_mode = DisaggregationMode.PREFILL
            else:
                calls.append("prefill")
                scheduler._shutdown_requested = True

        run_dispatch = _load_method(
            SCHEDULER_PATH,
            "Scheduler",
            "_run_pd_dispatch_loop",
            {"dispatch_event_loop": dispatch_event_loop},
        )
        scheduler = types.SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            _shutdown_requested=False,
        )

        run_dispatch(scheduler)

        self.assertEqual(calls, ["decode", "prefill"])

    def test_non_dynamic_run_event_loop_dispatches_only_once(self):
        calls = []

        class Stream:
            def synchronize(self):
                pass

        class StreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self.stream

            def __exit__(self, *_):
                return False

        device_module = types.SimpleNamespace(
            Stream=lambda priority=0: Stream(), StreamContext=StreamContext
        )
        envs = types.SimpleNamespace(
            SGLANG_ENABLE_WAR_BARRIER=types.SimpleNamespace(get=lambda: False)
        )
        run_event_loop = _load_method(
            SCHEDULER_PATH,
            "Scheduler",
            "run_event_loop",
            {
                "dispatch_event_loop": lambda scheduler: calls.append("dispatch"),
                "envs": envs,
                "is_cuda": lambda: False,
                "use_mlx": lambda: False,
            },
        )
        scheduler = types.SimpleNamespace(
            device_module=device_module,
            device="cpu",
            spec_algorithm=types.SimpleNamespace(is_dflash=lambda: False),
            pd_runtime_role_switch_enabled=lambda: False,
            _run_pd_dispatch_loop=lambda: self.fail("dynamic loop was entered"),
        )

        run_event_loop(scheduler)

        self.assertEqual(calls, ["dispatch"])

    def test_runtime_status_exposes_active_event_loop_role(self):
        status_method = _load_method(
            SCHEDULER_PATH, "Scheduler", "_pd_runtime_role_status_dict"
        )
        scheduler = types.SimpleNamespace(
            ps=types.SimpleNamespace(dp_rank=0, tp_rank=0),
            active_pd_event_loop_role="decode",
            is_fully_idle=lambda: True,
            pd_runtime_role=lambda: "prefill",
            pd_runtime_role_switch_enabled=lambda: True,
            pd_flip_should_reject_new_work=lambda: False,
            _pd_flip_capacity_status=lambda: {},
        )

        status = status_method(scheduler)

        self.assertEqual(status["active_event_loop_role"], "decode")

    def test_all_pd_loop_wrappers_clear_active_role_and_impls_guard_at_top(self):
        loop_specs = (
            (
                PREFILL_PATH,
                "SchedulerDisaggregationPrefillMixin",
                "event_loop_normal_disagg_prefill",
                "_event_loop_normal_disagg_prefill_impl",
                "prefill",
            ),
            (
                PREFILL_PATH,
                "SchedulerDisaggregationPrefillMixin",
                "event_loop_overlap_disagg_prefill",
                "_event_loop_overlap_disagg_prefill_impl",
                "prefill",
            ),
            (
                DECODE_PATH,
                "SchedulerDisaggregationDecodeMixin",
                "event_loop_normal_disagg_decode",
                "_event_loop_normal_disagg_decode_impl",
                "decode",
            ),
            (
                DECODE_PATH,
                "SchedulerDisaggregationDecodeMixin",
                "event_loop_overlap_disagg_decode",
                "_event_loop_overlap_disagg_decode_impl",
                "decode",
            ),
            (
                PP_PATH,
                "SchedulerPPMixin",
                "event_loop_pp_disagg_prefill",
                "_event_loop_pp_disagg_prefill_impl",
                "prefill",
            ),
            (
                PP_PATH,
                "SchedulerPPMixin",
                "event_loop_pp_disagg_decode",
                "_event_loop_pp_disagg_decode_impl",
                "decode",
            ),
        )

        for path, class_name, wrapper_name, impl_name, role in loop_specs:
            with self.subTest(method=wrapper_name):
                wrapper = _find_method(path, class_name, wrapper_name)
                active_value = _attribute_assignment_value(
                    wrapper.body, "active_pd_event_loop_role"
                )
                self.assertIsInstance(active_value, ast.Constant)
                self.assertEqual(active_value.value, role)
                role_try = next(
                    statement
                    for statement in wrapper.body
                    if isinstance(statement, ast.Try)
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.Attribute) and node.attr == impl_name
                        for node in ast.walk(role_try)
                    )
                )
                clear_guard = next(
                    statement
                    for statement in role_try.finalbody
                    if isinstance(statement, ast.If)
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.Attribute)
                        and node.attr == "active_pd_event_loop_role"
                        for node in ast.walk(clear_guard.test)
                    )
                )
                clear_value = _attribute_assignment_value(
                    clear_guard.body, "active_pd_event_loop_role"
                )
                self.assertIsInstance(clear_value, ast.Constant)
                self.assertIsNone(clear_value.value)

                impl = _find_method(path, class_name, impl_name)
                loop = next(
                    statement
                    for statement in impl.body
                    if isinstance(statement, ast.While)
                )
                guard = loop.body[0]
                self.assertIsInstance(guard, ast.If)
                self.assertTrue(
                    any(
                        isinstance(node, ast.Attribute)
                        and node.attr == "_pd_role_loop_should_exit"
                        for node in ast.walk(guard.test)
                    )
                )
                self.assertTrue(
                    any(isinstance(statement, ast.Return) for statement in guard.body)
                )

    def test_pd_loop_wrappers_clear_active_role_after_impl_exception(self):
        loop_specs = (
            (
                PREFILL_PATH,
                "SchedulerDisaggregationPrefillMixin",
                "event_loop_normal_disagg_prefill",
                "_event_loop_normal_disagg_prefill_impl",
            ),
            (
                PREFILL_PATH,
                "SchedulerDisaggregationPrefillMixin",
                "event_loop_overlap_disagg_prefill",
                "_event_loop_overlap_disagg_prefill_impl",
            ),
            (
                DECODE_PATH,
                "SchedulerDisaggregationDecodeMixin",
                "event_loop_normal_disagg_decode",
                "_event_loop_normal_disagg_decode_impl",
            ),
            (
                DECODE_PATH,
                "SchedulerDisaggregationDecodeMixin",
                "event_loop_overlap_disagg_decode",
                "_event_loop_overlap_disagg_decode_impl",
            ),
            (
                PP_PATH,
                "SchedulerPPMixin",
                "event_loop_pp_disagg_prefill",
                "_event_loop_pp_disagg_prefill_impl",
            ),
            (
                PP_PATH,
                "SchedulerPPMixin",
                "event_loop_pp_disagg_decode",
                "_event_loop_pp_disagg_decode_impl",
            ),
        )

        for path, class_name, wrapper_name, impl_name in loop_specs:
            with self.subTest(method=wrapper_name):
                wrapper = _load_method(path, class_name, wrapper_name)

                def fail_impl():
                    raise RuntimeError("loop failed")

                scheduler = types.SimpleNamespace(**{impl_name: fail_impl})
                with self.assertRaisesRegex(RuntimeError, "loop failed"):
                    wrapper(scheduler)
                self.assertIsNone(scheduler.active_pd_event_loop_role)

    def test_pd_loop_wrapper_does_not_clear_a_new_active_role(self):
        wrapper = _load_method(
            PREFILL_PATH,
            "SchedulerDisaggregationPrefillMixin",
            "event_loop_normal_disagg_prefill",
        )
        scheduler = types.SimpleNamespace()
        scheduler._event_loop_normal_disagg_prefill_impl = lambda: setattr(
            scheduler, "active_pd_event_loop_role", "decode"
        )

        wrapper(scheduler)

        self.assertEqual(scheduler.active_pd_event_loop_role, "decode")

    def _assert_overlap_drains_after_role_change(
        self, path, class_name, impl_name, role
    ):
        calls = []
        guard_results = iter((False, True, True))

        class Batch:
            def copy(self):
                return "batch-copy"

        scheduler = types.SimpleNamespace(
            result_queue=deque(),
            last_batch=None,
            cur_batch=None,
            enable_staging=False,
            _war_barrier_enabled=False,
            _engine_paused=False,
            pd_flip_quiesce_requested=False,
            request_receiver=types.SimpleNamespace(
                recv_requests=lambda: (calls.append("recv"), ["request"])[1]
            ),
            process_input_requests=lambda requests: calls.append("input"),
            process_decode_queue=lambda: calls.append("decode-queue"),
            _pd_flip_maybe_enter_batch_quiesce=lambda: False,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                pop_bootstrapped=lambda: []
            ),
            waiting_queue=[],
            _pd_role_loop_should_exit=lambda expected: next(guard_results),
            get_next_disagg_prefill_batch_to_run=lambda: (
                calls.append("get-batch"),
                Batch(),
            )[1],
            get_next_disagg_decode_batch_to_run=lambda: (
                calls.append("get-batch"),
                Batch(),
            )[1],
            run_batch=lambda batch: (calls.append("run-batch"), "result")[1],
            process_batch_result=lambda batch, result: calls.append(
                ("process-result", batch, result)
            ),
            process_disagg_prefill_inflight_queue=lambda: calls.append(
                "prefill-inflight"
            ),
            launch_batch_sample_if_needed=lambda result: calls.append("sample"),
            on_idle=lambda: calls.append("idle"),
            schedule_stream=types.SimpleNamespace(
                wait_stream=lambda stream: calls.append("wait-stream")
            ),
            forward_stream=object(),
        )
        namespace = {"DisaggregationMode": DisaggregationMode, "deque": deque}
        if role == "prefill":
            namespace["envs"] = types.SimpleNamespace(
                SGLANG_DISAGG_STAGING_BUFFER=types.SimpleNamespace(get=lambda: False)
            )
        impl = _load_method(path, class_name, impl_name, namespace)
        wrapper_name = impl_name.removeprefix("_").removesuffix("_impl")
        wrapper = _load_method(path, class_name, wrapper_name)
        setattr(scheduler, impl_name, lambda: impl(scheduler))

        wrapper(scheduler)

        self.assertEqual(calls.count("recv"), 1)
        self.assertEqual(calls.count("get-batch"), 1)
        self.assertEqual(calls.count("run-batch"), 1)
        self.assertEqual(
            [call for call in calls if isinstance(call, tuple)],
            [("process-result", "batch-copy", "result")],
        )
        self.assertEqual(list(scheduler.result_queue), [])
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.cur_batch)
        self.assertIsNone(scheduler.active_pd_event_loop_role)

    def test_prefill_overlap_drains_pending_result_without_launching_new_batch(self):
        self._assert_overlap_drains_after_role_change(
            PREFILL_PATH,
            "SchedulerDisaggregationPrefillMixin",
            "_event_loop_overlap_disagg_prefill_impl",
            "prefill",
        )

    def test_decode_overlap_drains_pending_result_without_launching_new_batch(self):
        self._assert_overlap_drains_after_role_change(
            DECODE_PATH,
            "SchedulerDisaggregationDecodeMixin",
            "_event_loop_overlap_disagg_decode_impl",
            "decode",
        )

    def _load_wait_for_role(self):
        clock = types.SimpleNamespace(value=0.0)

        def monotonic():
            clock.value += 0.1
            return clock.value

        return _load_method(
            CONTROLLER_PATH,
            "PDFlipController",
            "_wait_source_role",
            {
                "_first_successful_response": _load_top_level_function(
                    CONTROLLER_PATH, "_first_successful_response"
                ),
                "_normalize_role": lambda role: str(role or "").lower(),
                "_parse_runtime_status": lambda item: (
                    item["status"]["role"],
                    bool(item["status"].get("is_idle")),
                    False,
                ),
                "time": types.SimpleNamespace(
                    monotonic=monotonic, sleep=lambda _: None
                ),
            },
        )

    def _run_wait_for_role(self, responses):
        wait_for_role = self._load_wait_for_role()
        responses = iter(responses)
        calls = []
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(
                migration_timeout_seconds=2.0,
                migration_poll_interval_seconds=0.0,
            ),
            _record_get=lambda *args: (calls.append(args), next(responses))[1],
        )
        source = types.SimpleNamespace(name="source", worker_url="http://source")
        response = wait_for_role(
            controller, [], source, "prefill", "wait_source_prefill_loop"
        )
        return response, calls

    def test_controller_waits_for_every_dp_rank_to_be_active(self):
        ready = {
            "success": True,
            "status": {"role": "prefill", "active_event_loop_role": "prefill"},
        }
        slow = {
            "success": True,
            "status": {"role": "prefill", "active_event_loop_role": "decode"},
        }
        response, calls = self._run_wait_for_role([[ready, slow], [ready, dict(ready)]])

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(response), 2)

    def test_controller_retries_failed_and_incomplete_fanout_status(self):
        failed = {"success": False, "message": "rank unavailable"}
        missing_active = {"success": True, "status": {"role": "prefill"}}
        ready = {
            "success": True,
            "status": {"role": "prefill", "active_event_loop_role": "prefill"},
        }
        response, calls = self._run_wait_for_role(
            [[ready, failed], [ready, missing_active], [ready, ready]]
        )

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(item["success"] for item in response))

    def test_controller_retries_empty_fanout_status(self):
        ready = {
            "success": True,
            "status": {"role": "prefill", "active_event_loop_role": "prefill"},
        }
        response, calls = self._run_wait_for_role([[], [ready, ready]])

        self.assertEqual(len(calls), 2)
        self.assertEqual(response, [ready, ready])

    def test_controller_retries_record_get_failure(self):
        wait_for_role = self._load_wait_for_role()
        ready = {
            "success": True,
            "status": {"role": "prefill", "active_event_loop_role": "prefill"},
        }
        attempts = iter((RuntimeError("fanout failed"), [ready, ready]))
        calls = []

        def record_get(*args):
            calls.append(args)
            outcome = next(attempts)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(
                migration_timeout_seconds=2.0,
                migration_poll_interval_seconds=0.0,
            ),
            _record_get=record_get,
        )
        source = types.SimpleNamespace(name="source", worker_url="http://source")

        response = wait_for_role(
            controller, [], source, "prefill", "wait_source_prefill_loop"
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(response, [ready, ready])

    def test_controller_invalid_fanout_status_retries_until_timeout(self):
        wait_for_role = self._load_wait_for_role()
        ready = {
            "success": True,
            "status": {"role": "prefill", "active_event_loop_role": "prefill"},
        }
        calls = []
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(
                migration_timeout_seconds=0.5,
                migration_poll_interval_seconds=0.0,
            ),
            _record_get=lambda *args: (
                calls.append(args),
                [ready, "invalid-rank-status"],
            )[1],
        )
        source = types.SimpleNamespace(name="source", worker_url="http://source")

        with self.assertRaisesRegex(TimeoutError, "wait_source_prefill_loop"):
            wait_for_role(controller, [], source, "prefill", "wait_source_prefill_loop")

        self.assertGreater(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
