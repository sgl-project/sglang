import ast
import copy
import enum
import pathlib
import types
import unittest


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

    def test_all_pd_loops_guard_role_at_iteration_top_and_clear_active_role(self):
        loop_specs = (
            (
                PREFILL_PATH,
                "SchedulerDisaggregationPrefillMixin",
                "event_loop_normal_disagg_prefill",
                "prefill",
                False,
            ),
            (
                PREFILL_PATH,
                "SchedulerDisaggregationPrefillMixin",
                "event_loop_overlap_disagg_prefill",
                "prefill",
                True,
            ),
            (
                DECODE_PATH,
                "SchedulerDisaggregationDecodeMixin",
                "event_loop_normal_disagg_decode",
                "decode",
                False,
            ),
            (
                DECODE_PATH,
                "SchedulerDisaggregationDecodeMixin",
                "event_loop_overlap_disagg_decode",
                "decode",
                True,
            ),
            (
                PP_PATH,
                "SchedulerPPMixin",
                "event_loop_pp_disagg_prefill",
                "prefill",
                False,
            ),
            (
                PP_PATH,
                "SchedulerPPMixin",
                "event_loop_pp_disagg_decode",
                "decode",
                False,
            ),
        )

        for path, class_name, method_name, role, overlap in loop_specs:
            with self.subTest(method=method_name):
                method = _find_method(path, class_name, method_name)
                active_value = _attribute_assignment_value(
                    method.body, "active_pd_event_loop_role"
                )
                self.assertIsInstance(active_value, ast.Constant)
                self.assertEqual(active_value.value, role)

                loop = next(
                    statement
                    for statement in method.body
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
                active_clear = _attribute_assignment_value(
                    guard.body, "active_pd_event_loop_role"
                )
                self.assertIsInstance(active_clear, ast.Constant)
                self.assertIsNone(active_clear.value)
                self.assertTrue(
                    any(isinstance(statement, ast.Return) for statement in guard.body)
                )
                if overlap:
                    self.assertTrue(
                        any(
                            isinstance(node, ast.Attribute)
                            and node.attr == "result_queue"
                            for node in ast.walk(guard)
                        )
                    )

    def test_controller_waits_for_configured_and_active_roles(self):
        wait_for_role = _load_method(
            CONTROLLER_PATH,
            "PDFlipController",
            "_wait_source_role",
            {
                "_first_successful_response": lambda response: response,
                "_normalize_role": lambda role: str(role or "").lower(),
                "_parse_runtime_status": lambda item: (
                    item["status"]["role"],
                    bool(item["status"].get("is_idle")),
                    False,
                ),
                "time": types.SimpleNamespace(
                    monotonic=lambda: 0.0, sleep=lambda _: None
                ),
            },
        )
        responses = iter(
            [
                {
                    "status": {
                        "role": "prefill",
                        "active_event_loop_role": "decode",
                    }
                },
                {
                    "status": {
                        "role": "prefill",
                        "active_event_loop_role": "prefill",
                    }
                },
            ]
        )
        calls = []
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(
                migration_timeout_seconds=1.0,
                migration_poll_interval_seconds=0.0,
            ),
            _record_get=lambda *args: (calls.append(args), next(responses))[1],
        )
        source = types.SimpleNamespace(name="source", worker_url="http://source")

        response = wait_for_role(
            controller, [], source, "prefill", "wait_source_prefill_loop"
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(response["status"]["active_event_loop_role"], "prefill")


if __name__ == "__main__":
    unittest.main()
