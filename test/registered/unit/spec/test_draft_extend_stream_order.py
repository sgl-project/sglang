"""Check draft-extend stream dependencies without loading a model or CUDA."""

import ast
import contextlib
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SPEC = Path(__file__).resolve().parents[4] / "python/sglang/srt/speculative"


def preparation_prefix(filename):
    tree = ast.parse((SPEC / filename).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_draft_extend_for_decode"
    )
    # Execute through the reverse join, but not the model forward. Match the
    # join's receiver so a new producer wait cannot shorten the tested prefix.
    end = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "wait_stream"
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Attribute)
            and call.func.value.func.attr == "current_stream"
            for call in ast.walk(node)
        )
    )
    method.body = method.body[: end + 1]
    method.returns = None
    for arg in method.args.args + method.args.kwonlyargs:
        arg.annotation = None
    return compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        str(SPEC / filename),
        "exec",
    )


class TestDraftExtendStreamOrder(unittest.TestCase):
    def check_worker(self, filename, enabled, boundary=False, staged=False):
        events = []
        current = [None]
        test = self

        class Stream:
            def __init__(self, name):
                self.name = name
                self.dependencies = set()

            def wait_stream(self, producer):
                self.dependencies.add(producer)
                events.append((self.name, "wait", producer.name))

            def wait_event(self, event):
                self.dependencies.add(event.producer)
                events.append((self.name, "boundary"))

        class Event:
            def record(self):
                self.producer = current[0]
                test.assertIs(self.producer, compute)

        compute, plan = Stream("compute"), Stream("plan")
        current[0] = compute

        @contextlib.contextmanager
        def plan_context():
            current[0] = plan
            try:
                yield
            finally:
                current[0] = compute

        def prepare(*args, **kwargs):
            # Metadata preparation must not consume compute-produced inputs
            # unless its stream is ordered after that producer.
            if enabled:
                test.assertIs(current[0], plan)
                test.assertIn(compute, plan.dependencies)
            else:
                test.assertIs(current[0], compute)
                test.assertEqual(events, [])
            events.append(("prepare",))
            return MagicMock()

        batch_result, batch, worker = MagicMock(), MagicMock(), MagicMock()
        batch.seq_lens = [1, 2]
        worker.device = "cuda"
        worker.speculative_num_draft_tokens = 5
        worker.speculative_num_steps = 4
        worker.draft_extend_num_front_tokens = 1 if boundary else 0
        worker.plan_stream = plan if enabled else None
        worker.plan_stream_ctx = plan_context() if enabled else contextlib.nullcontext()
        worker._compute_boundary_kv_locs_positions.return_value = (
            (MagicMock(), MagicMock()) if boundary else (None, None)
        )
        torch = MagicMock()
        device = torch.get_device_module.return_value
        device.current_stream.side_effect = lambda: current[0]
        device.Event.side_effect = Event
        namespace = dict(
            torch=torch,
            EagleDraftExtendInput=MagicMock(),
            prepare_for_draft_extend=prepare,
        )
        exec(preparation_prefix(filename), namespace)
        kwargs = {"staged": staged} if filename.startswith("multi_layer") else {}
        namespace["_draft_extend_for_decode"](worker, batch, batch_result, **kwargs)
        if enabled:
            dependency = (
                ("plan", "boundary") if boundary else ("plan", "wait", "compute")
            )
            self.assertEqual(
                events, [dependency, ("prepare",), ("compute", "wait", "plan")]
            )
        else:
            self.assertEqual(events, [("prepare",)])
            device.current_stream.assert_not_called()

    def test_single_layer_plan(self):
        self.check_worker("eagle_worker_v2.py", True)

    def test_single_layer_no_plan(self):
        self.check_worker("eagle_worker_v2.py", False)

    def test_multi_layer_without_boundary_event(self):
        self.check_worker("multi_layer_eagle_worker_v2.py", True)

    def test_multi_layer_boundary_event(self):
        self.check_worker("multi_layer_eagle_worker_v2.py", True, boundary=True)

    def test_multi_layer_staged_boundary(self):
        self.check_worker(
            "multi_layer_eagle_worker_v2.py", True, boundary=True, staged=True
        )

    def test_multi_layer_staged_without_boundary(self):
        self.check_worker("multi_layer_eagle_worker_v2.py", True, staged=True)

    def test_multi_layer_no_plan(self):
        self.check_worker("multi_layer_eagle_worker_v2.py", False)


if __name__ == "__main__":
    unittest.main()
