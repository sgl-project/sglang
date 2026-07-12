import ast
import pathlib
import textwrap
import types
import unittest
from typing import Any, Dict


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"


def load_scheduler_method(name):
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        (
            node
            for node in scheduler.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )
    if method is None:
        raise AttributeError(f"Scheduler has no method {name}")
    namespace = {"Any": Any, "Dict": Dict, "logger": types.SimpleNamespace()}
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    return namespace[name]


class FakeReq:
    def __init__(self, rid, kv_committed_len, *, finished=False):
        self.rid = rid
        self.kv_committed_len = kv_committed_len
        self._finished = finished

    def finished(self):
        return self._finished


def make_scheduler(*, running, request_slots=99, max_running_requests=8):
    scheduler = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=running),
        max_running_requests=max_running_requests,
        req_to_token_pool=types.SimpleNamespace(available_size=lambda: request_slots),
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 1000),
        server_args=types.SimpleNamespace(num_reserved_decode_tokens=16),
    )
    scheduler._pd_flip_capacity_status = types.MethodType(
        load_scheduler_method("_pd_flip_capacity_status"), scheduler
    )
    return scheduler


class TestPDFlipCapacityStatus(unittest.TestCase):
    def test_reports_capacity_and_unfinished_requests_in_running_order(self):
        scheduler = make_scheduler(
            running=[
                FakeReq(17, "100"),
                FakeReq("done", 175, finished=True),
                FakeReq("r1", 250),
            ],
            request_slots=99,
        )

        out = scheduler._pd_flip_capacity_status()

        self.assertEqual(out["free_request_slots"], 5)
        self.assertEqual(out["available_kv_tokens"], 1000)
        self.assertEqual(out["max_running_requests_per_dp"], 8)
        self.assertEqual(out["reserved_decode_tokens_per_req"], 16)
        self.assertEqual(
            out["running_requests"],
            [
                {"rid": "17", "kv_committed_len": 100},
                {"rid": "r1", "kv_committed_len": 250},
            ],
        )

    def test_free_slots_obey_scheduler_limit_and_never_go_negative(self):
        scheduler = make_scheduler(
            running=[FakeReq(str(i), i) for i in range(6)],
            request_slots=99,
            max_running_requests=8,
        )
        self.assertEqual(scheduler._pd_flip_capacity_status()["free_request_slots"], 2)

        scheduler.running_batch.reqs.extend(
            [FakeReq("8", 8), FakeReq("9", 9), FakeReq("10", 10)]
        )
        self.assertEqual(scheduler._pd_flip_capacity_status()["free_request_slots"], 0)

    def test_runtime_role_status_includes_capacity_fields(self):
        method = load_scheduler_method("_pd_runtime_role_status_dict")
        scheduler = make_scheduler(running=[], request_slots=3)
        scheduler.ps = types.SimpleNamespace(dp_rank=0, tp_rank=0)
        scheduler.is_fully_idle = lambda: True
        scheduler.pd_runtime_role = lambda: "decode"
        scheduler.pd_runtime_role_switch_enabled = lambda: True
        scheduler.pd_flip_should_reject_new_work = lambda: False

        out = method(scheduler)

        self.assertEqual(out["free_request_slots"], 3)
        self.assertEqual(out["available_kv_tokens"], 1000)
        self.assertEqual(out["running_requests"], [])


if __name__ == "__main__":
    unittest.main()
