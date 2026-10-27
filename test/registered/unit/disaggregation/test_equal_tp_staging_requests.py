import ast
import pathlib
import unittest
from types import SimpleNamespace as NS

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/srt/disaggregation"


def method(file, name):
    tree = ast.parse((ROOT / file).read_text())
    matches = [
        x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name
    ]
    assert matches, f"Missing staging behavior: {name}"
    n = matches[0]
    n.decorator_list = []
    n.returns = None
    for a in n.args.args:
        a.annotation = None
    env = {"envs": NS(SGLANG_DISAGG_STAGING_ALWAYS=NS(get=lambda: False))}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])),
            str(file),
            "exec",
        ),
        env,
    )
    return env[name]


class ResponseWakeup(CustomTestCase):
    def test_response_updates_and_notifies_under_waiter_lock(self):
        tree = ast.parse((ROOT / "mooncake/conn.py").read_text())
        branch = next(
            x
            for x in ast.walk(tree)
            if isinstance(x, ast.If) and ast.unparse(x.test) == "room == 'STAGING_RSP'"
        )
        body = [
            x for x in branch.body if not isinstance(x, (ast.Continue, ast.ImportFrom))
        ]
        calls = []

        class CV:
            def __enter__(self):
                calls.append("lock")

            def __exit__(self, *a):
                calls.append("unlock")

            def notify_all(self):
                calls.append("notify")

        env = {
            "self": NS(_staging_ctx=NS(watermark_cv=CV()), transfer_infos={}),
            "waiting_req_bytes": [],
            "handle_staging_rsp": lambda *a: calls.append("update"),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                "response",
                "exec",
            ),
            env,
        )
        self.assertEqual(calls, ["lock", "update", "notify", "unlock"])


class ReviewRegression(CustomTestCase):
    def test_send_failure_rolls_back_and_unknown_error_propagates(self):
        import threading

        tree = ast.parse((ROOT / "mooncake/conn.py").read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_ensure_equal_tp_staging_req"
        )
        node.body = [
            n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]

        class Again(Exception):
            pass

        class Address:
            is_ipv6 = False

            def __init__(self, *args):
                pass

            def to_tcp(self):
                return "tcp://test:1"

        env = dict(
            staging_grid_tokens=lambda c, p: c,
            NetworkAddress=Address,
            zmq=NS(Again=Again),
            logger=NS(warning=lambda *args: None),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "actual_method",
                "exec",
            ),
            env,
        )
        f = env[node.name]
        ctx = NS(
            prefetch_requested=set(), request_lock=threading.Lock(), prefetch_sockets={}
        )
        mgr = NS(
            kv_buffer_tensors={"page_size": 1},
            server_args=NS(chunked_prefill_size=16),
            pp_rank=0,
            _staging_ctx=ctx,
        )
        chunk = NS(room=7, index_slice=slice(0, 8), prefill_kv_indices=list(range(8)))
        req = NS(endpoint="test", dst_port=1, mooncake_session_id="s")
        calls = []

        def send(*args, **kwargs):
            calls.append(args)
            raise Again()

        mgr._send_multipart_locked = send
        self.assertFalse(f(mgr, chunk, req))
        self.assertEqual(ctx.prefetch_requested, set())
        self.assertEqual(len(calls), 1)

        def unknown(*args, **kwargs):
            raise RuntimeError("unknown")

        mgr._send_multipart_locked = unknown
        with self.assertRaises(RuntimeError):
            f(mgr, chunk, req)
        self.assertEqual(ctx.prefetch_requested, set())
        mgr._send_multipart_locked = lambda *args, **kwargs: calls.append(args)
        self.assertTrue(f(mgr, chunk, req))
        self.assertTrue(f(mgr, chunk, req))
        self.assertEqual(len(calls), 2)


class TestConcurrentAllocationRequests(CustomTestCase):
    def test_duplicate_requests_submit_once(self):
        import threading
        import time

        node = next(
            n
            for n in ast.walk(ast.parse((ROOT / "mooncake/conn.py").read_text()))
            if isinstance(n, ast.FunctionDef)
            and n.name == "_ensure_equal_tp_staging_req"
        )
        node.body = [
            n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]

        class Address:
            is_ipv6 = False

            def __init__(self, *args):
                pass

            def to_tcp(self):
                return "tcp://test:1"

        env = dict(
            staging_grid_tokens=lambda c, p: c,
            NetworkAddress=Address,
            zmq=NS(Again=TimeoutError),
            logger=NS(warning=lambda *args: None),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "allocation_request",
                "exec",
            ),
            env,
        )
        calls = []

        def send(*args, **kwargs):
            time.sleep(0.005)
            calls.append(args)

        manager = NS(
            kv_buffer_tensors={"page_size": 1},
            server_args=NS(chunked_prefill_size=16),
            pp_rank=0,
            _staging_ctx=NS(prefetch_requested=set(), request_lock=threading.Lock()),
            _send_multipart_locked=send,
        )
        chunk = NS(room=7, index_slice=slice(0, 8), prefill_kv_indices=list(range(8)))
        req = NS(endpoint="test", dst_port=1, mooncake_session_id="s")
        results = []

        def run():
            results.append(env[node.name](manager, chunk, req))

        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(results, [True] * 8)
        self.assertEqual(len(calls), 1)


class TestReadyBeforeWait(CustomTestCase):
    def test_response_between_check_and_wait_is_observed(self):
        import copy
        import time

        node = next(
            n
            for n in ast.walk(ast.parse((ROOT / "mooncake/conn.py").read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_do_staging_transfer"
        )

        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

        node = RemoveImports().visit(copy.deepcopy(node))
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        state = {"ready": False}

        class CV:
            def __enter__(self):
                state["ready"] = True

            def __exit__(self, *args):
                pass

            def wait(self, timeout):
                raise AssertionError("waited despite ready response")

        env = dict(
            time=time,
            STAGING_READY_SPIN_S=0.5,
            STAGING_MAX_WAIT_S=60,
            StagingAllocator=NS(ALLOC_OVERSIZED=-2),
            logger=NS(warning_once=lambda *args: None),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "staging_wait",
                "exec",
            ),
            env,
        )
        strategy = NS(
            check_ready=lambda *args: (state["ready"], 0, 0, 0, 0),
            transfer=lambda *args: 0,
        )
        notifications = []
        manager = NS(
            _staging_ctx=NS(watermark_cv=CV()),
            send_chunk_ready=lambda *args, **kwargs: notifications.append(kwargs),
        )
        result = env[node.name](
            manager,
            strategy,
            NS(index_slice=slice(0, 8), prefill_kv_indices=list(range(8))),
            NS(room=7, endpoint="test", dst_port=1, mooncake_session_id="s"),
            NS(staging_base_ptr=1, staging_total_size=100),
            None,
            None,
            None,
        )
        self.assertEqual(result, (0, False))
        self.assertEqual(len(notifications), 1)


if __name__ == "__main__":
    unittest.main()
