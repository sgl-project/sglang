"""CPU-only checks of the actual staged-send return boundary, without CUDA imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/disaggregation/mooncake/conn.py"
)


class ReturnContractTest(CustomTestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "send_kvcache_staged"
        )
        start = next(
            i
            for i, n in enumerate(method.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "_transfer_data"
        )
        # Execute the unmodified post-gather boundary from the implementation.
        method.body = method.body[start:]
        method.args = ast.parse("def boundary(self): pass").body[0].args
        method.returns = None
        namespace = dict(
            mooncake_session_id="test",
            transfer_blocks=[],
            local_bytes=1,
            dst_staging_ptr=0,
            rank_offset=0,
            staging_buffer=SimpleNamespace(get_ptr=lambda: 0),
            logger=SimpleNamespace(error=lambda *args: None),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
        self.boundary = namespace["send_kvcache_staged"]

    def test_return_codes_are_preserved(self):
        for code in (0, -1, -5, 1):
            with self.subTest(code=code):
                self.assertEqual(
                    self.boundary(SimpleNamespace(_transfer_data=lambda *args: code)),
                    code,
                )

    def test_unknown_exception_is_not_swallowed(self):
        failure = RuntimeError("unknown transport state")

        def transfer(*args):
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            self.boundary(SimpleNamespace(_transfer_data=transfer))
        self.assertIs(caught.exception, failure)


class TestAllocationFailureIsolation(CustomTestCase):
    def test_room_failure_does_not_blacklist_session(self):
        import threading
        from collections import defaultdict

        tree = ast.parse(SOURCE.read_text())
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "ret != 0"
            and "session_failures" in ast.unparse(n)
            and "kv_chunk.room" in ast.unparse(n)
        )
        body = [n for n in branch.body if not isinstance(n, ast.Break)]

        class Address:
            def __init__(self, *args):
                pass

            def to_host_port_str(self):
                return "test:1"

        sentinel = object()
        for ret, failed_session in ((sentinel, False), (-1, True), (-5, True)):
            with self.subTest(ret=ret):
                conclusions = []
                manager = SimpleNamespace(
                    session_lock=threading.Lock(),
                    session_failures=defaultdict(int),
                    failed_sessions=set(),
                    _poison_unsettled_send=lambda *args: None,
                    conclude_failure=lambda **kwargs: conclusions.append(kwargs),
                )
                env = dict(
                    self=manager,
                    ret=ret,
                    _STAGING_REQUEST_FAILED=sentinel,
                    req=SimpleNamespace(
                        mooncake_session_id="s", endpoint="test", dst_port=1, room=7
                    ),
                    kv_chunk=SimpleNamespace(room=7),
                    KVPoll=SimpleNamespace(Failed=0),
                    NetworkAddress=Address,
                    prefill_unique_rank=0,
                    logger=SimpleNamespace(error=lambda *args: None),
                )
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=body, type_ignores=[])
                        ),
                        "worker_failure",
                        "exec",
                    ),
                    env,
                )
                self.assertEqual("s" in manager.failed_sessions, failed_session)
                self.assertEqual(manager.session_failures["s"], int(failed_session))
                # The room itself is always concluded as failed exactly once.
                self.assertEqual(len(conclusions), 1)

    def test_oversized_allocation_has_distinct_failure_result(self):
        from types import SimpleNamespace as NS

        node = next(
            n
            for n in ast.walk(ast.parse(SOURCE.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_do_staging_transfer"
        )
        # Remove the CUDA allocator import; retain the actual readiness logic.
        node = __import__("copy").deepcopy(node)

        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self, n):
                return None

        node = RemoveImports().visit(node)
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        sentinel = object()
        env = dict(
            _STAGING_REQUEST_FAILED=sentinel,
            StagingAllocator=NS(ALLOC_OVERSIZED=-2),
            logger=NS(warning_once=lambda *args: None),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "staging_allocation",
                "exec",
            ),
            env,
        )
        result = env[node.name](
            NS(),
            NS(check_ready=lambda *args: (False, 0, -2, 0, -1)),
            NS(index_slice=slice(0, 8), prefill_kv_indices=list(range(8))),
            NS(),
            None,
            None,
            None,
            None,
        )
        self.assertIs(result[0], sentinel)
        self.assertFalse(result[1])


if __name__ == "__main__":
    unittest.main()
