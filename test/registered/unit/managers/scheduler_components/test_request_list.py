import ast
import inspect
import re
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.models.unlimited_ocr_v2_support.document_prefetch import DocPrefetch
from sglang.srt.models.unlimited_ocr_v2_support.scheduler_runtime import (
    UocrSchedulerRuntime,
)
from sglang.test.scheduler_test_utils import FakeReq

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# The classes that own the request lists, with their mixins.
_OWNERS = (
    Scheduler,
    GrammarManager,
    PrefillBootstrapQueue,
    DecodePreallocQueue,
    DecodeTransferQueue,
    DllmManager,
    HiSparseCoordinator,
    UocrSchedulerRuntime,
    DocPrefetch,
    DeferredResponses,
)

# Fields annotated as holding requests that are not a `ReqList`: each holds
# requests a list already holds.
_NOT_LISTS = {
    # Shares its entries with `queue`.
    "DecodePreallocQueue.pending_reqs",
    # A subset of `waiting_queue`.
    "DllmManager.staging_queue",
}

_HOLDS_REQUESTS = re.compile(r"\b(Req|DecodeRequest)\b")


def _fields_holding_requests(owner) -> set:
    fields = set()
    for klass in owner.__mro__:
        if not klass.__module__.startswith("sglang."):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
                and _HOLDS_REQUESTS.search(ast.unparse(node.annotation))
            ):
                fields.add(f"{owner.__name__}.{node.target.attr}")
    return fields


class TestRequestHoldersAreLists(CustomTestCase):
    def test_every_field_holding_requests_is_a_request_list(self):
        fields = set().union(*map(_fields_holding_requests, _OWNERS))
        self.assertEqual(fields, _NOT_LISTS)


class TestKeyedReqList(CustomTestCase):
    def test_join_keeps_order_and_leave_ignores_outsiders(self):
        a, b, c, outsider = (FakeReq(rid=rid, finished=lambda: False) for rid in "abcx")
        running = KeyedReqList(ListKind.DECODING)
        running.join([a, b])
        running.join([c, b])
        self.assertEqual(list(running), [a, b, c])

        b.finished = lambda: True
        running.leave_finished()
        running.leave([a, outsider])

        self.assertEqual(list(running), [c])
        self.assertNotIn(a, running)


def _reqs(rids):
    return [FakeReq(rid=rid) for rid in rids]


class TestMove(CustomTestCase):
    def test_move_wraps_into_a_wrapper_list_and_keeps_its_index(self):
        a, b, c, d = _reqs("abcd")
        src = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        src.join([a, b, c])
        dst = IndexedReqList(ListKind.PD_RELEASING, req_of=lambda x: x[0])
        held = (d, "held")
        dst.join([held])

        wrapped = [(a, "a"), (b, "b")]
        move(wrapped, dst)

        self.assertEqual(list(src), [c])
        self.assertEqual(list(dst.items()), [held, *wrapped])
        self.assertEqual(list(dst), [d, a, b])

    def test_a_request_is_in_at_most_one_list(self):
        a, b = _reqs("ab")
        held = IndexedReqList(ListKind.PD_BOOTSTRAP)
        held.join([a])
        dst = KeyedReqList(ListKind.DECODING)

        with self.assertRaisesRegex(
            RuntimeError, "a is in PD_BOOTSTRAP, cannot join DECODING"
        ):
            dst.join([b, a])
        with self.assertRaisesRegex(RuntimeError, "a is already in PD_BOOTSTRAP"):
            move([a], held)

        self.assertEqual(list(held), [a])
        self.assertEqual(list(dst), [])
        self.assertIs(a.owner_list, held)
        self.assertIsNone(b.owner_list)

    def test_every_way_out_of_a_list_clears_the_owner(self):
        a, b, c, d = _reqs("abcd")
        queue = IndexedReqList(ListKind.WAITING)
        queue.join([a, b])
        running = KeyedReqList(ListKind.DECODING)
        running.join([c])

        queue.leave([a])
        running.clear()
        queue.replace([d])

        self.assertEqual([r.owner_list for r in (a, b, c)], [None, None, None])
        self.assertIs(d.owner_list, queue)


if __name__ == "__main__":
    unittest.main()
