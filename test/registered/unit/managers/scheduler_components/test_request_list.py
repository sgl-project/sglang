import ast
import collections
import inspect
import pathlib
import re
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt
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
    PD_KINDS,
    IndexedReqList,
    KeyedReqList,
    ListKind,
    assert_empty,
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
        self.assertEqual(fields, set())


_SRT = pathlib.Path(sglang.srt.__path__[0])
_EDITS = ("_join", "_leave", "_leave_finished", "_drop_all")
# `move`, and the retirer module, which takes ending requests out.
_EDITORS = {
    "managers/scheduler_components/request_list.py",
    "managers/scheduler_components/request_retirer.py",
}
# The other code that edits a list directly, with its count: each takes a
# request out before or without its release, or edits a list the retirer does
# not reach.
_DIRECT_EDITS = {
    "disaggregation/decode.py:DecodeTransferQueue.resolve_deferred_releases": 1,
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.process_prefill_chunk": 1,
    "dllm/mixin/scheduler.py:DllmManager.take": 1,
    "managers/hisparse_coordinator.py:HiSparseCoordinator._release_staging": 1,
    "managers/scheduler.py:Scheduler._on_step_processed": 2,
    "managers/scheduler.py:Scheduler._retract_batches": 1,
    "managers/scheduler_components/deferred_responses.py:DeferredResponses._abort": 1,
    "managers/scheduler_components/deferred_responses.py:DeferredResponses.stream_released": 1,
}


class _EditCalls(ast.NodeVisitor):
    def __init__(self, path: str, found: collections.Counter):
        self.path, self.found, self.scope = path, found, []

    def _visit_scope(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _visit_scope

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in _EDITS:
            self.found[f"{self.path}:{'.'.join(self.scope)}"] += 1
        self.generic_visit(node)


class TestListEdits(CustomTestCase):
    def test_lists_are_edited_through_move_and_the_retirer(self):
        found = collections.Counter()
        for file in sorted(_SRT.rglob("*.py")):
            path = file.relative_to(_SRT).as_posix()
            source = file.read_text()
            if path not in _EDITORS and any(f"{e}(" in source for e in _EDITS):
                _EditCalls(path, found).visit(ast.parse(source))
        self.assertEqual(dict(found), _DIRECT_EDITS)


class TestKeyedReqList(CustomTestCase):
    def test_join_keeps_order_and_leave_ignores_outsiders(self):
        a, b, c, outsider = (FakeReq(rid=rid, finished=lambda: False) for rid in "abcx")
        running = KeyedReqList(ListKind.DECODING)
        move([a, b], running)
        move([c], running)
        self.assertEqual(list(running), [a, b, c])

        b.finished = lambda: True
        running._leave_finished()
        running._leave([a, outsider])

        self.assertEqual(list(running), [c])
        self.assertNotIn(a, running)


def _reqs(rids):
    return [FakeReq(rid=rid) for rid in rids]


class TestMove(CustomTestCase):
    def test_move_wraps_into_a_wrapper_list_and_keeps_its_index(self):
        a, b, c, d = _reqs("abcd")
        src = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        move([a, b, c], src)
        dst = IndexedReqList(ListKind.PD_RELEASING, req_of=lambda x: x[0])
        held = (d, "held")
        move([held], dst)

        wrapped = [(a, "a"), (b, "b")]
        move(wrapped, dst)

        self.assertEqual(list(src), [c])
        self.assertEqual(list(dst.items()), [held, *wrapped])
        self.assertEqual(list(dst), [d, a, b])

    def test_a_request_is_in_at_most_one_list(self):
        a, b = _reqs("ab")
        held = IndexedReqList(ListKind.PD_BOOTSTRAP)
        move([a], held)
        dst = KeyedReqList(ListKind.DECODING)

        with self.assertRaisesRegex(
            RuntimeError, "a is in PD_BOOTSTRAP, cannot join DECODING"
        ):
            dst._join([b, a])
        with self.assertRaisesRegex(RuntimeError, "a is already in PD_BOOTSTRAP"):
            move([a], held)

        self.assertEqual(list(held), [a])
        self.assertEqual(list(dst), [])
        self.assertIs(a.owner_list, held)
        self.assertIsNone(b.owner_list)

    def test_every_way_out_of_a_list_clears_the_owner(self):
        a, b, c = _reqs("abc")
        queue = IndexedReqList(ListKind.WAITING)
        move([a, b], queue)
        running = KeyedReqList(ListKind.DECODING)
        move([c], running)

        queue._leave([a])
        running._drop_all()
        queue._drop_all()

        self.assertEqual([r.owner_list for r in (a, b, c)], [None, None, None])

    def test_a_request_moved_on_reads_as_gone(self):
        a, b = _reqs("ab")
        queue = IndexedReqList(ListKind.WAITING)
        move([a, b], queue)
        other = KeyedReqList(ListKind.DECODING)

        move([a], other)

        self.assertEqual(list(queue), [b])
        self.assertIs(queue[0], b)
        self.assertEqual(list(queue.items()), [b])
        self.assertEqual(len(queue), 1)
        self.assertNotIn(a, queue)
        queue._drop_all()
        self.assertIs(a.owner_list, other)
        self.assertIsNone(b.owner_list)

    def test_a_request_back_in_the_list_it_left_is_in_it_once(self):
        a, b = _reqs("ab")
        queue = IndexedReqList(ListKind.WAITING)
        move([a, b], queue)

        queue._leave([a])
        self.assertEqual(len(queue), 1)
        move([a], queue)

        self.assertEqual(list(queue), [b, a])
        self.assertEqual(len(queue), 2)

    def test_a_reorder_keeps_the_same_requests(self):
        a, b, c = _reqs("abc")
        queue = IndexedReqList(ListKind.WAITING)
        move([a, b], queue)

        queue.reorder([b, a])
        with self.assertRaisesRegex(RuntimeError, "reorder of WAITING changed"):
            queue.reorder([b, c])

        self.assertEqual(list(queue), [b, a])
        self.assertIsNone(c.owner_list)


class TestAssertEmpty(CustomTestCase):
    def test_names_the_lists_a_request_is_still_in(self):
        a, b = _reqs("ab")
        sending = IndexedReqList(ListKind.PD_SENDING)
        prealloc = IndexedReqList(ListKind.PD_PREALLOC)
        move([a], sending)
        move([b], prealloc)
        lists = {ListKind.PD_SENDING: sending, ListKind.PD_PREALLOC: prealloc}

        sending._leave([a])
        with self.assertRaisesRegex(AssertionError, r"\['PD_PREALLOC'\]"):
            assert_empty(lists, PD_KINDS)
        assert_empty(lists, {ListKind.PD_SENDING})


if __name__ == "__main__":
    unittest.main()
