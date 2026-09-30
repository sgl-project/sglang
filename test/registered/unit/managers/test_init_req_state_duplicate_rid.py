"""Regression test: TokenizerManager._init_req_state must not leak states.

When a batch request's rid list collides with an in-flight rid, the duplicate
check used to raise *after* states for the earlier items had already been
inserted into rid_to_state. The exception escapes before the cleanup block in
generate_request, so those states leaked forever: the rids stay poisoned
(every reuse fails with "Duplicate request ID") and repeated crafted batches
grow tokenizer-process memory without bound.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.test_utils import CustomTestCase


class _FakeBatchObj:
    """Minimal stand-in for a batch GenerateReqInput."""

    is_single = False
    received_time = 0.0
    external_trace_header = None
    bootstrap_room = None

    def __init__(self, rids):
        self.rid = list(rids)

    def __getitem__(self, i):
        return SimpleNamespace(rid=self.rid[i])


def _make_manager_stub():
    return SimpleNamespace(
        rid_to_state={},
        enable_trace=False,
        disaggregation_mode=None,
    )


class TestInitReqStateDuplicateRid(CustomTestCase):
    def test_duplicate_rid_does_not_leak_earlier_states(self):
        mgr = _make_manager_stub()
        mgr.rid_to_state["victim"] = object()  # an in-flight request
        obj = _FakeBatchObj(["new-1", "new-2", "victim"])
        with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
            TokenizerManager._init_req_state(mgr, obj)
        self.assertNotIn("new-1", mgr.rid_to_state)
        self.assertNotIn("new-2", mgr.rid_to_state)

    def test_unique_rids_all_registered(self):
        mgr = _make_manager_stub()
        obj = _FakeBatchObj(["a", "b", "c"])
        TokenizerManager._init_req_state(mgr, obj)
        self.assertEqual(set(mgr.rid_to_state), {"a", "b", "c"})

    def test_single_request_duplicate_raises(self):
        mgr = _make_manager_stub()
        existing = object()
        mgr.rid_to_state["x"] = existing
        obj = SimpleNamespace(
            is_single=True, rid="x", received_time=0.0, external_trace_header=None
        )
        with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
            TokenizerManager._init_req_state(mgr, obj)
        self.assertIs(mgr.rid_to_state["x"], existing)


if __name__ == "__main__":
    unittest.main()
