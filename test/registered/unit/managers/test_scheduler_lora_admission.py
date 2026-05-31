"""Regression tests for LoRA admission when chunked_req is present (#23141)."""

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_req(lora_id):
    req = Req.__new__(Req)
    req.lora_id = lora_id
    req.sampling_params = SimpleNamespace(max_new_tokens=16, ignore_eos=False)
    return req


def _scheduler_with_lora(
    *,
    running_batch_loras,
    max_loras_per_batch=2,
    enable_overlap_loading=False,
):
    s = Scheduler.__new__(Scheduler)
    s.enable_lora = True
    s.lora_drainer = None
    s.enable_lora_overlap_loading = enable_overlap_loading

    s.running_batch = MagicMock()
    s.running_batch.reqs = [_make_req(l) for l in running_batch_loras]

    lora_mgr = MagicMock()
    lora_mgr.max_loras_per_batch = max_loras_per_batch
    lora_mgr.validate_lora_batch = MagicMock(
        side_effect=lambda ids: len(ids) <= max_loras_per_batch
    )
    s.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(lora_manager=lora_mgr))
    return s


class TestCollectCommittedLoraIds(CustomTestCase):
    """``_collect_committed_lora_ids`` must report every lora_id that will
    appear in the next batch so admission can enforce ``max_loras_per_batch``."""

    def test_running_batch_only(self):
        s = _scheduler_with_lora(running_batch_loras=["A", "B"])
        self.assertEqual(s._collect_committed_lora_ids([]), {"A", "B"})

    def test_includes_chunked_req_from_can_run_list(self):
        s = _scheduler_with_lora(running_batch_loras=["A"])
        chunked = _make_req("X")
        self.assertEqual(s._collect_committed_lora_ids([chunked]), {"A", "X"})

    def test_base_model_request_counted_as_distinct_uid(self):
        # A base-model request (lora_id=None) is a distinct entry in the
        # set computed by fetch_new_loras, so admission must see it too.
        s = _scheduler_with_lora(running_batch_loras=[None])
        chunked = _make_req("X")
        self.assertEqual(s._collect_committed_lora_ids([chunked]), {None, "X"})


class TestLoraAdmissionWithChunkedReq(CustomTestCase):
    """End-to-end regression for #23141: when ``max_loras_per_batch=N`` and a
    chunked LoRA prefill already sits in ``can_run_list``, admission must
    reject a further distinct LoRA request that would form an (N+1)-th UID.
    """

    def test_rejects_new_adapter_when_chunked_fills_last_slot(self):
        s = _scheduler_with_lora(running_batch_loras=["A"], max_loras_per_batch=2)
        chunked_req = _make_req("X")
        running_loras = s._collect_committed_lora_ids([chunked_req])

        new_req = _make_req("B")
        self.assertFalse(s._can_schedule_lora_req(new_req, running_loras))

    def test_admits_request_with_same_adapter_as_chunked(self):
        s = _scheduler_with_lora(running_batch_loras=["A"], max_loras_per_batch=2)
        chunked_req = _make_req("X")
        running_loras = s._collect_committed_lora_ids([chunked_req])

        same_as_chunked = _make_req("X")
        self.assertTrue(s._can_schedule_lora_req(same_as_chunked, running_loras))

    def test_without_chunked_req_behavior_unchanged(self):
        s = _scheduler_with_lora(running_batch_loras=["A"], max_loras_per_batch=2)
        running_loras = s._collect_committed_lora_ids([])

        new_req = _make_req("B")
        self.assertTrue(s._can_schedule_lora_req(new_req, running_loras))

    def test_base_plus_n_loras_at_cap_rejects_next(self):
        # Mirrors Yunzez's N adapters + 1 base-model scenario: when the base
        # request already occupies one UID slot, admission must reject the
        # (N+1)-th distinct adapter regardless of whether it arrives via
        # chunked_req or waiting_queue.
        s = _scheduler_with_lora(running_batch_loras=[None], max_loras_per_batch=2)
        chunked_req = _make_req("A")
        running_loras = s._collect_committed_lora_ids([chunked_req])

        new_req = _make_req("B")
        self.assertFalse(s._can_schedule_lora_req(new_req, running_loras))

    def test_overlap_loading_sees_chunked_lora_in_running_set(self):
        # The overlap-loading branch consumes the same ``running_loras`` set
        # that admission computes, so the chunked_req's adapter must be
        # visible to ``try_overlap_load_lora`` as well.
        s = _scheduler_with_lora(
            running_batch_loras=["A"],
            max_loras_per_batch=2,
            enable_overlap_loading=True,
        )
        s.lora_overlap_loader = MagicMock()
        s.lora_overlap_loader.try_overlap_load_lora.return_value = False

        running_loras = s._collect_committed_lora_ids([_make_req("X")])
        self.assertFalse(s._can_schedule_lora_req(_make_req("B"), running_loras))
        s.lora_overlap_loader.try_overlap_load_lora.assert_called_once_with(
            "B", {"A", "X"}
        )


class TestAdmissionCallSiteWiring(CustomTestCase):
    """Source-level guard: the helper is only useful if admission actually
    calls it with ``adder.can_run_list``. A silent revert of the call site
    (re-inlining the old ``{req.lora_id for req in self.running_batch.reqs}``)
    must trip this test even when the helper itself is left intact."""

    def test_admission_invokes_helper_with_can_run_list(self):
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_collect_committed_lora_ids"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "can_run_list"
            ):
                return
        self.fail(
            "_get_new_batch_prefill_raw must call "
            "self._collect_committed_lora_ids(adder.can_run_list); regressing "
            "this call site silently re-introduces sgl-project/sglang#23141."
        )


if __name__ == "__main__":
    unittest.main()
