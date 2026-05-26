"""Regression test for the PP stale-chunked_req exclusion (S3 invariant).

In pipeline parallelism, after a chunked req's last chunk completes in
microbatch A, the same req still appears in microbatch B's
`last_batch.chunked_req` field as a stale pointer. Before merging
`last_batch` into `running_batch`, the scheduler must add that stale
pointer to `chunked_req_to_exclude` so it is filtered out.

Source site: python/sglang/srt/managers/scheduler.py, the branch
guarded by `self.last_batch.chunked_req is not None` inside
`get_next_batch_to_run` (search for the comment
"after the last chunk, the current microbatch still track outdated
chunked_req. We need to discard it.").
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _make_req(rid: str) -> Req:
    """Construct a minimal Req that the scheduler's exclusion path can compare by identity."""
    req = Req.__new__(Req)
    req.rid = rid
    req.origin_input_ids = array("q", [0])
    req.output_ids = array("q")
    req.fill_ids = array("q", [0])
    req.prefix_indices = torch.zeros(0, dtype=torch.int64)
    req.req_pool_idx = 0
    req.extend_input_len = 0
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.cache_protected_len = 0
    req.skip_radix_cache_insert = False
    req.last_node = None
    req.swa_uuid_for_lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_last_batch(*, reqs: list, chunked_req) -> MagicMock:
    """Mock `last_batch` with a `filter_batch` that honors `chunked_req_to_exclude`.

    Real `ScheduleBatch.filter_batch` is heavy (tensors, sampling_info, ...),
    but the only semantics this test cares about is: it must drop reqs whose
    identity is in `chunked_req_to_exclude`. We implement that minimal shape
    so the post-merge `running_batch.reqs` is a faithful function of the
    exclusion set the scheduler builds.
    """
    last_batch = MagicMock()
    last_batch.reqs = list(reqs)
    last_batch.chunked_req = chunked_req
    last_batch.forward_mode = ForwardMode.EXTEND
    last_batch.is_prefill_only = False
    last_batch.batch_is_full = False
    last_batch.batch_size = lambda: len(last_batch.reqs)
    last_batch.is_empty = lambda: len(last_batch.reqs) == 0

    def _filter_batch(chunked_req_to_exclude=None, keep_indices=None, **_):
        excluded = chunked_req_to_exclude or []
        last_batch.reqs = [r for r in last_batch.reqs if r not in excluded]

    last_batch.filter_batch = MagicMock(side_effect=_filter_batch)
    return last_batch


def _scheduler_for_get_next_batch(*, last_batch, chunked_req) -> Scheduler:
    """Mirror of test_scheduler_chunked_req_gate's helper, with `last_batch` wired in.

    Differences from that helper:
      * `last_batch` is non-None and exposes a filter_batch/merge surface, since
        the PP-stale exclusion branch only runs when `last_batch` is set.
      * `running_batch` participates in the merge, so we keep it as a MagicMock
        whose `is_empty` initially returns True; after assignment in the merge
        path we re-read the post-state directly off the scheduler.
    """
    s = Scheduler.__new__(Scheduler)
    s._abort_on_waiting_timeout = MagicMock()
    s._abort_on_running_timeout = MagicMock()
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    s.last_batch = last_batch
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
    s.get_new_batch_prefill = MagicMock(return_value=None)
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s._maybe_prepare_ngram_embedding = MagicMock(side_effect=lambda batch: batch)
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    s.tree_cache = MagicMock()
    s.chunked_req = chunked_req
    s._chunked_req_scheduled_last_iter = False
    s.stash_chunked_request = MagicMock()
    return s


class TestPpStaleChunkedReqExclude(CustomTestCase):
    def test_stale_chunked_req_in_last_batch_must_be_excluded(self):
        """Stale `last_batch.chunked_req` must be filtered before merge into running_batch."""
        stale_req = _make_req("stale-chunked-req")
        normal_req = _make_req("normal-req")
        last_batch = _make_last_batch(
            reqs=[stale_req, normal_req], chunked_req=stale_req
        )
        # `s.chunked_req is None`: the scheduler-level chunked_req is gone
        # (last chunk completed). The only reference to the stale req is the
        # outdated `last_batch.chunked_req` pointer from the *other* microbatch.
        s = _scheduler_for_get_next_batch(last_batch=last_batch, chunked_req=None)

        Scheduler.get_next_batch_to_run(s)

        last_batch.filter_batch.assert_called_once()
        call_kwargs = last_batch.filter_batch.call_args.kwargs
        self.assertIn("chunked_req_to_exclude", call_kwargs)
        self.assertIn(stale_req, call_kwargs["chunked_req_to_exclude"])

        # After merge, the stale req must not appear anywhere downstream.
        self.assertNotIn(stale_req, last_batch.reqs)
        self.assertNotIn(stale_req, s.running_batch.reqs)
        self.assertIn(normal_req, s.running_batch.reqs)

    def test_fresh_last_batch_chunked_req_still_filters_normally(self):
        """Without a stale pointer (`last_batch.chunked_req is None`), no over-exclusion."""
        normal_req_a = _make_req("normal-req-a")
        normal_req_b = _make_req("normal-req-b")
        last_batch = _make_last_batch(
            reqs=[normal_req_a, normal_req_b], chunked_req=None
        )
        s = _scheduler_for_get_next_batch(last_batch=last_batch, chunked_req=None)

        Scheduler.get_next_batch_to_run(s)

        # filter_batch is still called (the PP-stale branch is always reached
        # when last_batch.forward_mode.is_extend()), but no req gets excluded.
        last_batch.filter_batch.assert_called_once()
        excluded = last_batch.filter_batch.call_args.kwargs.get(
            "chunked_req_to_exclude", []
        )
        self.assertNotIn(normal_req_a, excluded)
        self.assertNotIn(normal_req_b, excluded)

        # Both reqs survive into the merged running_batch.
        self.assertIn(normal_req_a, s.running_batch.reqs)
        self.assertIn(normal_req_b, s.running_batch.reqs)


if __name__ == "__main__":
    unittest.main()
