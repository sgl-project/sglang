# SPDX-License-Identifier: Apache-2.0
"""Under PP, a chunk's KV send must not let the sender read pages before the
chunk's forward has written them."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE, CHUNK_TOKENS = 4, 8


class _Stream:
    """Tracks which streams' earlier work this stream is ordered after."""

    def __init__(self):
        self.after = set()

    def wait_stream(self, other):
        self.after |= {other} | other.after


class _Event:
    def record(self, stream):
        self.after = {stream} | stream.after


class _Req(SimpleNamespace):
    # process_prefill_chunk keeps requests in a set.
    __hash__ = object.__hash__


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, req):
        self.schedule_stream = _Stream()
        self.forward_stream = _Stream()
        self.device_module = SimpleNamespace(
            Event=_Event, current_stream=lambda: self.schedule_stream
        )
        self.chunked_req = req
        self.enable_overlap = False
        self.enable_staging = False
        self.kv_checksum_computer = None
        self.tree_cache = SimpleNamespace(supports_prefix_sharing=lambda: True)
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=PAGE_SIZE,
            translate_kv_indices_for_transfer=lambda x: x,
            storage=None,
        )
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(CHUNK_TOKENS).reshape(1, -1)
        )

    def checkpoint_disagg_prefill(self, req):
        pass

    def check_bootstrap(self, req):
        return True


def _send_computed_chunk(pp_size):
    """Run process_prefill_chunk on a non-final chunk whose forward was just
    launched, with a sender that reads KV once the indices event completes."""
    sender = SimpleNamespace(
        supports_host_tensor_indices=True,
        kv_mgr=SimpleNamespace(enable_all_cp_ranks_for_transfer=False, kv_shard_size=1),
        get_max_transfer_tokens=lambda: None,
        should_send_kv_chunk=lambda num_pages, is_last: True,
        send=Mock(),
    )
    req = _Req(
        rid="pp-chunk",
        kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=CHUNK_TOKENS),
        skip_radix_cache_insert=False,
        origin_input_ids=[0] * (2 * CHUNK_TOKENS),
        prefix_len=0,
        extend_end=CHUNK_TOKENS,
        start_send_idx=0,
        disagg_decode_prefix_len=0,
        disagg_kv_sender=sender,
    )
    scheduler = _Scheduler(req)
    with (
        published_topology(pp_size=pp_size),
        get_context().resources.override(streams={"kv_send_fence": _Stream()}),
    ):
        scheduler.process_prefill_chunk(
            last_batch=None, running_batch=SimpleNamespace(batch_is_full=True)
        )
    return scheduler, sender.send.call_args.kwargs["indices_event"]


class TestPrefillPPKVSendAfterForward(CustomTestCase):
    def test_pp_chunk_send_waits_for_its_forward(self):
        # The PP loop sends the previous chunk while its forward may still run
        # on forward_stream; the indices event alone covers only the copies.
        scheduler, event = _send_computed_chunk(pp_size=2)
        self.assertIn(scheduler.forward_stream, event.after)
        self.assertIn(scheduler.schedule_stream, event.after)
        # The scheduling stream itself must not wait for the forward.
        self.assertNotIn(scheduler.forward_stream, scheduler.schedule_stream.after)

    def test_non_pp_chunk_send_keeps_the_scheduling_stream(self):
        # Without PP the forward runs on the scheduling stream already.
        scheduler, event = _send_computed_chunk(pp_size=1)
        self.assertEqual(event.after, {scheduler.schedule_stream})


if __name__ == "__main__":
    unittest.main()
