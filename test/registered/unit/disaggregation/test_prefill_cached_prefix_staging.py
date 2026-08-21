import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.recorded = False
        self.ready = True

    def record(self):
        self.recorded = True

    def query(self):
        return self.ready


class _Allocator:
    page_size = 4

    @staticmethod
    def translate_kv_indices_for_transfer(indices):
        return indices + 100

    @staticmethod
    def translate_loc_from_full_to_swa(indices):
        return indices + 200


def test_stage_final_swa_transfer_respects_decode_prefix():
    req = SimpleNamespace(
        pending_bootstrap=False,
        extend_range=SimpleNamespace(end=20),
        origin_input_ids=list(range(20)),
        start_send_idx=0,
        req_pool_idx=0,
        rid="request-1",
        disagg_decode_prefix_len=12,
    )
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=_Allocator(),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(32).view(1, 32)),
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[StateType.SWA])
            )
        ),
        sliding_window_size=16,
        disagg_prefill_pending_chunk_rids=set(),
    )
    staged_full = torch.tensor([1])
    staged_swa = torch.tensor([2])

    with (
        patch(
            "sglang.srt.disaggregation.prefill._copy_page_indices_to_pinned_cpu",
            side_effect=[staged_full, staged_swa],
        ) as copy_page_ids,
        patch("sglang.srt.disaggregation.prefill.is_aborted", return_value=False),
    ):
        staged = SchedulerDisaggregationPrefillMixin.stage_prefill_transfer_indices(
            scheduler, SimpleNamespace(reqs=[req])
        )

    assert staged[req.rid].swa_page_indices is staged_swa
    torch.testing.assert_close(
        copy_page_ids.call_args_list[1].args[0], torch.arange(12, 20) + 200
    )


def test_stage_cached_prefix_transfer_indices_before_send():
    req = SimpleNamespace(
        pending_bootstrap=False,
        to_finish=False,
        early_send_prefix_end=None,
        prefix_indices=torch.arange(12),
        host_hit_length=4,
        start_send_idx=0,
        req_pool_idx=1,
    )
    scheduler = SimpleNamespace(
        enable_staging=True,
        token_to_kv_pool_allocator=_Allocator(),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(32).view(2, 16)),
        device_module=SimpleNamespace(Event=_Event),
    )
    staged_page_ids = torch.tensor([29, 30])

    with (
        patch(
            "sglang.srt.disaggregation.prefill." "_copy_page_indices_to_pinned_cpu",
            return_value=staged_page_ids,
        ) as copy_page_ids,
        patch(
            "sglang.srt.disaggregation.prefill.envs."
            "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get",
            return_value=True,
        ),
        patch("sglang.srt.disaggregation.prefill.is_aborted", return_value=False),
    ):
        SchedulerDisaggregationPrefillMixin.stage_cached_prefix_transfer_indices(
            scheduler, SimpleNamespace(reqs=[req])
        )

    assert req.early_send_prefix_end == 8
    staged = req._staged_cached_prefix_transfer_indices
    assert staged.end_idx == 8
    assert staged.page_indices is staged_page_ids
    assert staged.ready_event.recorded
    copy_page_ids.assert_called_once()
    torch.testing.assert_close(
        copy_page_ids.call_args.args[0],
        torch.arange(16, 24) + 100,
    )
    assert copy_page_ids.call_args.args[1] == 4


def test_stage_cached_prefix_skips_partial_page():
    req = SimpleNamespace(
        pending_bootstrap=False,
        to_finish=False,
        early_send_prefix_end=None,
        prefix_indices=torch.arange(10),
        host_hit_length=4,
        start_send_idx=0,
        req_pool_idx=0,
    )
    scheduler = SimpleNamespace(
        enable_staging=True,
        token_to_kv_pool_allocator=_Allocator(),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(16).view(1, 16)),
        device_module=SimpleNamespace(Event=_Event),
    )

    with (
        patch(
            "sglang.srt.disaggregation.prefill." "_copy_page_indices_to_pinned_cpu"
        ) as copy_page_ids,
        patch(
            "sglang.srt.disaggregation.prefill.envs."
            "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get",
            return_value=True,
        ),
        patch("sglang.srt.disaggregation.prefill.is_aborted", return_value=False),
    ):
        SchedulerDisaggregationPrefillMixin.stage_cached_prefix_transfer_indices(
            scheduler, SimpleNamespace(reqs=[req])
        )

    copy_page_ids.assert_not_called()
    assert not hasattr(req, "_staged_cached_prefix_transfer_indices")


def test_cached_prefix_early_send_does_not_wait_for_staging_copy():
    ready_event = _Event()
    ready_event.ready = False
    req = SimpleNamespace(
        pending_bootstrap=False,
        early_send_prefix_end=8,
        prefix_indices=torch.arange(12),
        host_hit_length=4,
        start_send_idx=0,
        _staged_cached_prefix_transfer_indices=SimpleNamespace(
            end_idx=8,
            ready_event=ready_event,
        ),
    )
    scheduler = SimpleNamespace(
        enable_staging=True,
        enable_overlap=True,
        token_to_kv_pool_allocator=_Allocator(),
        send_kv_chunk=Mock(),
    )

    with (
        patch(
            "sglang.srt.disaggregation.prefill.envs."
            "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get",
            return_value=True,
        ),
        patch("torch.cuda.Event") as cuda_event,
    ):
        SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk(
            scheduler, req
        )

    scheduler.send_kv_chunk.assert_not_called()
    cuda_event.assert_not_called()


def test_cached_prefix_early_send_uses_ready_staging_copy():
    ready_event = _Event()
    req = SimpleNamespace(
        pending_bootstrap=False,
        early_send_prefix_end=8,
        prefix_indices=torch.arange(12),
        host_hit_length=4,
        start_send_idx=0,
        _staged_cached_prefix_transfer_indices=SimpleNamespace(
            end_idx=8,
            ready_event=ready_event,
        ),
        disagg_kv_sender=SimpleNamespace(),
    )
    scheduler = SimpleNamespace(
        enable_staging=True,
        enable_overlap=False,
        token_to_kv_pool_allocator=_Allocator(),
        send_kv_chunk=Mock(),
    )

    with (
        patch(
            "sglang.srt.disaggregation.prefill.envs."
            "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get",
            return_value=True,
        ),
    ):
        SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk(
            scheduler, req
        )

    scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=False, end_idx=8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
