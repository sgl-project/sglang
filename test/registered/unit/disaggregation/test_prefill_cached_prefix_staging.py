from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
)


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


class _Allocator:
    page_size = 4

    @staticmethod
    def translate_kv_indices_for_transfer(indices):
        return indices + 100


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
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(32).view(2, 16)
        ),
        device_module=SimpleNamespace(Event=_Event),
    )
    staged_page_ids = torch.tensor([29, 30])

    with (
        patch(
            "sglang.srt.disaggregation.prefill."
            "_copy_page_indices_to_pinned_cpu",
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
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(16).view(1, 16)
        ),
        device_module=SimpleNamespace(Event=_Event),
    )

    with (
        patch(
            "sglang.srt.disaggregation.prefill."
            "_copy_page_indices_to_pinned_cpu"
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
