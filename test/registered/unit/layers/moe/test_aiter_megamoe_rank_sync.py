import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.mega_moe_flydsl import _should_overlap_shared_and_routed
from sglang.srt.managers.scheduler_components.dp_attn import (
    ForwardMode,
    _update_gather_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_rank_sync_forces_overlap_off(monkeypatch):
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_flydsl.get_is_capture_mode",
        lambda: True,
    )
    moe = SimpleNamespace(alt_stream=object(), num_fused_shared_experts=0)
    assert not _should_overlap_shared_and_routed(moe, 32)

    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "0")
    assert _should_overlap_shared_and_routed(moe, 32)


def test_mlp_sync_retains_full_vector_without_tp_gather():
    batch = SimpleNamespace()
    sync = SimpleNamespace(
        global_num_tokens=[32, 31, 7659, 256],
        global_num_tokens_for_logprob=[32, 31, 1, 1],
        num_tokens=32,
        num_tokens_for_logprob=32,
        is_extend_in_batch=True,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        can_run_decode_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
        tp0_info_cpu=torch.tensor(
            [
                [32, 0, 0, 0, 0, ForwardMode.DECODE.value, 0],
                [31, 0, 0, 0, 0, ForwardMode.DECODE.value, 0],
                [7659, 0, 0, 1, 0, ForwardMode.EXTEND.value, 0],
                [256, 0, 0, 1, 0, ForwardMode.EXTEND.value, 0],
            ],
            dtype=torch.int64,
        ),
    )
    _update_gather_batch(batch, sync, require_mlp_tp_gather=False)
    assert batch.global_num_tokens == [32]
    assert batch.mega_moe_global_num_tokens == [32, 31, 7659, 256]
    assert batch.mega_moe_sync_tokens == 7659
    assert batch.is_extend_in_batch


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
