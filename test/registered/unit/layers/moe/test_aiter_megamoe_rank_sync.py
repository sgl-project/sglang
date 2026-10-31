import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.mega_moe_flydsl import (
    forward_mega_moe,
    should_use_mega_moe,
)
from sglang.srt.managers.scheduler_components.dp_attn import (
    ForwardMode,
    _update_gather_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def test_forward_supports_moe_without_shared_experts(monkeypatch):
    expected = torch.ones(2, 3)
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_flydsl._run_mega_routed",
        lambda moe, hidden_states, forward_batch, input_ids_global: expected,
    )

    assert forward_mega_moe(SimpleNamespace(), torch.zeros(2, 3)) is expected


def test_target_mimo_megamoe_respects_token_cap(monkeypatch):
    moe = SimpleNamespace(
        _use_target_mega_moe=True,
        experts=SimpleNamespace(_mega_moe_weights_built=True),
    )
    monkeypatch.setenv("SGLANG_MIMO_MEGA_MAX_TOKENS", "32")

    assert should_use_mega_moe(moe, torch.zeros(32, 3))
    assert not should_use_mega_moe(moe, torch.zeros(33, 3))


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
        can_run_draft_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
        prefill_cuda_graph_max_prefix_len=0,
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
    assert batch.is_extend_in_batch


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
