from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.mega_moe_flydsl import (
    _forward_with_sync_config,
    _should_overlap_shared_and_routed,
    _sync_tokens,
)
from sglang.srt.managers.scheduler_components.dp_attn import (
    ForwardMode,
    _should_defer_decode_for_mega_rank_sync,
    _update_gather_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_sync_tokens_uses_rank_global_config(monkeypatch):
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    forward_batch = SimpleNamespace(
        mega_moe_sync_tokens=7659,
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=True,
    )
    assert _sync_tokens(forward_batch, local_tokens=32, mtpr=8192) == 8192
    assert _sync_tokens(forward_batch, local_tokens=7659, mtpr=8192) == 8192


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


def test_sync_tokens_gate_and_capacity(monkeypatch):
    forward_batch = SimpleNamespace(
        mega_moe_sync_tokens=8192,
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
    )
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "0")
    assert _sync_tokens(forward_batch, local_tokens=32, mtpr=8192) == 32

    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    forward_batch.mega_moe_sync_tokens = 8193
    with pytest.raises(ValueError, match="exceeds MTPR"):
        _sync_tokens(forward_batch, local_tokens=32, mtpr=8192)


def test_forward_uses_global_config_with_local_runtime_shape():
    calls = []

    class Config:
        stage1 = object()

    class FakeMega:
        def _select_config(self, tokens):
            calls.append(("select", tokens))
            return Config()

        def quantize(self, x):
            return x, torch.ones((x.shape[0], 1), dtype=torch.uint8)

        def _run_fused_stage1(self, x, weights, scales, ids, *, stream, config):
            calls.append(("stage1", x.shape[0], config, stream))

        def _run_stage2(self, run_tokens, stream, slice_output, config):
            calls.append(("stage2", run_tokens, slice_output, stream))
            return torch.zeros((run_tokens, 4), dtype=torch.bfloat16)

    output = _forward_with_sync_config(
        FakeMega(),
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int32),
        config_tokens=8192,
    )
    assert output.shape == (2, 4)
    assert calls[0] == ("select", 8192)
    assert calls[1][0:2] == ("stage1", 2)
    assert calls[2] == ("stage2", 2, False, None)


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


def test_idle_rank_keeps_single_materialized_row(monkeypatch):
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    forward_batch = SimpleNamespace(
        mega_moe_sync_tokens=8192,
        forward_mode=ForwardMode.IDLE,
        is_extend_in_batch=True,
    )
    assert _sync_tokens(forward_batch, local_tokens=1, mtpr=8192) == 1

    forward_batch.forward_mode = ForwardMode.EXTEND
    forward_batch._original_forward_mode = ForwardMode.IDLE
    assert _sync_tokens(forward_batch, local_tokens=1, mtpr=8192) == 1


def test_decode_is_deferred_when_any_rank_prefills(monkeypatch):
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "1")
    sync = SimpleNamespace(is_extend_in_batch=True)
    assert _should_defer_decode_for_mega_rank_sync(
        SimpleNamespace(forward_mode=ForwardMode.DECODE), sync
    )
