from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _make_backend():
    backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
    backend.is_draft_worker = False
    backend.req_to_token_pool = SimpleNamespace(mamba_pool=Mock())
    return backend


def _make_forward_batch(mode):
    return SimpleNamespace(
        forward_mode=mode,
        mamba_clear_indices=torch.tensor([3], dtype=torch.int64),
        mamba_cow_src_indices=torch.tensor([4], dtype=torch.int64),
        mamba_cow_dst_indices=torch.tensor([5], dtype=torch.int64),
    )


def test_target_verify_does_not_replay_deferred_mamba_state_ops():
    backend = _make_backend()
    forward_batch = _make_forward_batch(ForwardMode.TARGET_VERIFY)

    backend._execute_deferred_mamba_cow_and_clear(forward_batch)

    backend.req_to_token_pool.mamba_pool.clear_slots.assert_not_called()
    backend.req_to_token_pool.mamba_pool.copy_from.assert_not_called()


def test_prefill_executes_and_consumes_deferred_mamba_state_ops():
    backend = _make_backend()
    forward_batch = _make_forward_batch(ForwardMode.EXTEND)
    clear_indices = forward_batch.mamba_clear_indices
    cow_src_indices = forward_batch.mamba_cow_src_indices
    cow_dst_indices = forward_batch.mamba_cow_dst_indices

    backend._execute_deferred_mamba_cow_and_clear(forward_batch)

    backend.req_to_token_pool.mamba_pool.clear_slots.assert_called_once_with(
        clear_indices
    )
    backend.req_to_token_pool.mamba_pool.copy_from.assert_called_once_with(
        cow_src_indices, cow_dst_indices
    )
    assert forward_batch.mamba_clear_indices is None
    assert forward_batch.mamba_cow_src_indices is None
    assert forward_batch.mamba_cow_dst_indices is None
