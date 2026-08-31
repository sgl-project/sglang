import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.layers.attention.dots_hybrid_backend import (
    DotsHybridAttnBackend,
    DotsSWAMLAAttnBackend,
    _metadata_mismatches_dp_padded_batch,
    _normalize_cache_seqlens_rows,
)
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
from sglang.srt.layers.attention.swa_mla_fallback.ops import (
    gather_page64_kv_latent,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _batch(*, bs: int, num_tokens: int, original_bs: int | None = None):
    return SimpleNamespace(
        batch_size=bs,
        out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
        forward_mode=SimpleNamespace(),
        _original_batch_size=original_bs,
    )


def _fa_metadata(*, bs: int, num_tokens: int):
    return FlashAttentionMetadata(
        page_table=torch.zeros((bs, 4), dtype=torch.int32),
        swa_page_table=torch.zeros((bs, 4), dtype=torch.int32),
        cache_seqlens_int32=torch.ones(bs, dtype=torch.int32),
        swa_out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
    )


def test_mismatch_detects_short_page_table_and_swa_loc():
    metadata = _fa_metadata(bs=1, num_tokens=1)
    assert _metadata_mismatches_dp_padded_batch(metadata, _batch(bs=2, num_tokens=2))

    metadata = _fa_metadata(bs=2, num_tokens=2)
    assert not _metadata_mismatches_dp_padded_batch(
        metadata, _batch(bs=2, num_tokens=2)
    )


def test_swa_backend_rebuilds_when_dp_padding_changes_rows():
    inner = SimpleNamespace(
        forward_metadata=_fa_metadata(bs=1, num_tokens=1),
        init_forward_metadata=MagicMock(),
    )
    backend = object.__new__(DotsSWAMLAAttnBackend)
    backend.backend = inner
    backend._active_backend = inner
    backend._prefill_metadata = None
    backend._dp_rebuilt_batch_id = None
    backend.init_forward_metadata = MagicMock()

    backend.maybe_rebuild_metadata_after_dp_padding(_batch(bs=2, num_tokens=2))
    backend.init_forward_metadata.assert_called_once()


def test_hybrid_rebuilds_when_dp_padding_changes_batch_size():
    matching = _fa_metadata(bs=2, num_tokens=2)
    hybrid = object.__new__(DotsHybridAttnBackend)
    hybrid.dsa_backend = SimpleNamespace(forward_metadata=matching)
    hybrid.swa_backend = SimpleNamespace(forward_metadata=matching)
    hybrid._dp_rebuilt_batch_id = None
    hybrid.init_forward_metadata = MagicMock()

    hybrid.maybe_rebuild_metadata_after_dp_padding(
        _batch(bs=2, num_tokens=2, original_bs=1)
    )
    hybrid.init_forward_metadata.assert_called_once()


def test_normalize_cache_seqlens_preserves_planned_rows_and_pads_dummy_rows():
    cache_seqlens = torch.tensor([17, 23], dtype=torch.int32)
    seq_lens = torch.tensor([100, 200, 300, 400], dtype=torch.int64)

    normalized = _normalize_cache_seqlens_rows(cache_seqlens, seq_lens, 4)

    assert torch.equal(normalized, torch.tensor([17, 23, 300, 400], dtype=torch.int32))


def test_normalize_cache_seqlens_truncates_extra_rows():
    cache_seqlens = torch.tensor([17, 23, 29], dtype=torch.int32)
    seq_lens = torch.tensor([100, 200], dtype=torch.int64)

    normalized = _normalize_cache_seqlens_rows(cache_seqlens, seq_lens, 2)

    assert torch.equal(normalized, torch.tensor([17, 23], dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_page64_gather_masks_out_of_range_page_table_entries():
    kv_cache_dim = 128
    k_cache = torch.arange(64 * kv_cache_dim, dtype=torch.float32, device="cuda").view(
        64, 1, kv_cache_dim
    )
    # Row 0 has a sequence longer than its one-page table. Row 1 points past
    # the physical KV pool. Both can occur transiently when DP padding changes
    # the live batch after speculative metadata was planned.
    block_table = torch.tensor([[0], [9]], dtype=torch.int32, device="cuda")
    cache_seqlens = torch.tensor([130, 64], dtype=torch.int32, device="cuda")

    gathered, valid = gather_page64_kv_latent(
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        window_size=128,
        s_q=1,
        kv_cache_dim=kv_cache_dim,
    )
    torch.cuda.synchronize()

    assert valid[0, :62].all()
    assert not valid[0, 62:].any()
    assert not valid[1].any()
    torch.testing.assert_close(gathered[0, :62], k_cache[2:64, 0])
    assert not gathered[0, 62:].any()
    assert not gathered[1].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
