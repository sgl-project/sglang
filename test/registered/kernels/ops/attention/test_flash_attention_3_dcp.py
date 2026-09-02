"""FA3 absorbed-MLA correctness under cyclic decode context parallelism."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

skip_condition = not torch.cuda.is_available() or (
    torch.cuda.get_device_capability()[0] != 9
)


def _pack_sequences(sequences):
    batch_size = len(sequences)
    lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.int32,
        device=sequences[0].device,
    )
    max_len = max(int(lengths.max().item()), 1)
    page_table = torch.zeros(
        (batch_size, max_len), dtype=torch.int32, device=sequences[0].device
    )

    packed = []
    offset = 0
    for row, sequence in enumerate(sequences):
        length = sequence.shape[0]
        if length:
            packed.append(sequence)
            page_table[row, :length] = torch.arange(
                offset,
                offset + length,
                dtype=torch.int32,
                device=sequence.device,
            )
            offset += length

    return torch.cat(packed, dim=0).unsqueeze(1), page_table, lengths


def _run_absorbed_mla(q_rope, q_nope, k_sequences, v_sequences, only_qv):
    v_cache, page_table, cache_seqlens = _pack_sequences(v_sequences)
    if only_qv:
        k_cache = None
        q = None
    else:
        k_cache, k_page_table, k_seqlens = _pack_sequences(k_sequences)
        torch.testing.assert_close(k_page_table, page_table)
        torch.testing.assert_close(k_seqlens, cache_seqlens)
        q = q_rope

    batch_size, num_heads, _ = q_nope.shape
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q_nope.device)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    qk_dim = q_nope.shape[-1] + (0 if only_qv else q_rope.shape[-1])
    out, raw_lse, *rest = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=qk_dim**-0.5,
        causal=True,
        only_qv=only_qv,
        return_softmax_lse=True,
        num_splits=1,
        ver=3,
    )

    assert out.shape == (batch_size, num_heads, q_nope.shape[-1])
    # The backend feeds FA3 an unpadded [total_q, heads, dim] query. Pin the
    # corresponding kernel-facing LSE layout before it is converted to [B, H].
    assert raw_lse.shape == (num_heads, batch_size)
    return out, raw_lse.transpose(0, 1).contiguous(), cache_seqlens


def _check_cyclic_dcp_merge(only_qv):
    torch.manual_seed(7)
    device = "cuda"
    dtype = torch.bfloat16
    dcp_size = 4
    batch_size = 3
    num_heads = 8
    v_head_dim = 512
    rope_head_dim = 0 if only_qv else 64
    seq_lens = (1, 67, 259)

    q_nope = torch.randn(batch_size, num_heads, v_head_dim, dtype=dtype, device=device)
    q_rope = (
        None
        if only_qv
        else torch.randn(
            batch_size, num_heads, rope_head_dim, dtype=dtype, device=device
        )
    )
    v_sequences = [
        torch.randn(length, 1, v_head_dim, dtype=dtype, device=device)
        for length in seq_lens
    ]
    k_sequences = (
        None
        if only_qv
        else [
            torch.randn(length, 1, rope_head_dim, dtype=dtype, device=device)
            for length in seq_lens
        ]
    )

    full_out, full_lse, _ = _run_absorbed_mla(
        q_rope, q_nope, k_sequences, v_sequences, only_qv
    )

    partial_outputs = []
    partial_lses = []
    for rank in range(dcp_size):
        local_v = [sequence[rank::dcp_size] for sequence in v_sequences]
        local_k = (
            None if only_qv else [sequence[rank::dcp_size] for sequence in k_sequences]
        )
        local_out, local_lse, local_lens = _run_absorbed_mla(
            q_rope, q_nope, local_k, local_v, only_qv
        )

        # Match the backend's neutral state for requests with no KV on this rank.
        zero_kv = local_lens == 0
        local_out.masked_fill_(zero_kv[:, None, None], 0)
        local_lse.masked_fill_(zero_kv[:, None], float("-inf"))
        partial_outputs.append(local_out)
        partial_lses.append(local_lse)

    partial_outputs = torch.stack(partial_outputs)
    partial_lses = torch.stack(partial_lses)

    # FA3 emits natural-log LSE. The model-side DCP dispatch selects the
    # reducer's base-e path for the fa3 backend.
    merged_out, _ = dcp_lse_combine_triton(
        partial_outputs,
        partial_lses,
        is_lse_base_on_e=True,
        return_lse=False,
    )

    torch.testing.assert_close(
        merged_out.float(), full_out.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        torch.logsumexp(partial_lses, dim=0),
        full_lse,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.skipif(
    skip_condition, reason="FA3 DCP absorbed-MLA tests require Hopper (sm90)."
)
class TestFlashAttentionV3DCP(CustomTestCase):
    def test_decode_page_table_uses_rank_local_physical_pages(self):
        backend = object.__new__(FlashAttentionBackend)
        backend.page_size = 2
        backend.max_context_len = 16
        backend.req_to_token = torch.zeros(
            (2, backend.max_context_len), dtype=torch.int32, device="cuda"
        )
        # dcp_size=4, rank=2 owns global positions 2, 6, 10, ...
        # Each pair of local tokens forms one physical page.
        backend.req_to_token[0, 2] = 42
        backend.req_to_token[0, 10] = 82
        backend.req_to_token[1, 2] = 122
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        local_seq_lens = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
        page_table = torch.zeros((2, 2), dtype=torch.int32, device="cuda")

        parallel = SimpleNamespace(dcp_size=4, dcp_rank=2)
        with patch(
            "sglang.srt.layers.attention.flashattention_backend.get_parallel",
            return_value=parallel,
        ):
            backend._fill_mla_dcp_decode_page_table(
                page_table, req_pool_indices, local_seq_lens
            )

        expected = torch.tensor([[5, 10], [15, 0]], dtype=torch.int32, device="cuda")
        torch.testing.assert_close(page_table, expected)

    def test_gathered_extend_indices_are_unpacked_per_request(self):
        dcp_metadata = SimpleNamespace(
            dcp_kv_indptr=torch.tensor([0, 3, 5, 5], dtype=torch.int32, device="cuda"),
            dcp_kv_indices=torch.tensor(
                [8, 1, 4, 7, 3], dtype=torch.int32, device="cuda"
            ),
        )
        forward_batch = SimpleNamespace(
            attn_dcp_metadata=dcp_metadata,
            batch_size=3,
            seq_lens=torch.tensor([3, 2, 0], dtype=torch.int32, device="cuda"),
        )

        page_table = FlashAttentionBackend._build_mla_dcp_extend_page_table(
            forward_batch, max_seq_len_k=4
        )
        expected = torch.tensor(
            [[8, 1, 4, 0], [7, 3, 0, 0], [0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        torch.testing.assert_close(page_table, expected)

    def test_rope_mla(self):
        _check_cyclic_dcp_merge(only_qv=False)

    def test_nope_only_qv_mla(self):
        _check_cyclic_dcp_merge(only_qv=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
