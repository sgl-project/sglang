from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.kernels.ops.attention.dsa.torch_sparse_mla import torch_sparse_mla
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    torch_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-b-test-cpu")

PAGE_SIZE = 64
INDEX_DIM = 128
VALUE_DIM = 512
ROPE_DIM = 64


def _pack_index_cache(keys: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Test-only packer, independent of the runtime/DSV4 cache helpers."""

    num_pages = keys.shape[0]
    packed = torch.zeros(
        (num_pages, PAGE_SIZE * (INDEX_DIM + 4)),
        dtype=torch.uint8,
        device=keys.device,
    )
    key_end = PAGE_SIZE * INDEX_DIM
    packed[:, :key_end] = keys.contiguous().view(torch.uint8).reshape(num_pages, -1)
    packed[:, key_end:] = (
        scales.float().contiguous().view(torch.uint8).reshape(num_pages, -1)
    )
    return packed.view(num_pages, PAGE_SIZE, 1, INDEX_DIM + 4)


def _unpack_index_cache(
    packed: torch.Tensor, fp8_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-only decoder used only by the mathematical reference."""

    flat = packed.view(torch.uint8).reshape(packed.shape[0], -1)
    key_end = PAGE_SIZE * INDEX_DIM
    keys = (
        flat[:, :key_end]
        .contiguous()
        .view(fp8_dtype)
        .reshape(-1, PAGE_SIZE, INDEX_DIM)
        .float()
    )
    scales = flat[:, key_end:].contiguous().view(torch.float32).reshape(-1, PAGE_SIZE)
    return keys, scales


def _paged_mqa_reference(
    q_fp8: torch.Tensor,
    packed_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Independent, deliberately direct Torch definition of paged-MQA."""

    keys, scales = _unpack_index_cache(packed_cache, q_fp8.dtype)
    result = torch.zeros(
        (q_fp8.shape[0], max_seq_len), dtype=torch.float32, device=q_fp8.device
    )
    for query_idx in range(q_fp8.shape[0]):
        length = int(seq_lens[query_idx])
        query = q_fp8[query_idx].float()
        for logical_idx in range(min(length, max_seq_len)):
            logical_page, page_offset = divmod(logical_idx, PAGE_SIZE)
            if logical_page >= page_table.shape[1]:
                continue
            physical_page = int(page_table[query_idx, logical_page])
            if physical_page < 0 or physical_page >= keys.shape[0]:
                continue
            per_head = torch.einsum(
                "hd,hd->h", query, keys[physical_page, page_offset].expand_as(query)
            ).relu()
            result[query_idx, logical_idx] = (
                per_head * weights[query_idx]
            ).sum() * scales[physical_page, page_offset]
    return result


def _sparse_mla_reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Independent per-query/per-head sparse attention definition."""

    output = torch.zeros_like(q_nope)
    kv = kv_cache[:, 0].float()
    for query_idx in range(q_nope.shape[0]):
        valid = indices[query_idx]
        valid = valid[(valid >= 0) & (valid < kv.shape[0])].to(torch.long)
        if valid.numel() == 0:
            continue
        selected = kv[valid]
        for head_idx in range(q_nope.shape[1]):
            scores = (
                selected[:, :VALUE_DIM] @ q_nope[query_idx, head_idx].float()
                + selected[:, VALUE_DIM:] @ q_rope[query_idx, head_idx].float()
            ) * sm_scale
            probability = torch.softmax(scores, dim=0)
            output[query_idx, head_idx] = (probability @ selected[:, :VALUE_DIM]).to(
                output.dtype
            )
    return output


def _make_paged_case(
    seq_lens: list[int],
    num_heads: int,
    page_table: torch.Tensor | None = None,
    *,
    seed: int = 20260818,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    num_queries = len(seq_lens)
    max_pages = max(1, (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE)
    if page_table is None:
        page_table = torch.arange(max_pages, dtype=torch.int32).repeat(num_queries, 1)
    num_physical_pages = max(1, int(page_table.clamp_min(0).max()) + 1)
    keys = torch.randn(
        num_physical_pages,
        PAGE_SIZE,
        INDEX_DIM,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    scales = (
        torch.rand(
            num_physical_pages, PAGE_SIZE, generator=generator, dtype=torch.float32
        )
        + 0.25
    )
    q_fp8 = torch.randn(num_queries, num_heads, INDEX_DIM, generator=generator).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(
        num_queries, num_heads, generator=generator, dtype=torch.float32
    )
    return (
        q_fp8,
        _pack_index_cache(keys, scales),
        weights,
        torch.tensor(seq_lens, dtype=torch.int32),
        page_table,
    )


@pytest.mark.parametrize("seq_len", [1, 63, 64, 65, 4096])
def test_torch_paged_mqa_matches_independent_reference(seq_len: int):
    q, cache, weights, lengths, table = _make_paged_case([seq_len], 2)
    max_seq_len = table.shape[1] * PAGE_SIZE

    actual = torch_paged_mqa_logits(
        q,
        cache,
        weights,
        lengths,
        table,
        max_seq_len,
        query_chunk_size=1,
        page_chunk_size=3,
    )
    expected = _paged_mqa_reference(q, cache, weights, lengths, table, max_seq_len)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
    assert torch.count_nonzero(actual[:, seq_len:]) == 0


def test_torch_paged_mqa_noncontiguous_repeated_pages_batches_and_heads():
    table = torch.tensor(
        [
            [5, 1, 5, -1],
            [3, 3, 0, -1],
            [4, 2, -1, -1],
        ],
        dtype=torch.int32,
    )
    q, cache, weights, lengths, table = _make_paged_case([193, 130, 65], 3, table)
    max_seq_len = table.shape[1] * PAGE_SIZE

    actual = torch_paged_mqa_logits(
        q,
        cache,
        weights,
        lengths,
        table,
        max_seq_len,
        query_chunk_size=2,
        page_chunk_size=1,
    )
    expected = _paged_mqa_reference(q, cache, weights, lengths, table, max_seq_len)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
    # Row 0 declares 193 tokens but its fourth logical page is invalid.
    assert torch.count_nonzero(actual[0, 192:]) == 0


def test_zero_invalid_logits_are_safe_for_length_masked_topk_non_tie():
    q, cache, weights, lengths, table = _make_paged_case([65, 63], 2, seed=7)
    logits = torch_paged_mqa_logits(
        q, cache, weights, lengths, table, table.shape[1] * PAGE_SIZE
    )
    reference = _paged_mqa_reference(
        q, cache, weights, lengths, table, table.shape[1] * PAGE_SIZE
    )

    topk = 8
    actual_topk = DSATopKBackend.TORCH.topk_func(logits, lengths, topk)
    expected_topk = DSATopKBackend.TORCH.topk_func(reference, lengths, topk)
    torch.testing.assert_close(actual_topk, expected_topk, rtol=0, atol=0)
    assert torch.all(actual_topk < lengths.unsqueeze(1))
    assert torch.all(logits[1, 63:] == 0.0)


@pytest.mark.parametrize("num_heads", [1, 3])
def test_torch_sparse_mla_matches_independent_reference(num_heads: int):
    generator = torch.Generator().manual_seed(314159)
    q_nope = torch.randn(3, num_heads, VALUE_DIM, generator=generator).to(
        torch.bfloat16
    )
    q_rope = torch.randn(3, num_heads, ROPE_DIM, generator=generator).to(torch.bfloat16)
    kv = torch.randn(11, 1, VALUE_DIM + ROPE_DIM, generator=generator).to(
        torch.bfloat16
    )
    indices = torch.tensor(
        [[7, 2, 7, -1, 1, 9], [0, 10, 4, 3, -1, -1], [-1] * 6],
        dtype=torch.int32,
    )
    scale = 0.125

    actual = torch_sparse_mla(
        q_nope,
        q_rope,
        kv,
        indices,
        scale,
        query_chunk_size=2,
        topk_chunk_size=2,
    )
    expected = _sparse_mla_reference(q_nope, q_rope, kv, indices, scale)

    torch.testing.assert_close(actual.float(), expected.float(), atol=8e-3, rtol=8e-3)
    assert torch.count_nonzero(actual[2]) == 0


def test_torch_backends_prefill_and_32_decode_step_harness():
    generator = torch.Generator().manual_seed(271828)
    kv = torch.randn(96, 1, VALUE_DIM + ROPE_DIM, generator=generator).to(
        torch.bfloat16
    )

    # Ordinary prefill sparse attention with multiple query rows.
    q_nope = torch.randn(5, 2, VALUE_DIM, generator=generator).to(torch.bfloat16)
    q_rope = torch.randn(5, 2, ROPE_DIM, generator=generator).to(torch.bfloat16)
    prefill_indices = torch.randint(0, 96, (5, 24), generator=generator).to(torch.int32)
    prefill_indices[:, -3:] = -1
    prefill_actual = torch_sparse_mla(
        q_nope,
        q_rope,
        kv,
        prefill_indices,
        0.125,
        query_chunk_size=2,
        topk_chunk_size=5,
    )
    prefill_expected = _sparse_mla_reference(q_nope, q_rope, kv, prefill_indices, 0.125)
    torch.testing.assert_close(
        prefill_actual.float(), prefill_expected.float(), atol=8e-3, rtol=8e-3
    )

    # Consecutive eager decode steps exercise paged logits, Top-K and sparse MLA.
    index_keys = torch.randn(2, PAGE_SIZE, INDEX_DIM, generator=generator).to(
        torch.float8_e4m3fn
    )
    index_scales = torch.rand(2, PAGE_SIZE, generator=generator) + 0.5
    packed = _pack_index_cache(index_keys, index_scales)
    page_table = torch.tensor([[1, 0]], dtype=torch.int32)
    for step in range(32):
        seq_len = 65 + step
        query = torch.randn(1, 2, INDEX_DIM, generator=generator).to(
            torch.float8_e4m3fn
        )
        weights = torch.randn(1, 2, generator=generator)
        lengths = torch.tensor([seq_len], dtype=torch.int32)
        logits = torch_paged_mqa_logits(
            query, packed, weights, lengths, page_table, 2 * PAGE_SIZE
        )
        topk = DSATopKBackend.TORCH.topk_func(logits, lengths, 16)
        assert torch.all((topk >= 0) & (topk < seq_len))

        decode_q_nope = torch.randn(1, 2, VALUE_DIM, generator=generator).to(
            torch.bfloat16
        )
        decode_q_rope = torch.randn(1, 2, ROPE_DIM, generator=generator).to(
            torch.bfloat16
        )
        actual = torch_sparse_mla(
            decode_q_nope,
            decode_q_rope,
            kv,
            topk,
            0.125,
            query_chunk_size=1,
            topk_chunk_size=4,
        )
        expected = _sparse_mla_reference(decode_q_nope, decode_q_rope, kv, topk, 0.125)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=8e-3, rtol=8e-3
        )
        assert torch.isfinite(actual).all()


class _DispatchMetadata:
    def __init__(self, lengths: torch.Tensor, page_table: torch.Tensor):
        self.lengths = lengths
        self.page_table = page_table

    def get_page_table_64(self):
        return self.page_table

    def get_seqlens_int32(self):
        return self.lengths

    def get_seqlens_expanded(self):
        return self.lengths

    def get_dsa_extend_len_cpu(self):
        return [1] * self.lengths.shape[0]

    def topk_transform(self, logits, topk):
        return DSATopKBackend.TORCH.topk_func(logits, self.lengths, topk)


def test_torch_indexer_dispatch_never_calls_deepgemm_paged_apis():
    q, cache, weights, lengths, table = _make_paged_case([65, 63], 2)
    pool = SimpleNamespace(
        page_size=64,
        get_index_k_with_scale_buffer=lambda layer_id: cache.view(cache.shape[0], -1),
    )
    indexer = object.__new__(Indexer)
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.TORCH
    indexer.sm_count = 108
    indexer.n_heads = 2
    indexer.index_topk = 8
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    metadata = _DispatchMetadata(lengths, table)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    mock_deep_gemm = MagicMock()

    with (
        patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer.get_token_to_kv_pool",
            return_value=pool,
        ),
        patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer.deep_gemm",
            mock_deep_gemm,
            create=True,
        ),
        patch("sglang.srt.layers.attention.dsa.dsa_indexer._is_cuda", True),
    ):
        result = indexer._get_topk_paged(
            forward_batch, 0, q, weights.unsqueeze(-1), metadata
        )

    assert result.shape == (2, 8)
    mock_deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
    mock_deep_gemm.fp8_paged_mqa_logits.assert_not_called()


def test_torch_metadata_refresh_and_attention_dispatch_bypass_dg_and_fa3():
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.TORCH
    mock_deep_gemm = MagicMock()
    with patch(
        "sglang.srt.layers.attention.dsa_backend.deep_gemm",
        mock_deep_gemm,
        create=True,
    ), patch("sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=True):
        schedule = backend._build_paged_mqa_schedule_metadata(
            torch.tensor([[65]], dtype=torch.int32)
        )
        backend._refresh_paged_mqa_schedule_metadata(
            SimpleNamespace(paged_mqa_schedule_metadata=None),
            torch.tensor([[65]], dtype=torch.int32),
        )
    assert schedule is None
    mock_deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()

    generator = torch.Generator().manual_seed(99)
    backend.dsa_decode_impl = "torch"
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.use_fused_topk = True
    backend.hisparse_coordinator = None
    backend.forward_metadata = SimpleNamespace()
    kv = torch.randn(8, 1, VALUE_DIM + ROPE_DIM, generator=generator).to(torch.bfloat16)
    backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda layer_id: kv)
    layer = SimpleNamespace(
        is_cross_attention=False,
        tp_q_head_num=2,
        v_head_dim=VALUE_DIM,
        head_dim=VALUE_DIM + ROPE_DIM,
        scaling=0.125,
        layer_id=0,
    )
    forward_batch = SimpleNamespace(batch_size=1)
    q_nope = torch.randn(1, 2, VALUE_DIM, generator=generator).to(torch.bfloat16)
    q_rope = torch.randn(1, 2, ROPE_DIM, generator=generator).to(torch.bfloat16)
    topk = torch.tensor([[7, 2, -1]], dtype=torch.int32)

    with patch.object(backend, "_forward_fa3", side_effect=AssertionError) as mock_fa3:
        output = backend.forward_decode(
            q_nope,
            None,
            None,
            layer,
            forward_batch,
            save_kv_cache=False,
            q_rope=q_rope,
            topk_indices=topk,
        )

    assert output.shape == q_nope.shape
    mock_fa3.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="SM80 CUDA harness requires an NVIDIA A100-class GPU",
)
def test_sm80_cuda_prefill_and_32_decode_step_kernel_harness():
    # Cross-device reference for a page-boundary/non-contiguous case.
    table = torch.tensor([[3, 1, 3], [2, 0, -1]], dtype=torch.int32)
    cpu_case = _make_paged_case([129, 65], 3, table, seed=1234)
    expected = _paged_mqa_reference(*cpu_case, max_seq_len=table.shape[1] * PAGE_SIZE)
    q, cache, weights, lengths, table = (tensor.cuda() for tensor in cpu_case)
    actual = torch_paged_mqa_logits(
        q, cache, weights, lengths, table, table.shape[1] * PAGE_SIZE
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-3)

    # The planned CLI keeps the default SGL Top-K backend. On SM80 the Torch
    # DSA argument resolver disables cluster Top-K v2, so exercise the real
    # legacy fused page-table transform instead of merely checking the flag.
    legacy_logits = torch.randn(2, 4096, device="cuda", dtype=torch.float32)
    legacy_logits += torch.arange(4096, device="cuda", dtype=torch.float32) * 1e-4
    legacy_lengths = torch.tensor([3073, 2500], device="cuda", dtype=torch.int32)
    page_table_1 = torch.arange(4096, device="cuda", dtype=torch.int32).repeat(2, 1)
    page_table_1[1] += 8192
    expected_logical = DSATopKBackend.TORCH.topk_func(
        legacy_logits, legacy_lengths, 2048
    )
    expected_physical = torch.gather(page_table_1, 1, expected_logical.to(torch.long))
    with (
        envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        envs.SGLANG_DSA_FUSE_TOPK.override(True),
    ):
        actual_physical = DSATopKBackend.SGL_KERNEL.topk_transform(
            logits=legacy_logits,
            lengths=legacy_lengths,
            topk=2048,
            topk_transform_method=TopkTransformMethod.PAGED,
            attn_metadata=SimpleNamespace(
                page_table_1=page_table_1,
                real_page_table=table,
            ),
            cu_seqlens_q_topk=torch.arange(3, device="cuda", dtype=torch.int32),
        )
    torch.testing.assert_close(
        torch.sort(actual_physical, dim=-1).values,
        torch.sort(expected_physical, dim=-1).values,
        rtol=0,
        atol=0,
    )

    # Exercise the 4K page traversal without constructing a large reference
    # temporary on the GPU (the CPU test above fixes its exact math).
    long_case = _make_paged_case([4096], 2, seed=4321)
    q, cache, weights, lengths, table = (tensor.cuda() for tensor in long_case)
    long_logits = torch_paged_mqa_logits(q, cache, weights, lengths, table, 4096)
    assert long_logits.shape == (1, 4096)
    assert torch.isfinite(long_logits).all()

    generator = torch.Generator().manual_seed(5678)
    kv_cpu = torch.randn(128, 1, VALUE_DIM + ROPE_DIM, generator=generator).to(
        torch.bfloat16
    )
    kv = kv_cpu.cuda()

    # Ordinary sparse-MLA prefill.
    prefill_qn_cpu = torch.randn(4, 2, VALUE_DIM, generator=generator).to(
        torch.bfloat16
    )
    prefill_qr_cpu = torch.randn(4, 2, ROPE_DIM, generator=generator).to(torch.bfloat16)
    prefill_indices_cpu = torch.randint(
        0, 128, (4, 19), generator=generator, dtype=torch.int32
    )
    prefill_indices_cpu[:, -2:] = -1
    prefill_expected = _sparse_mla_reference(
        prefill_qn_cpu,
        prefill_qr_cpu,
        kv_cpu,
        prefill_indices_cpu,
        0.125,
    )
    prefill_actual = torch_sparse_mla(
        prefill_qn_cpu.cuda(),
        prefill_qr_cpu.cuda(),
        kv,
        prefill_indices_cpu.cuda(),
        0.125,
        query_chunk_size=2,
        topk_chunk_size=5,
    )
    torch.testing.assert_close(
        prefill_actual.cpu().float(),
        prefill_expected.float(),
        atol=2e-2,
        rtol=2e-2,
    )

    # At least 32 consecutive eager decode invocations on SM80.
    for step in range(32):
        qn = torch.randn(1, 2, VALUE_DIM, generator=generator).to(torch.bfloat16)
        qr = torch.randn(1, 2, ROPE_DIM, generator=generator).to(torch.bfloat16)
        decode_indices = torch.arange(step + 1, dtype=torch.int32).unsqueeze(0)
        decode_indices = torch.nn.functional.pad(
            decode_indices, (0, 32 - decode_indices.shape[1]), value=-1
        )
        output = torch_sparse_mla(
            qn.cuda(),
            qr.cuda(),
            kv,
            decode_indices.cuda(),
            0.125,
            query_chunk_size=1,
            topk_chunk_size=7,
        )
        assert output.shape == (1, 2, VALUE_DIM)
        assert torch.isfinite(output).all()
    torch.cuda.synchronize()
