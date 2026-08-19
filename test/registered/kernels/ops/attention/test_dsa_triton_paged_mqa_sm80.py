from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.kernels.ops.attention.dsa.torch_sparse_mla import torch_sparse_mla
from sglang.kernels.ops.attention.dsa.triton_paged_mqa_logits_sm80 import (
    triton_decode_e4m3fn,
    triton_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    torch_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="nightly", runner_config="1-gpu-large")

PAGE_SIZE = 64
HEAD_DIM = 128
PACKED_PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)
VALUE_DIM = 512
ROPE_DIM = 64


def _is_exact_sm80() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm80(), reason="SM80 Triton paged-MQA tests require NVIDIA A100"
)


def _python_e4m3fn(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return math.copysign(float("nan"), sign)
    if exponent == 0:
        return sign * mantissa * (2.0**-9)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def _pack_index_cache(keys: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    pages = keys.shape[0]
    packed = torch.empty(
        (pages, PACKED_PAGE_BYTES), dtype=torch.uint8, device=keys.device
    )
    key_bytes = PAGE_SIZE * HEAD_DIM
    packed[:, :key_bytes] = keys.contiguous().view(torch.uint8).reshape(pages, -1)
    packed[:, key_bytes:] = (
        scales.float().contiguous().view(torch.uint8).reshape(pages, -1)
    )
    return packed.view(pages, PAGE_SIZE, 1, HEAD_DIM + 4)


def _unpack_index_cache(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = packed.view(torch.uint8).reshape(packed.shape[0], -1)
    key_bytes = PAGE_SIZE * HEAD_DIM
    keys = (
        flat[:, :key_bytes]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(-1, PAGE_SIZE, HEAD_DIM)
        .float()
    )
    scales = flat[:, key_bytes:].contiguous().view(torch.float32)
    return keys, scales


def _independent_paged_mqa_reference(
    q_fp8: torch.Tensor,
    packed_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    keys, scales = _unpack_index_cache(packed_cache)
    result = torch.zeros(
        (q_fp8.shape[0], max_seq_len), dtype=torch.float32, device=q_fp8.device
    )
    num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    for query_idx in range(q_fp8.shape[0]):
        query = q_fp8[query_idx].float()
        seq_len = int(seq_lens[query_idx].cpu())
        for logical_page in range(num_pages):
            if logical_page >= page_table.shape[1]:
                continue
            physical_page = int(page_table[query_idx, logical_page].cpu())
            if physical_page < 0 or physical_page >= keys.shape[0]:
                continue
            start = logical_page * PAGE_SIZE
            end = min(start + PAGE_SIZE, seq_len, max_seq_len)
            if end <= start:
                continue
            qk = torch.matmul(query, keys[physical_page].T).relu()
            scores = torch.sum(qk * weights[query_idx, :, None], dim=0)
            scores *= scales[physical_page]
            result[query_idx, start:end] = scores[: end - start]
    return result


def _make_case(
    num_queries: int,
    num_heads: int,
    seq_len: int,
    max_seq_len: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    logical_pages = max(1, (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE)
    physical_pages = max(7, logical_pages + 3)
    keys = torch.randn(physical_pages, PAGE_SIZE, HEAD_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    scales = (
        torch.rand(physical_pages, PAGE_SIZE, dtype=torch.float32, device="cuda") + 0.25
    )
    q_fp8 = torch.randn(num_queries, num_heads, HEAD_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(num_queries, num_heads, dtype=torch.float32, device="cuda")

    base = (torch.arange(logical_pages, dtype=torch.int32) * 5 + 3) % physical_pages
    table = torch.stack(
        [torch.roll(base, shifts=query_idx) for query_idx in range(num_queries)]
    )
    if logical_pages >= 3:
        table[0, 2] = table[0, 0]
    if num_queries >= 2:
        table[1, -1] = -1
    if num_queries >= 3:
        table[2, min(1, logical_pages - 1)] = physical_pages + 11
    lengths = torch.tensor(
        [max(1, seq_len - (query_idx % 4)) for query_idx in range(num_queries)],
        dtype=torch.int32,
    )
    return (
        q_fp8,
        _pack_index_cache(keys, scales),
        weights,
        lengths.cuda(),
        table.cuda(),
    )


def _assert_three_way_close(
    q_fp8: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    *,
    atol: float = 2.0e-3,
    rtol: float = 2.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    triton_logits = triton_paged_mqa_logits(
        q_fp8, cache, weights, seq_lens, page_table, max_seq_len
    )
    torch_logits = torch_paged_mqa_logits(
        q_fp8, cache, weights, seq_lens, page_table, max_seq_len
    )
    reference = _independent_paged_mqa_reference(
        q_fp8, cache, weights, seq_lens, page_table, max_seq_len
    )
    torch.testing.assert_close(triton_logits, torch_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(triton_logits, reference, atol=atol, rtol=rtol)
    assert torch.isfinite(triton_logits).all()

    positions = torch.arange(max_seq_len, device="cuda")
    length_mask = positions[None, :] >= seq_lens[:, None]
    assert torch.count_nonzero(triton_logits.masked_select(length_mask)) == 0
    return triton_logits, torch_logits, reference


def test_device_e4m3fn_decoder_exhaustive_256_patterns():
    encoded = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    actual = triton_decode_e4m3fn(encoded).cpu()
    python_reference = torch.tensor([_python_e4m3fn(i) for i in range(256)])
    pytorch_reference = encoded.cpu().view(torch.float8_e4m3fn).float()

    finite = ~torch.isnan(python_reference)
    torch.testing.assert_close(actual[finite], python_reference[finite], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[finite], pytorch_reference[finite], rtol=0, atol=0
    )
    assert torch.isnan(actual[0x7F])
    assert torch.isnan(actual[0xFF])

    # Positive/negative zero, subnormal, normal, max finite, and both NaNs.
    assert actual[0x00] == 0 and not torch.signbit(actual[0x00])
    assert actual[0x80] == 0 and torch.signbit(actual[0x80])
    assert actual[0x01] == 2.0**-9
    assert actual[0x81] == -(2.0**-9)
    assert actual[0x08] == 2.0**-6
    assert actual[0x88] == -(2.0**-6)
    assert actual[0x7E] == 448.0
    assert actual[0xFE] == -448.0


@pytest.mark.parametrize(
    "num_queries,num_heads,seq_len,max_seq_len",
    [
        (1, 8, 1, 17),
        (2, 16, 63, 70),
        (8, 32, 64, 67),
        (32, 64, 65, 131),
        (2, 8, 4096, 4109),
    ],
)
def test_triton_paged_mqa_shape_boundary_matrix(
    num_queries: int, num_heads: int, seq_len: int, max_seq_len: int
):
    case = _make_case(num_queries, num_heads, seq_len, max_seq_len, seed=1000 + seq_len)
    _assert_three_way_close(*case, max_seq_len)


def test_triton_paged_mqa_invalid_noncontiguous_and_repeated_pages():
    q, cache, weights, _, _ = _make_case(4, 16, 193, 259, seed=31415)
    page_table = torch.tensor(
        [
            [5, 1, 5, -1, 0],
            [3, 3, 0, 2, 1],
            [4, 2, cache.shape[0] + 9, 1, 0],
            [6, 0, 4, 2, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([193, 65, 130, 1], dtype=torch.int32, device="cuda")
    actual, _, _ = _assert_three_way_close(q, cache, weights, seq_lens, page_table, 259)
    assert torch.count_nonzero(actual[0, 192:256]) == 0
    assert torch.count_nonzero(actual[2, 128:192]) == 0
    assert torch.count_nonzero(actual[3, 1:]) == 0


def test_triton_paged_mqa_special_finite_values_and_mixed_signs():
    num_queries, num_heads, pages = 2, 8, 2
    patterns = torch.tensor(
        [0x00, 0x80, 0x01, 0x81, 0x08, 0x88, 0x7E, 0xFE, 0x38, 0xB8],
        dtype=torch.uint8,
        device="cuda",
    )
    q_bytes = patterns.repeat(
        math.ceil(num_queries * num_heads * HEAD_DIM / patterns.numel())
    )[: num_queries * num_heads * HEAD_DIM].contiguous()
    q_fp8 = q_bytes.view(torch.float8_e4m3fn).reshape(num_queries, num_heads, HEAD_DIM)
    key_bytes = (
        patterns.flip(0)
        .repeat(math.ceil(pages * PAGE_SIZE * HEAD_DIM / patterns.numel()))[
            : pages * PAGE_SIZE * HEAD_DIM
        ]
        .contiguous()
    )
    keys = key_bytes.view(torch.float8_e4m3fn).reshape(pages, PAGE_SIZE, HEAD_DIM)
    scales = torch.full((pages, PAGE_SIZE), 1.0e-3, device="cuda")
    cache = _pack_index_cache(keys, scales)
    weights = (
        torch.arange(num_heads, dtype=torch.float32, device="cuda")
        .remainder(2)
        .mul_(2)
        .sub_(1)
        .mul_(1.0e-3)
        .repeat(num_queries, 1)
    )
    seq_lens = torch.tensor([65, 127], dtype=torch.int32, device="cuda")
    page_table = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32, device="cuda")
    _assert_three_way_close(
        q_fp8,
        cache,
        weights,
        seq_lens,
        page_table,
        127,
        atol=1.0,
        rtol=1.0e-5,
    )


def test_triton_paged_mqa_zero_inputs_and_non_tie_topk():
    num_heads, pages, max_seq_len = 16, 2, 128
    zero_q = torch.zeros(
        2, num_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    zero_keys = torch.zeros(
        pages, PAGE_SIZE, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    scales = torch.ones(pages, PAGE_SIZE, device="cuda")
    cache = _pack_index_cache(zero_keys, scales)
    weights = torch.randn(2, num_heads, device="cuda")
    lengths = torch.tensor([128, 65], dtype=torch.int32, device="cuda")
    table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device="cuda")
    logits = triton_paged_mqa_logits(
        zero_q, cache, weights, lengths, table, max_seq_len
    )
    assert torch.count_nonzero(logits) == 0
    assert torch.isfinite(logits).all()

    one_q = torch.ones_like(zero_q)
    one_keys = torch.ones_like(zero_keys)
    unique_scales = (
        1.0
        + torch.arange(pages * PAGE_SIZE, dtype=torch.float32, device="cuda").reshape(
            pages, PAGE_SIZE
        )
        / 1000.0
    )
    cache = _pack_index_cache(one_keys, unique_scales)
    positive_weights = torch.ones_like(weights)
    triton_logits, torch_logits, reference = _assert_three_way_close(
        one_q,
        cache,
        positive_weights,
        lengths,
        table,
        max_seq_len,
        atol=0,
        rtol=0,
    )
    topk = 16
    triton_topk = DSATopKBackend.TORCH.topk_func(triton_logits, lengths, topk)
    torch_topk = DSATopKBackend.TORCH.topk_func(torch_logits, lengths, topk)
    reference_topk = DSATopKBackend.TORCH.topk_func(reference, lengths, topk)
    torch.testing.assert_close(triton_topk, torch_topk, rtol=0, atol=0)
    torch.testing.assert_close(triton_topk, reference_topk, rtol=0, atol=0)


def test_e4m3fn_nan_patterns_propagate_in_runtime_kernel():
    q = torch.ones(1, 8, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    q.view(torch.uint8)[0, 0, 0] = 0x7F
    keys = torch.ones(1, PAGE_SIZE, HEAD_DIM, device="cuda").to(torch.float8_e4m3fn)
    cache = _pack_index_cache(keys, torch.ones(1, PAGE_SIZE, device="cuda"))
    weights = torch.ones(1, 8, device="cuda")
    lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
    table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    logits = triton_paged_mqa_logits(q, cache, weights, lengths, table, 65)
    assert torch.isnan(logits[0, 0])
    assert torch.count_nonzero(logits[0, 1:]) == 0

    q.view(torch.uint8)[0, 0, 0] = 0x38
    cache.view(torch.uint8).reshape(1, -1)[0, 0] = 0xFF
    logits = triton_paged_mqa_logits(q, cache, weights, lengths, table, 1)
    assert torch.isnan(logits[0, 0])


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


def _make_dispatch_indexer(num_heads: int) -> Indexer:
    indexer = object.__new__(Indexer)
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.TRITON
    indexer.sm_count = 108
    indexer.n_heads = num_heads
    indexer.index_topk = 8
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    return indexer


def test_triton_dispatch_padding_rows_and_deepgemm_bypass():
    q, cache, weights, lengths, table = _make_case(2, 8, 65, 128, seed=2718)
    q = torch.cat([q, torch.zeros_like(q)], dim=0)
    weights = torch.cat([weights, torch.zeros_like(weights)], dim=0).unsqueeze(-1)
    pool = SimpleNamespace(
        page_size=64,
        get_index_k_with_scale_buffer=lambda layer_id: cache.view(cache.shape[0], -1),
    )
    metadata = _DispatchMetadata(lengths, table)
    indexer = _make_dispatch_indexer(8)
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
    ):
        result = indexer._get_topk_paged(
            SimpleNamespace(forward_mode=ForwardMode.DECODE),
            0,
            q,
            weights,
            metadata,
        )

    assert result.shape == (4, 8)
    assert torch.all(result[:2] >= 0)
    assert torch.all(result[2:] == -1)
    mock_deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
    mock_deep_gemm.fp8_paged_mqa_logits.assert_not_called()


@pytest.mark.parametrize(
    "forward_mode", [ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2]
)
def test_triton_dispatch_rejects_unsupported_speculative_modes(forward_mode):
    q, cache, weights, lengths, table = _make_case(2, 8, 65, 128, seed=1618)
    pool = SimpleNamespace(
        page_size=64,
        get_index_k_with_scale_buffer=lambda layer_id: cache.view(cache.shape[0], -1),
    )
    indexer = _make_dispatch_indexer(8)
    metadata = _DispatchMetadata(lengths, table)
    with (
        patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer.get_token_to_kv_pool",
            return_value=pool,
        ),
        pytest.raises(RuntimeError, match="only ordinary eager decode"),
    ):
        indexer._get_topk_paged(
            SimpleNamespace(forward_mode=forward_mode),
            0,
            q,
            weights.unsqueeze(-1),
            metadata,
        )


def test_triton_metadata_build_and_refresh_never_call_deepgemm():
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.TRITON
    mock_deep_gemm = MagicMock()
    with (
        patch(
            "sglang.srt.layers.attention.dsa_backend.deep_gemm",
            mock_deep_gemm,
            create=True,
        ),
        patch("sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=True),
    ):
        schedule = backend._build_paged_mqa_schedule_metadata(
            torch.tensor([[65]], dtype=torch.int32, device="cuda")
        )
        backend._refresh_paged_mqa_schedule_metadata(
            SimpleNamespace(paged_mqa_schedule_metadata=None),
            torch.tensor([[65]], dtype=torch.int32, device="cuda"),
        )
    assert schedule is None
    mock_deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()


def test_backend_resolution_is_explicit_sm80_only_and_no_metadata():
    backend = DSAPagedMQALogitsBackend.resolve("triton")
    assert backend.is_triton()
    assert not backend.uses_deepgemm_metadata()
    with patch(
        "sglang.srt.layers.attention.dsa.paged_mqa_logits_backend.get_device_capability",
        return_value=(9, 0),
    ), pytest.raises(ValueError, match="requires NVIDIA SM80"):
        DSAPagedMQALogitsBackend.resolve("triton")
    with patch(
        "sglang.srt.layers.attention.dsa.paged_mqa_logits_backend.is_hip",
        return_value=True,
    ), pytest.raises(ValueError, match="ROCm"):
        DSAPagedMQALogitsBackend.resolve("triton")


def test_triton_indexer_torch_attention_32_step_eager_decode_harness():
    torch.manual_seed(20260819)
    num_queries, num_heads, pages = 2, 16, 2
    keys = torch.randn(pages, PAGE_SIZE, HEAD_DIM, device="cuda").to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pages, PAGE_SIZE, device="cuda") + 0.5
    cache = _pack_index_cache(keys, scales)
    page_table = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32, device="cuda")
    attention_kv = torch.randn(
        pages * PAGE_SIZE,
        1,
        VALUE_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    for step in range(32):
        lengths = torch.tensor([65 + step, 64 + step], dtype=torch.int32, device="cuda")
        q = torch.randn(num_queries, num_heads, HEAD_DIM, device="cuda").to(
            torch.float8_e4m3fn
        )
        weights = torch.randn(num_queries, num_heads, device="cuda")
        triton_logits = triton_paged_mqa_logits(
            q, cache, weights, lengths, page_table, pages * PAGE_SIZE
        )
        torch_logits = torch_paged_mqa_logits(
            q, cache, weights, lengths, page_table, pages * PAGE_SIZE
        )
        torch.testing.assert_close(
            triton_logits, torch_logits, atol=2.0e-3, rtol=2.0e-5
        )

        triton_topk = DSATopKBackend.TORCH.topk_func(triton_logits, lengths, 16)
        torch_topk = DSATopKBackend.TORCH.topk_func(torch_logits, lengths, 16)
        torch.testing.assert_close(triton_topk, torch_topk, rtol=0, atol=0)
        assert torch.all((triton_topk >= 0) & (triton_topk < lengths[:, None]))

        q_nope = torch.randn(
            num_queries, num_heads, VALUE_DIM, dtype=torch.bfloat16, device="cuda"
        )
        q_rope = torch.randn(
            num_queries, num_heads, ROPE_DIM, dtype=torch.bfloat16, device="cuda"
        )
        output = torch_sparse_mla(
            q_nope,
            q_rope,
            attention_kv,
            triton_topk,
            0.125,
            query_chunk_size=1,
            topk_chunk_size=4,
        )
        assert torch.isfinite(output).all()
