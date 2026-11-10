import pytest
import torch
from sglang.srt.layers.attention.tilelang_fa_v100 import _paged_adapter
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_paged_verify import (
    VERIFY_MIN_TOKENS_PER_SPLIT,
    _verify_min_tokens_per_split,
)
from sglang.kernels.ops.attention.decode_attention import (
    decode_attention_fwd,
)
from sglang.kernels.ops.attention.extend_attention import (
    extend_attention_fwd,
)
from sglang.kernels.ops.attention.fp8_sm70 import (
    dequantize_paged_kv_e4m3_sm70,
    store_paged_extend_kv_fp16_sm70,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 FP8 KV scratch kernels require an NVIDIA V100",
)


def test_verify_split_granularity_defaults_to_v100_occupancy_tuning(monkeypatch):
    monkeypatch.delenv("SGLANG_V100_VERIFY_TOKENS_PER_SPLIT", raising=False)

    assert VERIFY_MIN_TOKENS_PER_SPLIT == 128
    assert _verify_min_tokens_per_split() == 128


def test_d256_gather_policy_is_bounded_to_measured_shape(monkeypatch):
    monkeypatch.delenv("SGLANG_V100_PREFILL_D256_GATHER", raising=False)
    shape = dict(
        batch=1,
        heads=6,
        heads_kv=1,
        dim=256,
        num_tokens=4096,
        max_seq_len=8192,
        causal=True,
        sliding_window_size=-1,
        fp8_kv=False,
        fp16=True,
    )

    assert _paged_adapter._should_use_d256_gather(**shape)
    assert not _paged_adapter._should_use_d256_gather(**{**shape, "num_tokens": 3919})
    assert _paged_adapter._should_use_d256_gather(
        **{
            **shape,
            "num_tokens": 4000,
            "max_seq_len": 4000,
            "logical_dense_kv": True,
        }
    )
    assert _paged_adapter._should_use_d256_gather(
        **{
            **shape,
            "num_tokens": 15680,
            "max_seq_len": 15680,
            "logical_dense_kv": True,
        }
    )
    assert _paged_adapter._should_use_d256_gather(
        **{
            **shape,
            "num_tokens": 15681,
            "max_seq_len": 15681,
            "logical_dense_kv": True,
        }
    )
    assert not _paged_adapter._should_use_d256_gather(
        **{**shape, "sliding_window_size": 4096}
    )
    assert not _paged_adapter._should_use_d256_gather(**{**shape, "max_seq_len": 8191})
    assert _paged_adapter._should_use_d256_gather(
        **{**shape, "max_seq_len": 4096, "logical_dense_kv": True}
    )
    assert _paged_adapter._should_use_d256_gather(
        **{
            **shape,
            "num_tokens": 64,
            "max_seq_len": 32768,
            "logical_dense_kv": True,
        }
    )
    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_LOGICAL_TAIL", "0")
    assert not _paged_adapter._should_use_d256_gather(
        **{
            **shape,
            "num_tokens": 64,
            "max_seq_len": 32768,
            "logical_dense_kv": True,
        }
    )
    monkeypatch.delenv("SGLANG_V100_PREFILL_D256_LOGICAL_TAIL")
    assert not _paged_adapter._should_use_d256_gather(
        **{
            **shape,
            "num_tokens": 63,
            "max_seq_len": 32768,
            "logical_dense_kv": True,
        }
    )
    assert not _paged_adapter._should_use_d256_gather(
        **{**shape, "num_tokens": 64, "max_seq_len": 32768}
    )


@pytest.mark.parametrize(
    ("tokens", "expected_splits"),
    [
        (64, 64),
        (65, 32),
        (128, 32),
        (256, 16),
        (512, 8),
        (1024, 4),
        (2048, 2),
        (2049, 1),
    ],
)
def test_d256_logical_tail_split_policy(tokens, expected_splits):
    assert (
        _paged_adapter._d256_tail_split_kv(
            num_tokens=tokens,
            max_seq_len=245760,
            logical_dense_kv=True,
        )
        == expected_splits
    )
    assert (
        _paged_adapter._d256_tail_split_kv(
            num_tokens=tokens,
            max_seq_len=245760,
            logical_dense_kv=False,
        )
        == 1
    )


def test_d256_gathered_dense_matches_shuffled_paged_attention(monkeypatch):
    torch.manual_seed(29)
    device = "cuda:0"
    page_size = 16
    seq_len = 8192
    num_tokens = 4096
    num_pages = seq_len // page_size
    q = torch.randn(num_tokens, 6, 256, dtype=torch.float16, device=device).mul_(0.1)
    k_cache = torch.randn(
        num_pages, page_size, 1, 256, dtype=torch.float16, device=device
    ).mul_(0.1)
    v_cache = torch.randn_like(k_cache).mul_(0.1)
    block_table = torch.randperm(num_pages, dtype=torch.int32, device=device).view(
        1, -1
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    prefix_kv_lens = torch.tensor(
        [seq_len - num_tokens], dtype=torch.int32, device=device
    )
    args = (
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        prefix_kv_lens,
    )

    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_GATHER", "0")
    paged, _ = _paged_adapter.paged_forward(*args, max_seq_len_hint=seq_len)
    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_GATHER", "1")
    gathered, _ = _paged_adapter.paged_forward(*args, max_seq_len_hint=seq_len)

    # The native dense and paged TileLang kernels use different,
    # mathematically equivalent FP32 reduction trees before the final FP16
    # narrow.
    torch.testing.assert_close(gathered, paged, rtol=1e-3, atol=5e-6)


def test_d256_full_prompt_tail_padding_matches_paged_attention(monkeypatch):
    """A 4000-token request should use the exact D256 kernel via suffix pad."""
    torch.manual_seed(47)
    device = "cuda:0"
    page_size = 16
    seq_len = num_tokens = 4000
    q = torch.randn(num_tokens, 6, 256, dtype=torch.float16, device=device).mul_(0.1)
    k_cache = torch.randn(
        seq_len // page_size,
        page_size,
        1,
        256,
        dtype=torch.float16,
        device=device,
    ).mul_(0.1)
    v_cache = torch.randn_like(k_cache).mul_(0.1)
    block_table = torch.arange(
        seq_len // page_size, dtype=torch.int32, device=device
    ).view(1, -1)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    prefix_kv_lens = torch.tensor([0], dtype=torch.int32, device=device)
    args = (
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        prefix_kv_lens,
    )

    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_GATHER", "0")
    paged, _ = _paged_adapter.paged_forward(*args, max_seq_len_hint=seq_len)
    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_GATHER", "1")
    exact, _ = _paged_adapter.paged_forward(
        *args,
        max_seq_len_hint=seq_len,
        logical_dense_kv=True,
    )

    torch.testing.assert_close(exact, paged, rtol=1e-3, atol=5e-5)


def test_d256_logical_long_tail_auto_split_matches_unsplit(monkeypatch):
    torch.manual_seed(53)
    page_size = 16
    seq_len = 32768
    num_tokens = 64
    num_pages = seq_len // page_size
    q = torch.randn(num_tokens, 6, 256, dtype=torch.float16, device="cuda").mul_(0.1)
    k_cache = torch.randn(
        num_pages, page_size, 1, 256, dtype=torch.float16, device="cuda"
    ).mul_(0.1)
    v_cache = torch.randn_like(k_cache).mul_(0.1)
    block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").view(1, -1)
    args = (
        q,
        k_cache,
        v_cache,
        block_table,
        torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda"),
        torch.tensor([seq_len - num_tokens], dtype=torch.int32, device="cuda"),
    )
    kwargs = dict(max_seq_len_hint=seq_len, logical_dense_kv=True)

    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_SPLIT_KV", "1")
    unsplit, _ = _paged_adapter.paged_forward(*args, **kwargs)
    monkeypatch.setenv("SGLANG_V100_PREFILL_D256_SPLIT_KV", "auto")
    split, _ = _paged_adapter.paged_forward(*args, **kwargs)

    torch.testing.assert_close(split, unsplit, rtol=1e-3, atol=5e-6)


def test_native_e5m2_cache_writer_matches_torch_conversion():
    from sglang.kernels.ops.kvcache.sm70_fp8_kv import write_fp8_e5m2_cache_sm70

    torch.manual_seed(31)
    tokens, heads, dim = 37, 1, 256
    key = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    locations = torch.randperm(128, device="cuda")[:tokens].to(torch.int64)
    key_cache = torch.zeros(128, heads, dim, device="cuda", dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)
    scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)

    if not write_fp8_e5m2_cache_sm70(
        key,
        value,
        key_cache,
        value_cache,
        locations,
        scale,
        scale,
    ):
        pytest.skip("optional SM70 TurboMind extension is not built")

    expected_key = (key / scale).to(torch.float8_e5m2).view(torch.uint8)
    expected_value = (value / scale).to(torch.float8_e5m2).view(torch.uint8)
    torch.testing.assert_close(key_cache[locations], expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache[locations], expected_value, rtol=0, atol=0)


def test_native_e5m2_paged_bridge_matches_logical_torch_gather():
    from sglang.srt.layers.attention.tilelang_fa_v100._paged_adapter import (
        gather_fp8_paged_kv,
    )

    torch.manual_seed(37)
    pages, page_size, heads, dim = 11, 16, 1, 256
    key = torch.randn(
        pages, page_size, heads, dim, device="cuda", dtype=torch.float16
    ).to(torch.float8_e5m2)
    value = torch.randn_like(key.to(torch.float16)).to(torch.float8_e5m2)
    page_table = torch.randperm(pages, device="cuda", dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([pages * page_size - 3], device="cuda", dtype=torch.int32)
    key_out = torch.empty(
        pages, page_size, heads, dim, device="cuda", dtype=torch.float16
    )
    value_out = torch.empty_like(key_out)

    gather_fp8_paged_kv(
        key,
        value,
        page_table,
        seq_lens,
        key_out,
        value_out,
    )
    active = int(seq_lens[0])
    expected_key = key.index_select(0, page_table[0].to(torch.int64)).flatten(0, 1)
    expected_value = value.index_select(0, page_table[0].to(torch.int64)).flatten(0, 1)
    torch.testing.assert_close(
        key_out.flatten(0, 1)[:active], expected_key[:active].half(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        value_out.flatten(0, 1)[:active],
        expected_value[:active].half(),
        rtol=0,
        atol=0,
    )


def test_triton_e5m2_decode_fallback_matches_dequantized_reference():
    """Keep unsupported XQA shapes correct on the software-decoded fallback."""
    torch.manual_seed(41)
    seq_len, q_heads, kv_heads, dim = 289, 5, 1, 128
    k_scale, v_scale = 0.75, 1.25
    q = torch.randn(1, q_heads, dim, device="cuda", dtype=torch.float16).mul_(0.1)
    k_source = torch.randn(
        seq_len, kv_heads, dim, device="cuda", dtype=torch.float16
    ).mul_(0.1)
    v_source = torch.randn_like(k_source).mul_(0.1)
    k_cache = (k_source / k_scale).to(torch.float8_e5m2)
    v_cache = (v_source / v_scale).to(torch.float8_e5m2)
    kv_indptr = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(seq_len, device="cuda", dtype=torch.int64)
    max_splits = 8
    num_splits = torch.tensor([4], device="cuda", dtype=torch.int32)
    logits = torch.empty(
        1, q_heads, max_splits, dim, device="cuda", dtype=torch.float32
    )
    lse = torch.empty(1, q_heads, max_splits, device="cuda", dtype=torch.float32)
    out = torch.empty_like(q)

    decode_attention_fwd(
        q,
        k_cache,
        v_cache,
        out,
        kv_indptr,
        kv_indices,
        logits,
        lse,
        num_splits,
        max_splits,
        dim**-0.5,
        k_scale,
        v_scale,
    )

    k_ref = k_cache.float().mul(k_scale).repeat_interleave(q_heads, dim=1)
    v_ref = v_cache.float().mul(v_scale).repeat_interleave(q_heads, dim=1)
    scores = torch.einsum("bhd,lhd->bhl", q.float(), k_ref) * dim**-0.5
    expected = torch.einsum("bhl,lhd->bhd", scores.softmax(-1), v_ref)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=2e-3)


def test_triton_e5m2_extend_fallback_matches_mixed_kv_reference():
    """Exercise the DSpark draft-extend path with compact E5M2 prefix KV."""
    torch.manual_seed(43)
    prefix_len, query_len, q_heads, kv_heads, dim = 257, 7, 5, 1, 128
    k_scale, v_scale = 0.75, 1.25
    q = torch.randn(query_len, q_heads, dim, device="cuda", dtype=torch.float16).mul_(
        0.1
    )
    k_prefix = torch.randn(
        prefix_len, kv_heads, dim, device="cuda", dtype=torch.float16
    ).mul_(0.1)
    v_prefix = torch.randn_like(k_prefix).mul_(0.1)
    k_cache = (k_prefix / k_scale).to(torch.float8_e5m2)
    v_cache = (v_prefix / v_scale).to(torch.float8_e5m2)
    k_extend = torch.randn(
        query_len, kv_heads, dim, device="cuda", dtype=torch.float16
    ).mul_(0.1)
    v_extend = torch.randn_like(k_extend).mul_(0.1)
    qo_indptr = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, prefix_len], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(prefix_len, device="cuda", dtype=torch.int64)
    out = torch.empty_like(q)

    extend_attention_fwd(
        q,
        k_extend,
        v_extend,
        out,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=query_len,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    k_full = torch.cat((k_cache.float() * k_scale, k_extend.float()))
    v_full = torch.cat((v_cache.float() * v_scale, v_extend.float()))
    k_full = k_full.repeat_interleave(q_heads, dim=1)
    v_full = v_full.repeat_interleave(q_heads, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k_full) * dim**-0.5
    causal = torch.arange(prefix_len + query_len, device="cuda")[None, :] <= (
        prefix_len + torch.arange(query_len, device="cuda")[:, None]
    )
    scores.masked_fill_(~causal[:, None, :], float("-inf"))
    expected = torch.einsum("qhk,khd->qhd", scores.softmax(-1), v_full)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_verify_split_granularity_rejects_invalid_override(monkeypatch, value):
    monkeypatch.setenv("SGLANG_V100_VERIFY_TOKENS_PER_SPLIT", value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        _verify_min_tokens_per_split()


def test_batched_prefix_dequant_and_scaled_extend_overlay():
    torch.manual_seed(7)
    device = "cuda:0"
    page_size = 16
    heads = 1
    dim = 32
    batch_size = 2
    max_pages = 2

    k_source = (torch.randn(6, page_size, heads, dim, device=device) * 0.5).half()
    v_source = (torch.randn_like(k_source) * 0.5).half()
    k_cache = k_source.to(torch.float8_e4m3fn)
    v_cache = v_source.to(torch.float8_e4m3fn)
    block_table = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([18, 7], dtype=torch.int32, device=device)
    k_scratch = torch.full(
        (batch_size * max_pages, page_size, heads, dim),
        -999.0,
        dtype=torch.float16,
        device=device,
    )
    v_scratch = torch.full_like(k_scratch, -999.0)

    dequantize_paged_kv_e4m3_sm70(
        k_cache,
        v_cache,
        block_table,
        prefix_lens,
        k_scratch,
        v_scratch,
        max_seq_len=18,
    )

    for batch, prefix_len in enumerate((18, 7)):
        expected_k = torch.cat(
            [k_cache[int(page)] for page in block_table[batch].cpu()]
        )[:prefix_len]
        expected_v = torch.cat(
            [v_cache[int(page)] for page in block_table[batch].cpu()]
        )[:prefix_len]
        page_slice = slice(batch * max_pages, (batch + 1) * max_pages)
        actual_k = k_scratch[page_slice].flatten(0, 1)[:prefix_len]
        actual_v = v_scratch[page_slice].flatten(0, 1)[:prefix_len]
        torch.testing.assert_close(actual_k, expected_k.half(), rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_v.half(), rtol=0, atol=0)

    qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    k_extend = torch.randn(5, heads, dim, dtype=torch.float16, device=device)
    v_extend = torch.randn_like(k_extend)
    store_paged_extend_kv_fp16_sm70(
        k_extend,
        v_extend,
        qo_indptr,
        prefix_lens,
        k_scratch,
        v_scratch,
        max_extend_len=3,
        k_scale=2.0,
        v_scale=0.5,
    )

    for batch, (start, end) in enumerate(((0, 3), (3, 5))):
        page_slice = slice(batch * max_pages, (batch + 1) * max_pages)
        token_slice = slice(
            int(prefix_lens[batch]),
            int(prefix_lens[batch]) + end - start,
        )
        actual_k = k_scratch[page_slice].flatten(0, 1)[token_slice]
        actual_v = v_scratch[page_slice].flatten(0, 1)[token_slice]
        torch.testing.assert_close(actual_k, k_extend[start:end] / 2.0, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, v_extend[start:end] / 0.5, rtol=0, atol=0)


def test_paged_adapter_applies_non_unit_kv_scales(monkeypatch):
    captured = {}

    def fake_get_paged_verify_kernels(**_kwargs):
        def partial(*args):
            captured["softmax_scale"] = args[-1]
            return torch.empty(1, device="cuda"), torch.empty(1, device="cuda")

        def combine(*_args):
            return torch.full((1, 1, 32), 4.0, dtype=torch.float16, device="cuda")

        return partial, combine, None

    monkeypatch.setattr(
        _paged_adapter,
        "get_paged_verify_kernels",
        fake_get_paged_verify_kernels,
    )
    q = torch.zeros((1, 1, 32), dtype=torch.float16, device="cuda")
    k_cache = torch.zeros((1, 16, 1, 32), dtype=torch.float16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    out, _ = _paged_adapter.paged_forward(
        q,
        k_cache,
        v_cache,
        torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        softmax_scale=0.5,
        linear_verify=True,
        k_scale=2.0,
        v_scale=0.25,
    )

    assert captured["softmax_scale"] == 1.0
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
