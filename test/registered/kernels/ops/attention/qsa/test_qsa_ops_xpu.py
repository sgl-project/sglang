import sys

import pytest
import torch

from sglang.srt.layers.attention.qsa.kernel import (
    expand_qsa_block_indices,
    qsa_fast_topk,
    qsa_sparse_attention,
    qsa_sparse_attention_reference,
    torch_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qsa.mqa import (
    qsa_mqa_decode,
    qsa_mqa_prefill,
    torch_qsa_mqa_decode,
    torch_qsa_mqa_prefill,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU required"
)

COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
FINAL_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_xpu_mqa_prefill_dispatch_and_reference(monkeypatch, dtype):
    import sgl_kernel  # noqa: F401

    native = torch.ops.sgl_kernel.qsa_mqa_prefill
    calls = []

    def traced(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(torch.ops.sgl_kernel, "qsa_mqa_prefill", traced)
    torch.manual_seed(62)
    q = torch.randn(5, 4, 128, device="xpu", dtype=torch.bfloat16).to(dtype)
    k = torch.randn(137, 1, 128, device="xpu", dtype=torch.bfloat16).to(dtype)
    starts = torch.tensor([0, 0, 5, 5, 27], device="xpu", dtype=torch.int64)
    ends = torch.tensor([0, 9, 97, 137, 104], device="xpu", dtype=torch.int64)
    expected = torch_qsa_mqa_prefill(q, k, starts, ends)
    actual = qsa_mqa_prefill(q, k, starts, ends)
    assert calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_xpu_mqa_decode_dispatch_and_reference(monkeypatch, dtype):
    import sgl_kernel  # noqa: F401

    native = torch.ops.sgl_kernel.qsa_mqa_decode
    calls = []

    def traced(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(torch.ops.sgl_kernel, "qsa_mqa_decode", traced)
    torch.manual_seed(63)
    q = torch.randn(3, 4, 128, device="xpu", dtype=torch.bfloat16).to(dtype)
    cache = torch.randn(5, 16, 1, 128, device="xpu", dtype=torch.bfloat16).to(dtype)
    pages = torch.randint(0, 5, (3, 2), device="xpu", dtype=torch.int64)
    lengths = torch.tensor([0, 17, 29], device="xpu", dtype=torch.int64)
    expected = torch_qsa_mqa_decode(q, cache, pages, lengths, 49)
    actual = qsa_mqa_decode(q, cache, pages, lengths, 49)
    assert calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0.004, atol=0.004)


def test_qsa_xpu_mqa_rejects_mixed_fp8_and_bf16():
    q = torch.ones(1, 4, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k = torch.ones(16, 1, 128, device="xpu", dtype=torch.bfloat16)
    starts = torch.zeros(1, device="xpu", dtype=torch.int32)
    ends = torch.ones_like(starts)
    with pytest.raises(ValueError, match="same dtype"):
        qsa_mqa_prefill(q, k, starts, ends)
    with pytest.raises(ValueError, match="same dtype"):
        qsa_mqa_decode(
            q,
            k.reshape(1, 16, 1, 128),
            starts.reshape(1, 1),
            ends,
            16,
        )


def test_qsa_xpu_fp8_mqa_topk_pipeline():
    torch.manual_seed(64)
    q = torch.randn(2, 4, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(769, 1, 128, device="xpu", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    starts = torch.tensor([0, 19], device="xpu", dtype=torch.int32)
    ends = torch.tensor([513, 769], device="xpu", dtype=torch.int32)
    scores = qsa_mqa_prefill(q, k, starts, ends)
    reference = torch_qsa_mqa_prefill(q, k, starts, ends)
    torch.testing.assert_close(scores, reference, rtol=0.004, atol=0.004)
    actual = qsa_fast_topk(scores, starts, ends, topk=512)
    expected = qsa_fast_topk(reference, starts, ends, topk=512)
    torch.testing.assert_close(actual.sort(dim=-1).values, expected.sort(dim=-1).values)


def test_qsa_xpu_mqa_unsupported_shape_keeps_torch_fallback():
    q = torch.randn(1, 2, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(9, 1, 128, device="xpu", dtype=torch.bfloat16)
    starts = torch.zeros(1, device="xpu", dtype=torch.int32)
    ends = torch.full_like(starts, 9)
    torch.testing.assert_close(
        qsa_mqa_prefill(q, k, starts, ends),
        torch_qsa_mqa_prefill(q, k, starts, ends),
    )
    cache = k[:8].reshape(1, 8, 1, 128)
    pages = torch.zeros(1, 1, device="xpu", dtype=torch.int32)
    torch.testing.assert_close(
        qsa_mqa_decode(q, cache, pages, ends, 8),
        torch_qsa_mqa_decode(q, cache, pages, ends, 8),
    )


@pytest.mark.parametrize("topk", [17, 65, 512, FINAL_TOPK])
def test_qsa_xpu_sparse_attention_matches_reference(monkeypatch, topk):
    import sgl_kernel

    native_op = torch.ops.sgl_kernel.qsa_sparse_attention
    native_wrapper = sgl_kernel.qsa_sparse_attention
    calls = []
    wrapper_calls = []

    def traced(*args):
        calls.append(True)
        return native_op(*args)

    def traced_wrapper(*args):
        wrapper_calls.append(True)
        return native_wrapper(*args)

    monkeypatch.setattr(torch.ops.sgl_kernel, "qsa_sparse_attention", traced)
    monkeypatch.setattr(sgl_kernel, "qsa_sparse_attention", traced_wrapper)
    torch.manual_seed(41)
    rows, heads, kv_heads, dim = 2, 24, 2, 256
    q = torch.randn(rows, heads, dim, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1024, kv_heads, dim, device="xpu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.randint(0, k.shape[0], (rows, topk), device="xpu", dtype=torch.int32)
    slots[:, 3::7] = -1
    expected = qsa_sparse_attention_reference(q, k, v, slots, 0.0625)
    actual = qsa_sparse_attention(q, k, v, slots, 0.0625)
    assert wrapper_calls == calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [0, 1, 511, 512, 513, 1024, 4096])
@pytest.mark.parametrize("start", [0, 7])
def test_qsa_xpu_fast_topk_matches_reference(batch, length, start):
    torch.manual_seed(42)
    logits = torch.randn(batch, start + length, device="xpu", dtype=torch.float32)
    starts = torch.full((batch,), start, device="xpu", dtype=torch.int32)
    ends = starts + length
    actual = qsa_fast_topk(logits, starts, ends, topk=512)
    assert actual.shape == (batch, 512)
    assert actual.dtype == torch.int32

    expected = torch.full((batch, 512), -1, device="xpu", dtype=torch.int32)
    if length:
        width = min(length, 512)
        expected[:, :width] = torch.topk(
            logits[:, start : start + length], width, dim=-1
        ).indices.to(torch.int32)
    torch.testing.assert_close(actual.sort(dim=-1).values, expected.sort(dim=-1).values)


def test_qsa_xpu_fast_topk_2048():
    topk = 2048
    logits = torch.randn(2, 4103, device="xpu")
    starts = torch.full((2,), 7, device="xpu", dtype=torch.int32)
    ends = starts + 4096
    actual = qsa_fast_topk(logits, starts, ends, topk=topk)
    expected = torch.topk(logits[:, 7:], topk, dim=-1).indices.to(torch.int32)
    torch.testing.assert_close(actual.sort(dim=-1).values, expected.sort(dim=-1).values)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_qsa_xpu_expansion_matches_reference_and_dispatches_triton(
    monkeypatch, position_dtype
):
    from sglang.srt.layers.attention.qsa import kernel as kernel_module

    blocks = torch.full((4, BLOCK_TOPK), -1, device="xpu", dtype=torch.int32)
    blocks[0, 0] = 0
    blocks[1, :2] = torch.tensor([1, 0], device="xpu")
    blocks[2:, :] = torch.arange(BLOCK_TOPK, device="xpu")
    positions = torch.tensor([5, 10, 2047, 3000], device="xpu", dtype=position_dtype)
    lengths = positions + 1
    expected = torch_expand_qsa_block_indices(
        blocks, positions, lengths, COMPRESS_RATIO, TOKEN_TOPK
    )

    original = kernel_module.triton_expand_qsa_block_indices
    calls = []

    def traced(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(kernel_module, "triton_expand_qsa_block_indices", traced)
    actual = expand_qsa_block_indices(
        blocks, positions, lengths, COMPRESS_RATIO, TOKEN_TOPK
    )
    assert calls == [True]
    torch.testing.assert_close(actual, expected)


def test_qsa_xpu_topk_expansion_pipeline_mixed_lengths():
    logits = torch.randn(2, 1031, device="xpu", dtype=torch.float32)
    starts = torch.tensor([0, 7], device="xpu", dtype=torch.int32)
    ends = starts + torch.tensor([128, 1024], device="xpu", dtype=torch.int32)
    positions = torch.tensor([511, 4095], device="xpu", dtype=torch.int64)
    selected = qsa_fast_topk(logits, starts, ends, topk=BLOCK_TOPK)

    actual = expand_qsa_block_indices(
        selected, positions, positions + 1, COMPRESS_RATIO, TOKEN_TOPK
    )
    expected = torch_expand_qsa_block_indices(
        selected, positions, positions + 1, COMPRESS_RATIO, TOKEN_TOPK
    )
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
