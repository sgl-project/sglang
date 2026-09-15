"""NPU sparse attention comparisons against the existing Torch reference."""

import pytest
import torch

from sglang.srt.hardware_backend.npu.kernels.qwen3_8_flash_next.sparse_attention import (
    can_run_sparse_attention,
    sparse_attention,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")

pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


def make_inputs(rows, heads, kv_heads, dim, width, dtype, strided=False):
    torch.manual_seed(42)
    q = torch.randn(rows, heads, dim, device="npu", dtype=dtype)
    k = torch.randn(3072, kv_heads, dim, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    slots = torch.randint(1, 3072, (rows, width), device="npu", dtype=torch.int32)
    if rows and width:
        slots[0] = -1
        slots[:, ::7] = -1
        if rows > 1:
            slots[1, ::5] = 1  # Repeated valid slots must retain their multiplicity.
    k[0] = float("nan")
    v[0] = float("inf")
    if strided:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        k = k.transpose(0, 1).contiguous().transpose(0, 1)
        v = v.transpose(0, 1).contiguous().transpose(0, 1)
        slots = slots.t().contiguous().t()
    return q, k, v, slots


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,heads,kv_heads,dim,width",
    [
        (0, 3, 1, 256, 2051),
        (2, 3, 1, 256, 0),
        (2, 3, 1, 256, 1),
        (3, 4, 2, 64, 33),
        (8, 3, 1, 256, 2051),
        (32, 3, 1, 256, 2051),
        (3, 4, 2, 128, 257),
    ],
)
def test_sparse_attention_reference(rows, heads, kv_heads, dim, width, dtype):
    args = make_inputs(rows, heads, kv_heads, dim, width, dtype)
    before = [x.clone() for x in args]
    expected = qsa_sparse_attention_reference(*args)
    actual = sparse_attention(*args)
    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-3}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.isfinite(actual).all()
    for original, saved in zip(args, before):
        torch.testing.assert_close(original, saved, equal_nan=True)


def test_sparse_attention_strides_and_scale():
    args = make_inputs(3, 4, 2, 128, 257, torch.float32, strided=True)
    assert can_run_sparse_attention(*args)
    expected = qsa_sparse_attention_reference(*args, 0.3)
    actual = sparse_attention(*args, 0.3)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_sparse_attention_graph_replay():
    args = make_inputs(8, 3, 1, 256, 2051, torch.bfloat16)
    for _ in range(2):
        sparse_attention(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = sparse_attention(*args)
    for length in (0, 33, 2051):
        args[0].normal_()
        args[3].fill_(-1)
        args[3][:, :length] = 1
        graph.replay()
        torch.npu.synchronize()
        expected = qsa_sparse_attention_reference(*args)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_sparse_attention_unsupported_layout():
    args = make_inputs(2, 3, 1, 256, 33, torch.bfloat16)
    q, k, v, slots = args
    unsupported = (q[..., ::2], k[..., ::2], v[..., ::2], slots)
    assert not can_run_sparse_attention(*unsupported)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        sparse_attention(*unsupported)


def test_sparse_attention_dispatch(monkeypatch):
    from sglang.srt.layers.attention.qsa import kernel

    args = make_inputs(8, 3, 1, 256, 2051, torch.bfloat16)
    expected = qsa_sparse_attention_reference(*args)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Supported inputs must execute the Triton kernel")

    monkeypatch.setattr(kernel, "qsa_sparse_attention_reference", unexpected_fallback)
    # Exercise NPU rank-4 cache adaptation through the public entry point.
    q, k, v, slots = args
    actual = kernel.qsa_sparse_attention(q, k.unsqueeze(1), v.unsqueeze(1), slots)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_sparse_attention_large_batch():
    args = list(make_inputs(128, 3, 1, 256, 2051, torch.bfloat16))
    args[-1] = args[-1].to(torch.int64)
    expected = qsa_sparse_attention_reference(*args)
    actual = sparse_attention(*args)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
