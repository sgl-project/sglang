"""Paged NPU MQA comparisons with the unchanged Torch scoring reference."""

import pytest
import torch

from sglang.srt.hardware_backend.npu.kernels.qwen3_8_flash_next.mqa import (
    can_run_mqa_decode,
    mqa_decode,
)
from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")

pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


def make_inputs(rows, heads, dim, page_size, pages, dtype, strided=False):
    torch.manual_seed(73)
    q = torch.randn(rows, heads, dim, device="npu", dtype=dtype)
    cache = torch.randn(23, page_size, 1, dim, device="npu", dtype=dtype)
    table = torch.randint(0, 23, (rows, pages), device="npu", dtype=torch.int64)
    lengths = torch.arange(rows, device="npu", dtype=torch.int32)
    lengths = lengths * (pages * page_size) // max(rows - 1, 1)
    if rows and pages:
        table[-1, 0] = -1  # The Torch reference clamps negative page ids to zero.
    if strided:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        cache = cache.transpose(0, 1).contiguous().transpose(0, 1)
        table = table.t().contiguous().t()
        lengths = torch.stack((lengths, lengths), dim=1)[:, 0]
    return q, cache, table, lengths


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,heads,dim,page_size,pages,width",
    [
        (0, 4, 128, 16, 3, 48),
        (3, 4, 128, 16, 3, 0),
        (3, 4, 128, 16, 0, 19),
        (3, 4, 128, 16, 3, 61),
        (8, 4, 128, 64, 7, 257),
        (4, 3, 64, 3, 7, 21),
        (4, 1, 256, 1, 17, 129),
        (32, 4, 128, 16, 129, 2065),
        (128, 8, 128, 16, 9, 65536),
    ],
)
def test_mqa_reference(rows, heads, dim, page_size, pages, width, dtype):
    args = make_inputs(rows, heads, dim, page_size, pages, dtype)
    before = [x.clone() for x in args]
    expected = mqa.torch_qsa_mqa_decode(*args, width)
    actual = mqa_decode(*args, width)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for original, saved in zip(args, before):
        torch.testing.assert_close(original, saved)


def test_mqa_strides_scale_and_dispatch(monkeypatch):
    args = make_inputs(4, 4, 128, 16, 19, torch.bfloat16, strided=True)
    expected = mqa.torch_qsa_mqa_decode(*args, 333, 3.0)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Supported NPU inputs must execute Triton")

    monkeypatch.setattr(mqa, "torch_qsa_mqa_decode", unexpected_fallback)
    actual = mqa.qsa_mqa_decode(*args, 333, 3.0)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_mqa_padding_and_graph_replay():
    args = make_inputs(4, 4, 128, 16, 9, torch.bfloat16)
    q, cache, table, lengths = args
    width = 161
    for _ in range(2):
        mqa_decode(*args, width)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = mqa_decode(*args, width)
    for length in (0, 1, 129, 144):
        q.normal_()
        lengths.fill_(length)
        table.fill_(1)
        cache[1].normal_()
        cache[0] = float("nan")
        table[:, (length + 15) // 16 :] = 0
        graph.replay()
        torch.npu.synchronize()
        expected = mqa.torch_qsa_mqa_decode(*args, width)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_mqa_unsupported_layout_keeps_reference():
    args = make_inputs(3, 4, 128, 16, 3, torch.bfloat16)
    q, cache, table, lengths = args
    args = q[..., ::2], cache[..., ::2], table, lengths
    assert not can_run_mqa_decode(*args, 48)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        mqa_decode(*args, 48)
    expected = mqa.torch_qsa_mqa_decode(*args, 48)
    torch.testing.assert_close(mqa.qsa_mqa_decode(*args, 48), expected)
