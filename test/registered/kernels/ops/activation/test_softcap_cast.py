from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits,
    softcap_to_float32_logits,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_softcap_cast_strides_and_replay(dtype):
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
        .view(32, 2048)
    )
    source = torch.empty((32, 4096), device="cuda", dtype=dtype)[:, :2048]
    source.copy_(values)
    backing = torch.full((32, 4096), 123.0, device="cuda")
    out = backing[:, :2048]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        softcap_to_float32_logits(source, 30.0, out)
    source.copy_(source.flip(0))
    graph.replay()
    expected = softcap_inplace_logits(source.float(), 30.0)
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        out.view(torch.int32)[finite], expected.view(torch.int32)[finite]
    )
    assert torch.all(backing[:, 2048:] == 123.0)


@pytest.mark.parametrize("buffer_rows", [1, 7])
@torch.inference_mode()
def test_softcap_padded_vocab_buffer(buffer_rows):
    logits = torch.randn(7, 160, device="cuda", dtype=torch.bfloat16)
    buffer = torch.empty(buffer_rows, 131, device="cuda")
    actual = LogitsProcessor._copy_logits_to_buffer(
        SimpleNamespace(vocab_size=131),
        logits,
        SimpleNamespace(next_token_logits_buffer=buffer),
        use_buffer=True,
        softcap=30.0,
    )
    torch.testing.assert_close(
        actual, softcap_inplace_logits(logits[:, :131].float(), 30.0), atol=0, rtol=0
    )
    assert (actual.data_ptr() == buffer.data_ptr()) == (buffer_rows == 7)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
