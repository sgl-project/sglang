"""The --lm-head-dtype mxfp8 copy on a Blackwell GPU: weight bytes, logits against
a dequantized reference, and CUDA graph replay before and after an in-place
rebuild of the copy."""

import sys

import pytest
import torch

from sglang.srt.layers.quantization.mxfp8_lm_head import MIN_ROWS, Mxfp8LmHead
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.mxfp8 import assert_matches_flashinfer_mxfp8

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 scaled_mm needs a Blackwell GPU",
)

VOCAB, HIDDEN = 3008, 1024  # 3008 rows: the last 128-row scale tile is padded


def _weight() -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((VOCAB, HIDDEN), device="cuda", generator=g) * 0.02
    return w.to(torch.bfloat16)


def _dequantized(x: torch.Tensor) -> torch.Tensor:
    import flashinfer

    q, scale = flashinfer.mxfp8_quantize(x.contiguous(), False)
    rows, k = x.shape
    scale = scale.view(torch.uint8).reshape(-1)[: rows * k // 32].view(rows, k // 32)
    blocks = q.float().view(rows, k // 32, 32)
    return (blocks * torch.exp2(scale.float() - 127)[..., None]).view(rows, k)


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())


@pytest.mark.parametrize("rows", [1, 7, 240, 1000])
def test_logits_match_the_dequantized_reference(rows: int) -> None:
    weight = _weight()
    head = Mxfp8LmHead(weight)
    assert_matches_flashinfer_mxfp8((head.qweight, head.weight_scale), weight)

    x = torch.randn((rows, HIDDEN), device="cuda", dtype=torch.bfloat16)
    logits = head(x)
    assert logits.dtype == torch.bfloat16
    assert logits.shape == (rows, VOCAB)
    # Same products, FP32 accumulation; only the BF16 output rounding remains.
    reference = _dequantized(x) @ _dequantized(weight).t()
    assert _rel_rms(logits, reference) < 1e-2


def test_applies_to_its_own_weight_from_min_rows() -> None:
    weight = _weight()
    head = Mxfp8LmHead(weight)

    def rows(n):
        return torch.zeros((n, HIDDEN), device="cuda", dtype=torch.bfloat16)

    assert head.applies(rows(MIN_ROWS), weight)
    assert not head.applies(rows(MIN_ROWS - 1), weight)
    assert not head.applies(rows(MIN_ROWS), weight.clone())
    assert not head.applies(rows(MIN_ROWS).half(), weight)


def test_graph_replay_reads_the_rebuilt_copy() -> None:
    weight = _weight()
    head = Mxfp8LmHead(weight)
    x = torch.randn((240, HIDDEN), device="cuda", dtype=torch.bfloat16)
    eager = head(x)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        head(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = head(x)
    graph.replay()
    assert torch.equal(captured, eager)

    pointers = (head.qweight.data_ptr(), head.weight_scale.data_ptr())
    weight.mul_(-0.37)
    head.rebuild()
    graph.replay()
    assert (head.qweight.data_ptr(), head.weight_scale.data_ptr()) == pointers
    assert torch.equal(captured, Mxfp8LmHead(weight.clone())(x))
    assert not torch.equal(captured, eager)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
