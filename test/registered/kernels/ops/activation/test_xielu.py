import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.activation.xielu import xielu, xielu_coeffs, xielu_reference
from sglang.srt.layers.activation import XIELU
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=15, stage="nightly", runner_config="1-gpu-large")

# Post-softplus coefficients of layer 0 of swiss-ai/Apertus-8B-Instruct-2509. Its large
# alpha_n makes the negative branch cancel near x = -2 * beta / alpha_n, which is where
# lower-precision evaluations go wrong.
COEFFS = torch.tensor([166.0, 41.25, 0.5, -9.98377799987793e-07], device="cuda")
DTYPES = [torch.bfloat16, torch.float16]
# Apertus's MLP width is 21504; the row counts cover every vector width the launcher
# picks, from decode batches to prefill chunks.
SHAPES = get_ci_test_range(
    full_range=[(m, 21504) for m in (1, 2, 3, 16, 31, 64, 128, 1024, 4096)]
    + [(1,), (7,), (17,), (1000003,), (2, 5, 1000)],
    ci_range=[(1, 21504), (16, 21504), (128, 21504), (1024, 21504), (17,)],
)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16).int()


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> None:
    """bf16 must be bit-identical. In fp16 the expm1f of the CUDA toolkit can differ by
    one ulp from torch's, which shows up after the cancellation, so allow one ulp."""
    assert out.shape == ref.shape and out.dtype == ref.dtype
    both_nan = out.isnan() & ref.isnan()
    assert torch.equal(out.isnan(), ref.isnan())
    diff = torch.where(both_nan, 0, (_bits(out) - _bits(ref)).abs())
    limit = 0 if out.dtype == torch.bfloat16 else 1
    assert diff.numel() == 0 or diff.max().item() <= limit, diff.max().item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_xielu_matches_fp32_reference(dtype: torch.dtype, shape: tuple) -> None:
    x = (torch.randn(shape, device="cuda") * 3).to(dtype)
    _assert_matches(xielu(x, COEFFS), xielu_reference(x, COEFFS))


@pytest.mark.parametrize("dtype", DTYPES)
def test_xielu_every_bit_pattern(dtype: torch.dtype) -> None:
    x = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
    x = x.to(torch.int16).view(dtype)
    _assert_matches(xielu(x, COEFFS), xielu_reference(x, COEFFS))


@pytest.mark.parametrize("offset", [1, 2, 3, 4, 8])
def test_xielu_misaligned_views(offset: int) -> None:
    base = (torch.randn(4 * 21504 + 16, device="cuda") * 3).to(torch.bfloat16)
    x = base[offset : offset + 2 * 21504]
    _assert_matches(xielu(x, COEFFS), xielu_reference(x, COEFFS))


def test_xielu_special_values() -> None:
    eps = COEFFS[3].item()
    vals = [0.0, -0.0, eps, 2 * eps, 1e-38, -1e-38, 3e38, -3e38]
    vals += [88.0, -88.0, float("inf"), float("-inf"), float("nan")]
    x = torch.tensor(vals * 3, device="cuda").to(torch.bfloat16)
    _assert_matches(xielu(x, COEFFS), xielu_reference(x, COEFFS))


def test_xielu_out_param_and_empty() -> None:
    x = torch.randn(3, 21504, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    assert xielu(x, COEFFS, out=out).data_ptr() == out.data_ptr()
    _assert_matches(out, xielu_reference(x, COEFFS))
    empty = torch.empty(0, 21504, device="cuda", dtype=torch.bfloat16)
    assert xielu(empty, COEFFS).shape == empty.shape


def test_xielu_cuda_graph_follows_coefficient_updates() -> None:
    """The kernel reads the coefficients on device, so a captured graph sees an
    in-place update (as after a weight update) without recapture."""
    coeffs = COEFFS.clone()
    x = torch.randn(16, 21504, device="cuda", dtype=torch.bfloat16)
    xielu(x, coeffs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = xielu(x, coeffs)
    x.copy_(torch.randn_like(x) * 3)
    graph.replay()
    _assert_matches(out, xielu_reference(x, coeffs))
    coeffs.copy_(torch.tensor([0.32, 0.51, 0.5, -1e-6], device="cuda"))
    graph.replay()
    _assert_matches(out, xielu_reference(x, coeffs))


def _layer(alpha_p: float, alpha_n: float) -> XIELU:
    layer = XIELU().cuda()
    layer.alpha_p.data.fill_(alpha_p)
    layer.alpha_n.data.fill_(alpha_n)
    layer.update_coefficients()
    return layer


def test_xielu_layer_uses_loaded_coefficients() -> None:
    """The layer derives the coefficients from the loaded weights, and its CUDA and
    native paths agree bit for bit."""
    layer = _layer(166.0, 40.75)
    expected = xielu_coeffs(layer.alpha_p, layer.alpha_n, layer.beta, layer.eps)
    assert torch.equal(layer._coeffs, expected)
    x = (torch.randn(64, 21504, device="cuda") * 3).to(torch.bfloat16)
    ref = xielu_reference(x, expected)
    _assert_matches(layer.forward_cuda(x), ref)
    _assert_matches(layer.forward_native(x), ref)
    # A non-contiguous input takes the native path and gives the same result.
    xt = x.t()
    _assert_matches(layer.forward_cuda(xt), layer.forward_native(xt))


def test_xielu_layer_float32_uses_native_path() -> None:
    layer = _layer(5.5625, 2.359375)
    x = torch.randn(8, 21504, device="cuda")
    torch.testing.assert_close(
        layer.forward_cuda(x), layer.forward_native(x), rtol=0, atol=0
    )


def test_xielu_layer_builds_on_meta_device() -> None:
    """Zero-copy IPC loading builds the model on the meta device."""
    with torch.device("meta"):
        layer = XIELU()
    assert layer._coeffs.is_meta
    assert "_coeffs" not in layer.state_dict()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
