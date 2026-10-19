"""WO-A on an exact E4M3 copy of the BF16 weight: same products as the BF16-weight
kernel, torch.bmm's accuracy, batch invariance, FlashInfer MXFP8 output."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.gemm.dsv4_wo_a import (
    WO_A_FP8_MAX_ROWS,
    quantize_wo_a_fp8,
    wo_a_bf16_small_batch,
    wo_a_fp8_small_batch,
    wo_a_fp8_small_batch_mxfp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA Triton kernels",
)

DEVICE = "cuda"


def _fp8_weight(seed=0):
    """A BF16 weight dequantized from E4M3 with power-of-two 32x32 block scales."""
    gen = torch.Generator(DEVICE).manual_seed(seed)
    # Every finite E4M3 value, subnormals and +-448 included (0x7f/0xff are NaN).
    bits = torch.randint(
        0, 256, (2048, 4096), device=DEVICE, dtype=torch.uint8, generator=gen
    )
    q = bits.masked_fill_(bits % 128 == 127, 0).view(torch.float8_e4m3fn)
    exponent = torch.randint(-14, -6, (64, 1, 128, 1), device=DEVICE, generator=gen)
    scale = torch.exp2(exponent.float()).bfloat16()
    w = q.view(64, 32, 128, 32).bfloat16() * scale
    return w.view(2, 1024, 4096)


def _activations(rows, seed=1):
    # The model passes a strided [rows, 2, 4096] view (stride(0) > 8192).
    gen = torch.Generator(DEVICE).manual_seed(seed)
    base = torch.randn(rows, 3, 4096, device=DEVICE, generator=gen)
    return base.to(torch.bfloat16)[:, :2]


def _fp64_reference(x, w):
    out = torch.empty(x.shape[0], 2, 1024, dtype=torch.float64, device=DEVICE)
    for g in range(2):
        for r in range(0, 1024, 256):
            out[:, g, r : r + 256] = x[:, g].double() @ w[g, r : r + 256].double().T
    return out


def _args(w, enabled=True):
    """wo_a_fp8_small_batch's weight arguments for w: BF16 weight, copy, flag."""
    q, s = quantize_wo_a_fp8(w)
    return w, q, s, torch.full((1,), int(enabled), dtype=torch.int32, device=DEVICE)


def test_quantize_round_trip():
    w = _fp8_weight()
    w[1, 32:64, 4064:] = 0  # an all-zero block
    q, s = quantize_wo_a_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    back = q.view(64, 32, 128, 32).bfloat16() * s.view(64, 1, 128, 1).bfloat16()
    assert torch.equal(back.view(2, 1024, 4096), w)
    # BF16 values that are not an FP8 dequantization have no exact copy.
    assert quantize_wo_a_fp8(w + 1e-3 * torch.randn_like(w)) is None
    assert quantize_wo_a_fp8(w.half()) is None


@pytest.mark.parametrize("rows", [2, 5, 8])
def test_matches_bf16_weight_kernel(rows):
    w = _fp8_weight()
    x = _activations(rows)
    actual = wo_a_fp8_small_batch(x, *_args(w)).float()
    expected = wo_a_bf16_small_batch(x, w).float()
    # Same operands and K order; the MMA staging may round a few outputs differently.
    ulp = torch.finfo(torch.bfloat16).eps * expected.abs()
    assert ((actual - expected).abs() <= ulp).all()
    assert (actual != expected).float().mean() < 0.01


@pytest.mark.parametrize("rows", [1, 9, 16, 17, 33, WO_A_FP8_MAX_ROWS])
def test_as_accurate_as_bmm(rows):
    # torch.bmm on the BF16 weight is what these rows use without the copy.
    w = _fp8_weight()
    x = _activations(rows)
    actual = wo_a_fp8_small_batch(x, *_args(w)).double()
    bmm = torch.empty((rows, 2, 1024), dtype=torch.bfloat16, device=DEVICE)
    torch.bmm(x.transpose(0, 1), w.transpose(1, 2), out=bmm.transpose(0, 1))
    reference = _fp64_reference(x, w)
    error = (actual - reference).abs()
    bmm_error = (bmm.double() - reference).abs()
    # Both round FP32 sums to BF16 once; the sum order only moves the error slightly.
    assert error.max() <= 1.01 * bmm_error.max()
    assert error.mean() <= 1.01 * bmm_error.mean()


def test_batch_invariant():
    w = _fp8_weight()
    copy = _args(w)
    x = _activations(WO_A_FP8_MAX_ROWS)
    full = wo_a_fp8_small_batch(x, *copy)
    for rows in (1, 2, 8, 9, 16, 17, 32, 33):
        assert torch.equal(wo_a_fp8_small_batch(x[:rows], *copy), full[:rows])
    perm = torch.randperm(WO_A_FP8_MAX_ROWS, device=DEVICE)
    assert torch.equal(wo_a_fp8_small_batch(x[perm], *copy), full[perm])


@pytest.mark.parametrize("rows", [1, 8, 33, WO_A_FP8_MAX_ROWS])
def test_mxfp8_output_matches_flashinfer_under_graph_replay(rows):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the MXFP8 reference quantizer is Blackwell-only")
    from flashinfer import mxfp8_quantize

    w = _fp8_weight()
    copy = _args(w)
    x = _activations(rows)
    wo_a_fp8_small_batch_mxfp8(x, *copy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = wo_a_fp8_small_batch_mxfp8(x, *copy)
    for seed in range(2):
        x.copy_(_activations(rows, seed=seed + 2))
        s.fill_(255)  # padding rows of the 128-row scale tile must be rewritten
        graph.replay()
        expected_q, expected_s = mxfp8_quantize(
            wo_a_fp8_small_batch(x, *copy).flatten(1), True, alignment=32
        )
        assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(s, expected_s)


@pytest.mark.parametrize("rows", [2, 8, 33])
def test_disabled_flag_reads_bf16_weight_under_graph_replay(rows):
    weight, q, s, enabled = _args(_fp8_weight())
    x = _activations(rows)
    wo_a_fp8_small_batch(x, weight, q, s, enabled)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = wo_a_fp8_small_batch(x, weight, q, s, enabled)
    graph.replay()
    from_copy = y.clone()
    # The BF16 weight changes to values the stale copy does not hold.
    weight.mul_(1.01)
    enabled.zero_()
    graph.replay()
    if rows <= 8:
        assert torch.equal(y, wo_a_bf16_small_batch(x, weight))
    else:
        error = (y.double() - _fp64_reference(x, weight)).abs()
        assert error.max() <= torch.finfo(torch.bfloat16).eps * y.abs().max()
    enabled.fill_(1)
    graph.replay()
    assert torch.equal(y, from_copy)


def _layer(weight):
    from sglang.srt.models.deepseek_v4 import MQALayer

    class Layer(torch.nn.Module):
        refresh_derived_weights = MQALayer.refresh_derived_weights
        invalidate_derived_weights = MQALayer.invalidate_derived_weights

    layer = Layer()
    for name in ("wo_a_fp8_weight", "wo_a_fp8_scale", "wo_a_fp8_enabled"):
        layer.register_buffer(name, None, persistent=False)
    layer.is_dsv41, layer.n_local_groups, layer.o_lora_rank = True, 2, 1024
    layer.wo_a = SimpleNamespace(weight=weight.view(2048, 4096))
    return layer


def test_layer_copy_follows_online_updates():
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _invalidate_derived_weights,
    )
    from sglang.srt.models import deepseek_v4

    platform = SimpleNamespace(device_sm=120)
    with (
        mock.patch.object(deepseek_v4, "get_platform", return_value=platform),
        envs.SGLANG_DSV41_WO_A_FP8_COPY.override(True),
    ):
        layer = _layer(_fp8_weight(0))
        layer.refresh_derived_weights()
        weight = layer.wo_a.weight.view(2, 1024, 4096)
        buffers = (layer.wo_a_fp8_weight, layer.wo_a_fp8_scale, layer.wo_a_fp8_enabled)
        assert torch.equal(
            buffers[0].view(torch.uint8), quantize_wo_a_fp8(weight)[0].view(torch.uint8)
        )
        x = _activations(4)
        wo_a_fp8_small_batch(x, weight, *buffers)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = wo_a_fp8_small_batch(x, weight, *buffers)

        # An update that skips post_load_weights (load_format="direct"): the
        # updater's invalidation makes the captured graph read the BF16 weight.
        _invalidate_derived_weights(layer)
        weight.mul_(1.01)
        graph.replay()
        assert torch.equal(y, wo_a_bf16_small_batch(x, weight))
        # post_load_weights with no exact copy keeps the BF16 fallback.
        layer.refresh_derived_weights()
        graph.replay()
        assert torch.equal(y, wo_a_bf16_small_batch(x, weight))

        # An exact update is rewritten into the same buffers and used again.
        weight.copy_(_fp8_weight(1))
        layer.refresh_derived_weights()
        current = (layer.wo_a_fp8_weight, layer.wo_a_fp8_scale, layer.wo_a_fp8_enabled)
        assert all(a is b for a, b in zip(buffers, current))
        assert buffers[2].item() == 1
        assert torch.equal(
            buffers[0].view(torch.uint8), quantize_wo_a_fp8(weight)[0].view(torch.uint8)
        )
        graph.replay()
        assert torch.equal(y, wo_a_fp8_small_batch(x, weight, *buffers))

        # A refresh that cannot allocate keeps serving the BF16 weight.
        with mock.patch.object(
            deepseek_v4, "quantize_wo_a_fp8", side_effect=torch.OutOfMemoryError
        ):
            layer.refresh_derived_weights()
        assert buffers[2].item() == 0

        # Without an exact copy at load, no copy is built (BF16 kernels).
        del layer, weight, buffers, current, graph
        layer = _layer(_fp8_weight(0).mul_(1.01))
        layer.refresh_derived_weights()
        assert layer.wo_a_fp8_weight is None and layer.wo_a_fp8_enabled is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
