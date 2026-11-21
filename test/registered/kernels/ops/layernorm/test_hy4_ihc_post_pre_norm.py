from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.kernels.ops.layernorm.hy4_ihc import (
    _IHC_DOT_MIN_T,
    _IHC_TOKEN_BLOCKED_MIN_T,
    fused_hy4_ihc_post,
    hy4_ihc_post_pre_norm,
    ihc_fuses_out_norm,
)
from sglang.srt.models import hunyuan_v4, hunyuan_v4_flash
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference_post_pre_norm(
    output,
    residual,
    post,
    hc_fn,
    hc_scale,
    hc_base,
    magnitude,
    norm_eps,
    hc_eps,
    rms_weight,
    rms_eps,
    out_norm_weight=None,
    out_norm_eps=0.0,
):
    """The unfused math: iHC post, then iHC pre, then the block-input RMSNorm."""
    _, hc_mult, hidden_size = residual.shape
    if output is not None:
        out = output.float()
        if out_norm_weight is not None:
            # The unfused path writes the normalized output as bf16 before the post.
            inv = torch.rsqrt(out.square().mean(-1, keepdim=True) + out_norm_eps)
            out = (out * inv * out_norm_weight.float()).to(residual.dtype).float()
        new_residual = (
            post.float().unsqueeze(-1) * out.unsqueeze(1) + residual.float()
        ).to(residual.dtype)
    else:
        new_residual = residual

    flat = new_residual.float().flatten(1)
    inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    gates = flat @ hc_fn.float().T
    pre = torch.sigmoid(gates[:, :hc_mult] * inv * hc_scale[0] + hc_base[:hc_mult])
    pre = pre + hc_eps
    next_post = torch.sigmoid(
        gates[:, hc_mult:] * inv * hc_scale[1] + hc_base[hc_mult:]
    )
    next_post = magnitude * next_post + hc_eps

    # The fused kernel rounds the reduced row to the residual dtype before the
    # RMSNorm reads it, exactly like the unfused stage-2 kernel.
    reduced = (pre.unsqueeze(-1) * new_residual.float()).sum(1).to(residual.dtype)
    reduced_f = reduced.float()
    reduced = (
        reduced_f
        * torch.rsqrt(reduced_f.square().mean(-1, keepdim=True) + rms_eps)
        * rms_weight.float()
    ).to(residual.dtype)
    return new_residual, reduced, next_post


def _make_inputs(num_tokens, hidden_size, hc_mult, with_output, with_out_norm, dtype):
    device = torch.device("cuda")
    torch.manual_seed(0)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=dtype)
    residual *= 0.1
    output = None
    if with_output:
        output = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1
    post = torch.rand(num_tokens, hc_mult, device=device, dtype=torch.float32)
    hc_fn = torch.randn(2 * hc_mult, hc_mult * hidden_size, device=device) * 0.01
    hc_fn = hc_fn.to(torch.float32)
    hc_scale = torch.tensor([0.5, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(2 * hc_mult, device=device, dtype=torch.float32)
    rms_weight = torch.rand(hidden_size, device=device, dtype=dtype)
    out_norm_weight = (
        torch.rand(hidden_size, device=device, dtype=dtype) if with_out_norm else None
    )
    return dict(
        output=output,
        residual=residual,
        post=post,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        magnitude=2.0,
        norm_eps=1e-6,
        hc_eps=1e-6,
        rms_weight=rms_weight,
        rms_eps=1e-6,
        out_norm_weight=out_norm_weight,
        out_norm_eps=1e-6,
    )


# num_tokens < _IHC_TOKEN_BLOCKED_MIN_T is the per-token stage 1, then the
# token-blocked stage 1, then (from _IHC_DOT_MIN_T) the tl.dot stage 1.
# hc_mult 3 and 5 exercise the padded HC_POW2 lanes; 4 is the Hy4-Flash shape.
@pytest.mark.parametrize(
    "hidden_size,hc_mult", [(4096, 3), (4096, 4), (4096, 5), (6144, 4)]
)
@pytest.mark.parametrize("num_tokens", [1, 63, 64, 127, 128, 1024])
@pytest.mark.parametrize("with_output", [True, False])
def test_hy4_ihc_post_pre_norm_matches_reference(
    hidden_size, hc_mult, num_tokens, with_output
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    inputs = _make_inputs(
        num_tokens, hidden_size, hc_mult, with_output, False, torch.bfloat16
    )
    expected = _reference_post_pre_norm(**inputs)
    actual = hy4_ihc_post_pre_norm(**inputs)
    torch.cuda.synchronize()

    # The residual update is a plain elementwise bf16 expression: bit exact.
    assert torch.equal(actual[0], expected[0])
    # The reduced row is one bf16 rounding away from the fp32 reference.
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("hidden_size,hc_mult", [(4096, 4), (4096, 5)])
@pytest.mark.parametrize("num_tokens", [1, 63])
def test_hy4_ihc_post_pre_norm_fuses_out_norm(hidden_size, hc_mult, num_tokens):
    """The output RMSNorm is fused only for token counts where it is allowed."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert ihc_fuses_out_norm(num_tokens)
    assert num_tokens < _IHC_TOKEN_BLOCKED_MIN_T

    inputs = _make_inputs(num_tokens, hidden_size, hc_mult, True, True, torch.bfloat16)
    expected = _reference_post_pre_norm(**inputs)
    actual = hy4_ihc_post_pre_norm(**inputs)
    torch.cuda.synchronize()

    # The fused row RMS reduces in a different order than torch, so the
    # normalized output (and the residual built from it) is within a bf16 ulp.
    torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", [_IHC_TOKEN_BLOCKED_MIN_T, _IHC_DOT_MIN_T, 1024])
def test_hy4_ihc_post_pre_norm_rejects_out_norm_above_small_batch(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert not ihc_fuses_out_norm(num_tokens)

    hidden_size, hc_mult = 4096, 4
    inputs = _make_inputs(num_tokens, hidden_size, hc_mult, True, True, torch.bfloat16)
    with pytest.raises(ValueError, match="output RMSNorm"):
        hy4_ihc_post_pre_norm(**inputs)


def test_hy4_ihc_post_pre_norm_empty():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    hidden_size, hc_mult = 4096, 4
    inputs = _make_inputs(0, hidden_size, hc_mult, True, False, torch.bfloat16)
    residual, reduced, next_post = hy4_ihc_post_pre_norm(**inputs)
    assert residual.shape == (0, hc_mult, hidden_size)
    assert reduced.shape == (0, hidden_size)
    assert next_post.shape == (0, hc_mult)


def test_decoder_layer_ihc_pre_fusion_matches_deferred():
    """The model-level contract: hunyuan_v4_flash fuses the previous block's
    output RMSNorm into the iHC post+pre only while ihc_fuses_out_norm holds,
    and the deferred norm -> post -> pre chain must match it exactly."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    class TupleLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(output_size, input_size))

        def forward(self, inputs):
            return nn.functional.linear(inputs, self.weight), None

    hidden_size, hc_mult = 4096, 4
    config = SimpleNamespace(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    # The layer's own weights are random: only the two code paths are compared.
    with patch.object(hunyuan_v4, "ReplicatedLinear", TupleLinear):
        next_layer = hunyuan_v4.HYV4HCLayer(config, "next").cuda()
    norm = hunyuan_v4.RMSNorm(hidden_size, eps=1e-5, force_native=True).cuda()
    out_norm = hunyuan_v4.RMSNorm(hidden_size, eps=1e-5, force_native=True).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.rand(hidden_size, device="cuda", dtype=torch.bfloat16))
        out_norm.weight.copy_(
            torch.rand(hidden_size, device="cuda", dtype=torch.bfloat16)
        )

    torch.manual_seed(0)
    for num_tokens in (1, 7):
        residual = (
            torch.randn(
                num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )
        output = (
            torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.1
        )
        post = torch.rand(num_tokens, hc_mult, device="cuda", dtype=torch.float32)

        # The small-batch path: one launch, the out norm folded in.
        fused = hunyuan_v4_flash.HYV4FlashDecoderLayer._ihc_pre(
            next_layer, norm, residual, output, post, out_norm
        )
        # The deferred path the model takes above the small-batch threshold.
        deferred_residual = fused_hy4_ihc_post(out_norm(output), residual, post)
        deferred = hunyuan_v4_flash.HYV4FlashDecoderLayer._ihc_pre(
            next_layer, norm, deferred_residual
        )

        for fused_out, deferred_out in zip(fused, deferred):
            assert torch.equal(fused_out, deferred_out)
