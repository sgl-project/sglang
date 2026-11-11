"""The launch recorder on the block-scaled CuTeDSL providers (SM100).

The recorder test reaches these providers through one runner test, as a
launch structure without operand bytes. Here every block-scaled encoding
(MXFP8, MXFP4, NVFP4 with a static activation scale and with per-token
scales) runs the recorder test's base pipeline twice on identical inputs, on
masked and on contiguous rows, with dropped pairs and padding rows in the
batch, and with the down input written by the activation kernel or by the
separate quantizer. At every launch boundary the valid data of every declared
buffer must be the same in both forwards, by logical pair, and so must the
outputs.

A run that keeps a junit record keeps the cases beside it, as the recorder
test does; two such files compare with
``python -m sglang.test.kernels.moe_lora_launch_recorder before after``.
"""

from __future__ import annotations

import pytest
import test_moe_lora_blockscaled_provider_numerics as B
import test_moe_lora_launch_recorder as L
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import cutedsl_ready

# The block-scaled kernels are SM100 only; one GPU of the pool is used.
register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not cutedsl_ready(), reason="the cases run the CuTeDSL providers"
)

# Relative L2 distance from the reference that quantizes both GEMM inputs with
# the provider's rule (the bound of the provider test this reuses).
_TOLERANCE = 1.5e-2


@pytest.fixture(
    scope="module",
    params=[
        (family, rows, separate_quant)
        for family in ("mxfp8", "mxfp4", "nvfp4", "nvfp4_pt")
        for rows in ("expert_major", "route_major")
        for separate_quant in (False, True)
    ],
    ids=lambda p: (
        f"{p[0]}-{'masked' if p[1] == 'expert_major' else 'contiguous'}"
        f"-{'separate' if p[2] else 'activation'}-quant"
    ),
)
def blockscaled_case(request):
    B._skip_unless_sm100()
    from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
    from sglang.srt.runtime_context import get_context

    family, rows, separate_quant = request.param
    generator = torch.Generator().manual_seed(1631)
    w13 = (torch.randn(B._E, 2 * B._I, B._H, generator=generator) * 0.05).to(
        device="cuda", dtype=torch.bfloat16
    )
    w2 = (torch.randn(B._E, B._H, B._I, generator=generator) * 0.05).to(
        device="cuda", dtype=torch.bfloat16
    )
    info, w13_eff, w2_eff, elem = B._quant_info(family, w13, w2)
    with get_context().override_server_args():
        provider = select_provider_cls(rows, B._family_and_mode(family)[0], "cutedsl")(
            info
        )
        yield provider, w13_eff, w2_eff, elem, separate_quant


def _check(output, hidden, ids, weights, w13, w2, elem) -> None:
    """Finite, zero for a token without an expert, and the reference on the
    dequantized weights with both GEMM inputs quantized as the provider does."""
    inter = w2.shape[2]
    hidden_eff = B._quant_activation(hidden.float(), elem, "gemm1")
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(w13.shape[0]):
        coefficients = (weights * (ids == expert)).sum(dim=1)
        gateup = (hidden_eff @ w13[expert].float().T).to(torch.bfloat16).float()
        act = (F.silu(gateup[:, :inter]) * gateup[:, inter:]).to(torch.bfloat16)
        act_eff = B._quant_activation(act.float(), elem, "gemm2")
        down = (act_eff @ w2[expert].float().T).to(torch.bfloat16).float()
        expected.add_(down * coefficients[:, None])
    expected *= 0.75
    actual = output.float()
    assert bool(torch.isfinite(actual).all())
    distance = (actual - expected).norm() / expected.norm().clamp(min=1e-12)
    assert distance.item() < _TOLERANCE, distance.item()
    assert bool((actual[(ids < 0).all(dim=1)] == 0).all())


def test_blockscaled_forwards_agree_at_every_launch_boundary(
    blockscaled_case, record_property, request
) -> None:
    provider, w13, w2, elem, separate_quant = blockscaled_case
    L.fresh_forwards_case(
        provider,
        w13,
        w2,
        elem,
        record_property,
        request,
        kinds={"triton", "cutedsl"},
        fixture=dict(
            seed=L._SEED,
            tokens=L._TOKENS,
            traffic="padded",
            case=request.node.callspec.id,
        ),
        separate_quant=separate_quant,
        check=_check,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
