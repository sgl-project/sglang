# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch
from torch.nn import functional as F

from sglang.kernels.ops.diffusion.norm.gated_residual_layernorm_modulate_triton import (
    can_use_gated_residual_layernorm_modulate,
    gated_residual_add,
    gated_residual_layernorm_modulate,
    layernorm_modulate,
)
from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# On (1, 1, 128) a single 1-ulp difference can break the error gate's max ratio.
LARGE_SHAPES = [(1, 2048, 4096), (2, 17, 3072), (1, 928, 6144)]
SHAPES = [(1, 1, 128), *LARGE_SHAPES]
ACTIVATIONS = ["randn", "outliers"]


def _inputs(shape, per_batch_dim, activations="randn"):
    torch.manual_seed(0)
    batch, _, hidden = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if activations == "outliers":
        # DiT-style: a constant offset plus a few large outlier channels.
        x = x + 4.0
        x[..., torch.arange(8, device="cuda") * (hidden // 8)] *= 32
    update = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    rows = (batch, 1, hidden) if per_batch_dim else (batch, hidden)
    gate = torch.randn(rows, device="cuda", dtype=torch.bfloat16).tanh()
    scale = torch.randn(rows, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(rows, device="cuda", dtype=torch.bfloat16) * 0.1
    return x, update, gate, scale, shift


def _per_batch(t):
    return t if t.dim() == 3 else t[:, None]


def _eager_modulation(x, scale, eps, shift=None):
    out = F.layer_norm(x, (x.shape[-1],), eps=eps) * (1 + _per_batch(scale))
    return out if shift is None else out + _per_batch(shift)


def _fp64_modulation(x, scale, eps, shift=None):
    out = F.layer_norm(x.double(), (x.shape[-1],), eps=eps) * (
        1 + _per_batch(scale).double()
    )
    return out if shift is None else out + _per_batch(shift).double()


def _assert_lossless(inp, scale, actual, label, shift=None):
    base, cand = assert_error_no_worse_than_reference(
        reference_fp64=_fp64_modulation(inp, scale, 1e-6, shift),
        baseline=_eager_modulation(inp, scale, 1e-6, shift),
        candidate=actual,
        label=label,
    )
    print(f"{label}: baseline {base} candidate {cand}")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("per_batch_dim", [True, False])
def test_gated_residual_add_matches_eager_bits(shape, per_batch_dim):
    x, update, gate, _, _ = _inputs(shape, per_batch_dim)
    assert can_use_gated_residual_layernorm_modulate(x, gate)
    expected = x + update * _per_batch(gate)
    actual = gated_residual_add(x, update, gate)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("shape", LARGE_SHAPES)
@pytest.mark.parametrize("per_batch_dim", [True, False])
@pytest.mark.parametrize("activations", ACTIVATIONS)
@pytest.mark.parametrize("with_shift", [False, True])
def test_layernorm_modulate_matches_eager(
    shape, per_batch_dim, activations, with_shift
):
    x, _, _, scale, shift = _inputs(shape, per_batch_dim, activations)
    shift = shift if with_shift else None
    actual = layernorm_modulate(x, scale, 1e-6, shift=shift)
    _assert_lossless(
        x,
        scale,
        actual,
        f"layernorm_modulate {shape} {per_batch_dim} {activations} shift={with_shift}",
        shift,
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("activations", ACTIVATIONS)
@pytest.mark.parametrize("with_shift", [False, True])
def test_gated_residual_layernorm_modulate_matches_eager(
    shape, activations, with_shift
):
    x, update, gate, scale, shift = _inputs(
        shape, per_batch_dim=True, activations=activations
    )
    shift = shift if with_shift else None
    residual = x + update * gate
    actual_residual, actual = gated_residual_layernorm_modulate(
        x, update, gate, scale, 1e-6, shift=shift
    )
    assert torch.equal(actual_residual.view(torch.int16), residual.view(torch.int16))
    if shape in LARGE_SHAPES:
        _assert_lossless(
            residual,
            scale,
            actual,
            f"gated_residual_layernorm_modulate {shape} {activations} shift={with_shift}",
            shift,
        )


def test_rejects_unsupported_layouts():
    x, update, gate, scale, _ = _inputs((2, 17, 128), per_batch_dim=True)
    assert not can_use_gated_residual_layernorm_modulate(x.float(), gate)
    assert not can_use_gated_residual_layernorm_modulate(x.transpose(1, 2), gate)
    assert not can_use_gated_residual_layernorm_modulate(x, gate[:1])
    assert not can_use_gated_residual_layernorm_modulate(x, scale.expand(2, 17, 128))


def test_entry_points_raise_on_unsupported_inputs():
    x, update, gate, scale, shift = _inputs((2, 17, 128), per_batch_dim=True)
    bad_inputs = [x.float(), x.transpose(1, 2)]
    for bad_x in bad_inputs:
        with pytest.raises(ValueError):
            layernorm_modulate(bad_x, scale, 1e-6)
        with pytest.raises(ValueError):
            gated_residual_add(bad_x, update, gate)
        with pytest.raises(ValueError):
            gated_residual_layernorm_modulate(bad_x, update, gate, scale, 1e-6)
    for bad_row in (scale[:1], scale.expand(2, 17, 128)):
        with pytest.raises(ValueError):
            layernorm_modulate(x, bad_row, 1e-6)
        with pytest.raises(ValueError):
            gated_residual_layernorm_modulate(x, update, gate, bad_row, 1e-6)
    for bad_shift in (shift[:1], shift.expand(2, 17, 128)):
        with pytest.raises(ValueError):
            layernorm_modulate(x, scale, 1e-6, shift=bad_shift)
        with pytest.raises(ValueError):
            gated_residual_layernorm_modulate(
                x, update, gate, scale, 1e-6, shift=bad_shift
            )
    for bad_gate in (gate[:1], gate.expand(2, 17, 128)):
        with pytest.raises(ValueError):
            gated_residual_add(x, update, bad_gate)
        with pytest.raises(ValueError):
            gated_residual_layernorm_modulate(x, update, bad_gate, scale, 1e-6)
    for bad_update in (update.float(), update.transpose(1, 2), update.cpu()):
        with pytest.raises(ValueError):
            gated_residual_add(x, bad_update, gate)
        with pytest.raises(ValueError):
            gated_residual_layernorm_modulate(x, bad_update, gate, scale, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
