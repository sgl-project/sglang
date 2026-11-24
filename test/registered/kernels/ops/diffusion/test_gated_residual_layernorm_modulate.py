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
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

SHAPES = [(1, 1, 128), (1, 2048, 4096), (2, 17, 3072)]


def _inputs(shape, per_batch_dim):
    torch.manual_seed(0)
    batch, _, hidden = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    rows = (batch, 1, hidden) if per_batch_dim else (batch, hidden)
    gate = torch.randn(rows, device="cuda", dtype=torch.bfloat16).tanh()
    scale = torch.randn(rows, device="cuda", dtype=torch.bfloat16) * 0.1
    return x, update, gate, scale


def _per_batch(t):
    return t if t.dim() == 3 else t[:, None]


def _eager_modulation(x, scale, eps):
    return F.layer_norm(x, (x.shape[-1],), eps=eps) * (1 + _per_batch(scale))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("per_batch_dim", [True, False])
def test_gated_residual_add_matches_eager_bits(shape, per_batch_dim):
    x, update, gate, _ = _inputs(shape, per_batch_dim)
    assert can_use_gated_residual_layernorm_modulate(x, gate)
    expected = x + update * _per_batch(gate)
    actual = gated_residual_add(x, update, gate)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("per_batch_dim", [True, False])
def test_layernorm_modulate_matches_eager(shape, per_batch_dim):
    x, _, _, scale = _inputs(shape, per_batch_dim)
    expected = _eager_modulation(x, scale, 1e-6)
    actual = layernorm_modulate(x, scale, 1e-6)
    # only the mean/variance reduction order differs from the eager chain
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", SHAPES)
def test_gated_residual_layernorm_modulate_matches_eager(shape):
    x, update, gate, scale = _inputs(shape, per_batch_dim=True)
    residual = x + update * gate
    expected = _eager_modulation(residual, scale, 1e-6)
    actual_residual, actual = gated_residual_layernorm_modulate(
        x, update, gate, scale, 1e-6
    )
    assert torch.equal(actual_residual.view(torch.int16), residual.view(torch.int16))
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)


def test_rejects_unsupported_layouts():
    x, update, gate, scale = _inputs((2, 17, 128), per_batch_dim=True)
    assert not can_use_gated_residual_layernorm_modulate(x.float(), gate)
    assert not can_use_gated_residual_layernorm_modulate(x.transpose(1, 2), gate)
    assert not can_use_gated_residual_layernorm_modulate(x, gate[:1])
    assert not can_use_gated_residual_layernorm_modulate(x, scale.expand(2, 17, 128))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
