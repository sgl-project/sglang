"""One example of the inherited limit of the 4over6 strict candidate error,
reproducible on CPU. flashinfer's strict form dequantizes a candidate as
``((code * sf) * row_amax) / D`` in fp32 and the per-token quantizer mirrors
that order (the semantics are flashinfer's, see ``_candidate_error`` in
``quant_blockscaled.py``). A lane's product overflows once ``|code * sf| *
row_amax`` exceeds fp32_max, which happens for values above about fp32_max /
D (1.3e35 under e4m3_max 448, 2.2e35 under 256, inside BF16's range), while
smaller lanes of the same row stay finite; under MSE the squared difference
overflows earlier, and the in-order lane sum can overflow on finite terms. An
infinite error cannot win ``err4 < err6``; when both are infinite sf6 is kept
whatever the reconstruction. The block here has all three nonzero values
above the bound, so both errors are inf and sf6 stays where sf4 is exact: a
specific case, not the general rule. The fast form compares values bounded by
D (``x * min(D / row_amax, fp32_max)`` against ``code * sf``) and does not
overflow. The GPU test ``test_strict_error_overflow_keeps_sf6`` (the recipe
numerics file) pins the kernel on the same block; this file keeps the
arithmetic readable without a GPU."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FLT_MAX = torch.finfo(torch.float32).max
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_BLOCK = [1536.0, 576.0, 1152.0] + [0.0] * 13  # the ``above_256`` recipe case


def _e2m1_round(q: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the E2M1 magnitudes (``cvt.rn.satfinite``)."""
    magnitude = q.abs().clamp(max=6.0)
    distance = (magnitude[:, None] - _E2M1[None, :]).abs()
    nearest = distance.argmin(dim=1)
    tie = distance.min(dim=1).values[:, None] == distance
    even = tie & (torch.arange(8) % 2 == 0)[None, :]
    nearest = torch.where(even.any(dim=1), even.float().argmax(dim=1), nearest)
    return torch.copysign(_E2M1[nearest], q)


def _select(x: torch.Tensor, e4m3_max: int, fast: bool):
    """One block's 4over6 selection in the arithmetic the kernels use: fp32
    products and divisions, the E4M3 cast saturating at 448, the MAE summed
    over the lanes. Returns the chosen scale and each candidate's error."""
    denominator = torch.tensor(6.0 * e4m3_max)
    row_amax = x.abs().max()
    s = row_amax / denominator  # the stored per-token scale
    sf6 = x.abs().max() / (6.0 * s)
    encode = torch.minimum(denominator / row_amax, torch.tensor(_FLT_MAX))
    errors, scales = [], []
    for sf in (sf6, sf6 * 1.5):
        decoded = sf.clamp(max=448.0).to(torch.float8_e4m3fn).float()
        values = _e2m1_round((x / s) / decoded)
        candidate = values * decoded
        if fast:
            diff = candidate - x * encode
        else:
            diff = ((candidate * row_amax) / denominator) - x
        errors.append(diff.abs().sum())
        scales.append(decoded)
    use4 = errors[1] < errors[0]
    return scales[int(use4)], errors


def _reconstruction_error(x: torch.Tensor, e4m3_max: int, scale: torch.Tensor):
    """Float64 absolute error of the block quantized at ``scale``."""
    s = (x.abs().max() / torch.tensor(6.0 * e4m3_max)).double()
    values = _e2m1_round(((x / s.float()) / scale).float()).double()
    return float((values * scale.double() * s - x.double()).abs().sum())


@pytest.mark.parametrize("exponent", (110, 100), ids=["2^110", "2^100"])
def test_strict_error_overflow_keeps_sf6(exponent: int):
    """The block times 2^110 (its nonzero values 7.5e35 to 2.0e36, each above
    fp32_max / 1536 = 2.2e35) under 256: every nonzero lane overflows for
    both candidates, the strict errors are inf and sf6 = 256 stays with a
    reconstruction error of 192 * 2^110, while sf4 = 384 is exact and the
    fast error selects it. The same block times 2^100 (values below 2.0e33)
    stays finite and both forms select sf4."""
    x = (torch.tensor(_BLOCK) * 2.0**exponent).to(torch.bfloat16).float()
    assert float(x[0]) == 1536.0 * 2.0**exponent  # BF16 holds the row
    scale_strict, strict = _select(x, 256, fast=False)
    scale_fast, fast = _select(x, 256, fast=True)
    assert float(scale_fast) == 384.0 and float(fast[1]) == 0.0 < float(fast[0])
    exact = _reconstruction_error(x, 256, torch.tensor(384.0))
    coarse = _reconstruction_error(x, 256, torch.tensor(256.0))
    assert exact == 0.0 and coarse == 192.0 * 2.0**exponent
    if x.abs().max() > _FLT_MAX / 1536:
        assert all(bool(torch.isinf(e)) for e in strict)
        assert float(scale_strict) == 256.0  # kept although sf4 is exact
    else:
        assert all(bool(torch.isfinite(e)) for e in strict)
        assert float(scale_strict) == 384.0


def test_under_448_the_candidates_coincide():
    """Under 448 the same row's sf6 is 448 and sf4 saturates to 448: the
    overflow still makes both strict errors inf, but there is nothing to
    decide."""
    x = (torch.tensor(_BLOCK) * 2.0**110).to(torch.bfloat16).float()
    scale, strict = _select(x, 448, fast=False)
    assert float(scale) == 448.0 and all(bool(torch.isinf(e)) for e in strict)
    scale, fast = _select(x, 448, fast=True)
    assert float(scale) == 448.0 and float(fast[0]) == float(fast[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
