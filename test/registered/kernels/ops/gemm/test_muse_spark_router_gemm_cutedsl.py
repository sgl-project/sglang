"""Muse Spark router logits on the CuTe DSL TF32 cluster split-K GEMM.

Every serving tactic of this GPU's table must match an fp64 reference within TF32 error
and be bitwise reproducible (the split-K slices are summed in a fixed order), at the
boundaries of each table range; ranges the table leaves to cuBLAS must stay ineligible.
Explicit tactics cover the kernel on SM10x GPUs without a serving table.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=120, stage="base-c", runner_config="8-gpu-b300")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.gemm import (  # noqa: E402
    muse_spark_router_gemm_cutedsl as router,
)

# Muse Spark router: 256 experts over the 8192-wide hidden state.
E, K = 256, 8192
CAPABILITY = torch.cuda.get_device_capability()
TABLE = router._SERVE_TABLES.get(CAPABILITY, ())
needs_table = pytest.mark.skipif(not TABLE, reason="no serving table for this GPU")


@pytest.fixture(autouse=True)
def _tf32_matmuls():
    # The server allows TF32 matmuls; the kernel is only eligible then.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(previous)


def _inputs(m: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(
        m, K, device="cuda", generator=torch.Generator("cuda").manual_seed(m)
    )
    w = torch.randn(
        E, K, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    return x, w * 0.02


def _check_against_fp64(x, w, tactic=None):
    got = router.router_tf32(x, w, tactic=tactic)
    again = router.router_tf32(x, w, tactic=tactic)
    ref = x.double() @ w.double().T
    torch.cuda.synchronize()
    # TF32 keeps 10 mantissa bits; over K = 8192 the measured max error is about 3e-4 of the max logit.
    rel = ((got.double() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-3, f"M={x.shape[0]}: max |err| / max |ref| = {rel:.2e}"
    assert torch.equal(got, again), f"M={x.shape[0]}: two runs differ"


def _table_edges() -> list[int]:
    """Both ends of every table range, plus M = 2 (the smallest eligible M)."""
    edges, low = {2}, 1
    for max_m, _ in TABLE:
        edges |= {max(low, 2), max_m}
        low = max_m + 1
    return sorted(edges)


@needs_table
@pytest.mark.parametrize("m", _table_edges())
def test_serving_tactics_match_fp64(m: int):
    x, w = _inputs(m)
    if not router.router_tf32_eligible(x, w):
        assert router._serve_overrides(m, CAPABILITY) is None, (
            f"M={m} has a tactic but is ineligible"
        )
        return
    _check_against_fp64(x, w)


@needs_table
def test_table_covers_eligible_range():
    """Every M up to SERVE_MAX_TOKENS resolves to a tactic or to cuBLAS (None)."""
    x_w = torch.empty(E, K, device="cuda")
    for m in range(2, router.SERVE_MAX_TOKENS + 1):
        overrides = router._serve_overrides(m, CAPABILITY)
        x = torch.empty(m, K, device="cuda")
        assert router.router_tf32_eligible(x, x_w) == (overrides is not None), f"M={m}"
        if overrides is not None:
            router.validate(router.serve_tactic(m, E, K, 0, CAPABILITY), m, E, K)


@needs_table
def test_highest_precision_keeps_fp32():
    """Under float32_matmul_precision "highest" the router logits must not be
    rounded to TF32: no M is eligible, so cuBLAS keeps them FP32."""
    torch.set_float32_matmul_precision("highest")
    w = torch.empty(E, K, device="cuda")
    for m in range(2, router.SERVE_MAX_TOKENS + 1):
        x = torch.empty(m, K, device="cuda")
        assert not router.router_tf32_eligible(x, w), f"M={m}"


# One explicit tactic per kernel variant, independent of any serving table: a segment
# shorter than one logits row, split 16 with deep K stages, a partial last token tile
# without PDL, and the 128-expert tile (32dp32b TMEM loads).
@pytest.mark.skipif(CAPABILITY[0] != 10, reason="SM10x required")
@pytest.mark.parametrize(
    "m, overrides",
    [
        (8, dict(split_k=16, cta_k=128, mma_n=8)),
        (64, dict(split_k=16, cta_k=128, mma_n=32)),
        (200, dict(split_k=4, cta_k=64, mma_n=64, pdl=False)),
        (512, dict(split_k=4, cta_k=32, mma_m=128, mma_n=64)),
    ],
)
def test_explicit_tactics_match_fp64(m: int, overrides: dict):
    x, w = _inputs(m)
    _check_against_fp64(x, w, router.default_tactic(m, E, K, **overrides))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
