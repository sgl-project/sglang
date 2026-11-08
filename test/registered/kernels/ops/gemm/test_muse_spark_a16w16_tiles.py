"""Muse Spark aiter Triton a16w16 tiles (TP8 DFlash draft gate/up / down at 8 / 16 rows, the O projection at 8):
each tile computes the bytes of aiter's default config, and a registered (M, N, K) config reaches the kernel
through UnquantizedLinearMethod.apply while other shapes keep the default."""

import pytest
import torch
from torch import nn

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)


def _tiles():
    from sglang.srt.rocm_muse_spark.model import _TRITON_A16W16_TILES

    assert _TRITON_A16W16_TILES, "SGLANG_MUSE_A16W16_TILES is on by default"
    return _TRITON_A16W16_TILES


@pytest.mark.parametrize("shape", sorted(_tiles()) if is_hip() else [])
def test_tile_matches_default_config(shape):
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

    m, n, k = shape
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    config = compute_splitk_params(dict(_tiles()[shape]), k)
    tiled = gemm_a16w16(x, w, None, dtype=torch.bfloat16, config=config)
    default = gemm_a16w16(x, w, None, dtype=torch.bfloat16)
    assert torch.equal(tiled, default)
    ref = x.double() @ w.double().t()
    assert ((tiled.double() - ref).norm() / ref.norm()).item() < 1e-2


def test_registered_config_reaches_the_kernel(monkeypatch):
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

    from sglang.srt.layers.quantization import unquant

    calls = []

    def fake_a16w16(x, w, bias=None, dtype=None, config=None):
        calls.append((tuple(x.shape), tuple(w.shape), config))
        return x.new_zeros(x.shape[0], w.shape[0])

    monkeypatch.setattr(unquant, "_use_aiter", True)
    monkeypatch.setattr(unquant, "_skip_aiter_unquant", False)
    monkeypatch.setattr(unquant, "_aiter_triton_a16w16", fake_a16w16, raising=False)
    monkeypatch.setattr(
        unquant, "compute_splitk_params", compute_splitk_params, raising=False
    )
    monkeypatch.setattr(unquant, "_gfx950_triton_a16w16_shapes", set())
    monkeypatch.setattr(unquant, "_gfx950_triton_a16w16_configs", {})

    n, k = 8192, 1024
    tile = _tiles()[(8, n, k)]
    unquant.register_gfx950_unquantized_gemm_shapes(
        triton_a16w16_configs={(8, n, k): tile}
    )

    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.zeros(n, k, device="cuda", dtype=torch.bfloat16), requires_grad=False
    )
    method = unquant.UnquantizedLinearMethod()
    for m in (8, 4):
        method.apply(layer, torch.zeros(m, k, device="cuda", dtype=torch.bfloat16))
    (m8, w8, config8), (m4, w4, config4) = calls
    assert m8 == (8, k) and w8 == (n, k)
    assert config8["BLOCK_SIZE_M"] == tile["BLOCK_SIZE_M"]
    assert config8["BLOCK_SIZE_N"] == tile["BLOCK_SIZE_N"]
    assert m4 == (4, k) and config4 is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
