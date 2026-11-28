"""ROCm tests for the decode-sized MLA v absorb."""

import pytest
import torch

from sglang.kernels.ops.attention.dsa import mla_v_absorb
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="base-a-jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")

_H, _K, _N = 12, 512, 128


def _w_vc(layout: str) -> torch.Tensor:
    w = torch.randn(_H, _N, _K, device="cuda", dtype=torch.bfloat16)
    # DeepSeek-style loaders (Kimi-K3, DFLASH's MLA draft) keep w_vc as the
    # transpose of a contiguous [heads, v_head_dim, kv_lora_rank] tensor.
    return (
        w.transpose(1, 2) if layout == "transposed" else w.transpose(1, 2).contiguous()
    )


@pytest.mark.parametrize("layout", ["transposed", "contiguous"])
@pytest.mark.parametrize("tokens", [1, 6, 8, 64])
def test_matches_bmm(layout, tokens):
    torch.manual_seed(0)
    w_vc = _w_vc(layout)
    # attn_output only needs a contiguous last dim
    x = torch.randn(tokens, _H, _K + 64, device="cuda", dtype=torch.bfloat16)[..., :_K]
    assert mla_v_absorb.covered(x, w_vc)

    out = torch.empty(tokens, _H, _N, device="cuda", dtype=torch.bfloat16)
    mla_v_absorb.mla_v_absorb(x, w_vc, out)

    ref = torch.bmm(x.float().transpose(0, 1), w_vc.float()).transpose(0, 1)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=5e-2)


def test_covered_rejects():
    w_vc = _w_vc("transposed")
    too_many = torch.randn(
        mla_v_absorb.MAX_TOKENS + 1, _H, _K, device="cuda", dtype=torch.bfloat16
    )
    assert not mla_v_absorb.covered(too_many, w_vc)
    x = torch.randn(8, _H, _K, device="cuda", dtype=torch.bfloat16)
    assert not mla_v_absorb.covered(x, w_vc.float())
    # neither the K nor the N axis of w_vc is unit-stride
    strided = torch.randn(_H, _K, 2 * _N, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert not mla_v_absorb.covered(x, strided)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
