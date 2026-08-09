"""Sana fused LN+modulate fast path must stay bit-exact vs eager."""

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.sana as sana
from sglang.multimodal_gen.runtime.models.dits.sana import (
    _eager_ln_modulate,
    _sana_ln_modulate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "shape,nmod,transposed",
    [
        ((2, 1024, 2240), 6, False),
        ((2, 1024, 2240), 2, False),
        ((1, 1024, 2240), 6, True),
        ((1, 37, 2240), 6, False),
    ],
)
def test_sana_fused_ln_modulate_is_bit_exact(shape, nmod, transposed):
    # (., 1024, 2240) is the real Sana 1024px shape; hidden 2240 % 512 != 0
    # exercises the kernel's partial tail chunk.  nmod mirrors the two adaLN
    # chunk layouts, transposed the permuted layout the Sana DiT serves.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = torch.nn.LayerNorm(hidden, eps=1e-6, elementwise_affine=False).cuda()
    x = (torch.randn(batch, seq, hidden, device="cuda") * 4).bfloat16()
    if transposed:
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    emb = torch.randn(batch, nmod, hidden, device="cuda").bfloat16()
    shift, scale = emb.chunk(nmod, dim=1)[0], emb.chunk(nmod, dim=1)[-1]
    # default-stream eager serving must stay on the untouched eager chain
    n_sigs = len(sana._SANA_FUSED_LN_MOD_OK_SIGS)
    _sana_ln_modulate(norm, x, scale, shift)
    assert len(sana._SANA_FUSED_LN_MOD_OK_SIGS) == n_sigs
    # the fusion engages on non-default streams (the BCG warmup/capture path)
    with torch.cuda.stream(torch.cuda.Stream()):
        out = _sana_ln_modulate(norm, x, scale, shift)
        assert len(sana._SANA_FUSED_LN_MOD_OK_SIGS) == n_sigs + 1  # verified
        out2 = _sana_ln_modulate(norm, x, scale, shift)  # verified-sig lane
    torch.cuda.synchronize()
    assert torch.equal(out, _eager_ln_modulate(norm, x, scale, shift))
    assert torch.equal(out2, out) and not sana._SANA_FUSED_LN_MOD_DISABLED


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
