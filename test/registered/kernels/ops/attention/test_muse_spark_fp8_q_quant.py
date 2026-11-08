"""Muse Spark extend FP8 Q quantization (two launches) against the torch chain it replaces: same FP8 bytes and the
same descale, for TP8 / TP4 head counts, token counts off the 8192-element block grid, an all-zero query (descale
clamped to 1e-12) and wide-range values. One compiled variant per kernel, whatever the token count."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

F8 = torch.float8_e4m3fn


def _torch_chain(q):
    q_descale = (q.abs().amax().float() / 448.0).clamp_min(1e-12).view(1)
    return (q.float() / q_descale).to(F8), q_descale


def _assert_same(q):
    from sglang.kernels.ops.attention.fp8_q_quant import (
        fp8_q_quant,
        fp8_q_quant_supported,
    )

    assert fp8_q_quant_supported(q)
    out, descale = fp8_q_quant(q)
    ref, ref_descale = _torch_chain(q)
    assert out.shape == ref.shape and out.dtype == F8
    assert torch.equal(descale.view(torch.int32), ref_descale.view(torch.int32))
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("tokens", [1, 7, 255, 256, 1000, 4097, 16384, 40000])
def test_matches_torch_chain(heads, tokens):
    torch.manual_seed(tokens * 10 + heads)
    q = torch.randn(tokens, heads, 64, device="cuda").to(torch.bfloat16)
    _assert_same(q)


def test_zero_and_wide_range():
    _assert_same(torch.zeros(300, 16, 64, device="cuda", dtype=torch.bfloat16))
    torch.manual_seed(0)
    q = torch.randn(3000, 16, 64, device="cuda")
    q = q * torch.exp2(torch.randint(-30, 30, q.shape, device="cuda").float())
    _assert_same(q.to(torch.bfloat16))


def test_token_count_does_not_recompile(monkeypatch):
    import triton.knobs as knobs

    if not hasattr(getattr(knobs, "compilation", None), "listener"):
        pytest.skip("triton.knobs.compilation.listener is not available")
    from sglang.kernels.ops.attention.fp8_q_quant import fp8_q_quant

    compiled = []
    prev = knobs.compilation.listener

    def listener(*, src, **kwargs):
        if src.name in ("_q_absmax_kernel", "_q_quant_kernel"):
            compiled.append(src.name)
        if prev is not None:
            prev(src=src, **kwargs)

    monkeypatch.setattr(knobs.compilation, "listener", listener)
    for tokens in (1, 3, 16, 256, 257, 4096, 4103, 30001):
        fp8_q_quant(torch.randn(tokens, 16, 64, device="cuda").to(torch.bfloat16))
    torch.cuda.synchronize()
    assert len(compiled) <= 2, compiled


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
