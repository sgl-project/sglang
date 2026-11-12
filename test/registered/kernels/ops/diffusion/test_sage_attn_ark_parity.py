# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import sage_attn
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=45, suite="stage-b-test-1-gpu-xpu")

ark_sagev1 = sage_attn._load_ark_sagev1() if is_xpu() else None

pytestmark = pytest.mark.skipif(
    ark_sagev1 is None, reason="Intel XPU with a loadable auto-round-lib required"
)

# ARK and the Triton kernel quantize Q with 64- vs 128-row scales and differ by
# up to 1.8e-2; a 10% scale error or a flipped causal mask moves the output by >= 1.1e-1
_PARITY_TOL = 3e-2


def _sage(q, k, v, *, ark, causal, sm_scale):
    return sage_attn._xpu_sage_attn(
        q,
        k,
        v,
        ark_sagev1=ark,
        causal=causal,
        sm_scale=sm_scale,
        return_lse=False,
    )


def _assert_parity(q, k, v, *, causal, sm_scale):
    out_ark = _sage(q, k, v, ark=ark_sagev1, causal=causal, sm_scale=sm_scale)
    out_triton = _sage(q, k, v, ark=None, causal=causal, sm_scale=sm_scale)
    assert out_ark.shape == out_triton.shape == q.shape
    assert out_ark.dtype == out_triton.dtype == q.dtype
    # bit-identical outputs mean both calls ran the same kernel
    assert not torch.equal(out_ark, out_triton)
    diff = (out_ark.float() - out_triton.float()).norm() / out_triton.float().norm()
    assert diff.item() < _PARITY_TOL


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,qo_len,kv_len,heads,kv_heads,head_dim,causal,sm_scale",
    [
        (1, 4096, 4096, 12, 12, 128, False, None),
        (2, 1000, 1000, 8, 8, 64, False, None),  # partial tiles at both ends
        (1, 2048, 512, 12, 12, 128, False, None),  # cross-attention
        (1, 1024, 1024, 8, 2, 128, False, None),  # GQA
        (1, 1000, 1000, 8, 8, 64, True, None),
        (1, 1024, 1024, 8, 8, 128, False, 0.25),  # non-default softmax scale
    ],
)
def test_ark_matches_triton(
    batch, qo_len, kv_len, heads, kv_heads, head_dim, causal, sm_scale, dtype
):
    torch.manual_seed(0)
    q = torch.randn(batch, qo_len, heads, head_dim, device="xpu", dtype=dtype)
    # real keys carry a per-channel offset, which K smoothing removes
    k = torch.randn(batch, kv_len, kv_heads, head_dim, device="xpu", dtype=dtype)
    k = k + 0.5
    v = torch.randn(batch, kv_len, kv_heads, head_dim, device="xpu", dtype=dtype)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    _assert_parity(q, k, v, causal=causal, sm_scale=sm_scale)


def test_ark_matches_triton_on_strided_and_sliced_inputs():
    torch.manual_seed(0)
    qkv = torch.randn(1, 2048, 3, 12, 128, device="xpu", dtype=torch.bfloat16)
    q, k, v = qkv.unbind(dim=2)
    _assert_parity(q, k, v, causal=False, sm_scale=128**-0.5)

    # a sequence slice of a batch-1 tensor keeps the parent's batch stride
    full = torch.randn(3, 1, 3000, 12, 128, device="xpu", dtype=torch.bfloat16)
    q, k, v = (t[:, 500:2500] for t in full)
    _assert_parity(q, k, v, causal=False, sm_scale=128**-0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
