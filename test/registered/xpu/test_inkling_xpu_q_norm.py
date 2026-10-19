"""Inkling attention on Intel XPU: import, AR fallback, and the in-place q norm.

The first two cases guard the XPU unblocking fixes and run against any
sgl-kernel-xpu wheel:

  A. `inkling_common.attn` imports on XPU, where `cute/__init__` pulls in the
     CUDA-only `cutlass` and `is_batch_invariant` has to fall back.
  B. The custom all-reduce resource lookup returns None on XPU instead of
     raising from `torch.cuda.is_current_stream_capturing()`.

The rest drive `InklingAttention._xpu_norm_q_inplace`, which norms q in place on
the packed QKVR row through `sgl_kernel.rmsnorm_heads_inplace`. They skip when
the installed wheel predates that op; the op's own contract (tail untouched,
rotary off, refusals) is tested in sgl-kernel-xpu.

This file lives under `test/registered/xpu/` and registers XPU CI only, so no
per-class device skip is needed -- see the `register_xpu_ci` BKM. Do not add
`register_cuda_ci` here: CI collects by file, and the fused path is XPU-only.
"""

import types
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.models.inkling_common import attn
from sglang.srt.models.inkling_common.attn import InklingAttention, xpu_qkvr_width
from sglang.srt.models.inkling_common.kernels import comm
from sglang.srt.models.inkling_common.norm import RMSNorm
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"
HEAD_DIM = 128
D_REL = 16
EPS = 1e-6

# (num_tp_heads, num_tp_kv_heads) for Inkling's two layer groups at TP=8/TP=4.
ADMITTED = {
    "swa_tp8": (8, 2),
    "full_tp8": (8, 1),
    "swa_tp4": (16, 4),
    "full_tp4": (16, 2),
}
TOLERANCE = {
    torch.bfloat16: dict(rtol=8e-3, atol=1e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-4),
}
HAS_OP = attn._xpu_rmsnorm_heads_inplace is not None


class _Attn:
    """Binds the real `_xpu_norm_q_inplace` to the attributes it reads.

    Instantiating `InklingAttention` needs a full model config and an
    initialized distributed group, so the method under test is bound onto a stub
    carrying only what it touches. The row width comes from the production
    helper rather than a copy of it.
    """

    norm_q_inplace = InklingAttention._xpu_norm_q_inplace

    def __init__(self, num_tp_heads, num_tp_kv_heads, dtype):
        self.num_tp_heads = num_tp_heads
        self.num_tp_kv_heads = num_tp_kv_heads
        self.head_dim = HEAD_DIM
        self._xpu_qkvr_width = xpu_qkvr_width(
            head_dim=HEAD_DIM,
            num_tp_heads=num_tp_heads,
            num_tp_kv_heads=num_tp_kv_heads,
            d_rel=D_REL,
        )
        self.q_norm = RMSNorm(HEAD_DIM, eps=EPS).to(DEVICE).to(dtype)
        with torch.no_grad():
            self.q_norm.weight.normal_(1.0, 0.05)

    def project(self, num_tokens, dtype, seed=0):
        """A packed row and its q/k/v/r views, split exactly as _project_qkvr does."""
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        row = torch.randn(
            num_tokens, self._xpu_qkvr_width, generator=g, device=DEVICE, dtype=dtype
        )
        sizes = [
            HEAD_DIM * self.num_tp_heads,
            HEAD_DIM * self.num_tp_kv_heads,
            HEAD_DIM * self.num_tp_kv_heads,
            D_REL * self.num_tp_heads,
        ]
        return row, row.split(sizes, dim=-1)


class TestInklingXpuUnblock(CustomTestCase):
    def test_attention_module_imports_on_xpu(self):
        # Reaching here means the module-level import already succeeded.
        self.assertFalse(attn.is_batch_invariant())

    def test_ar_resources_fall_back_on_xpu(self):
        stub = types.SimpleNamespace(
            group=types.SimpleNamespace(group_name="g"), world_size=8
        )
        self.assertIsNone(comm._get_inkling_ar_resources(stub))


@unittest.skipUnless(HAS_OP, "installed sgl-kernel-xpu lacks rmsnorm_heads_inplace")
class TestInklingXpuQNorm(CustomTestCase):
    def test_admits_inkling_geometries(self):
        for name, (heads, kv_heads) in ADMITTED.items():
            width = xpu_qkvr_width(
                head_dim=HEAD_DIM,
                num_tp_heads=heads,
                num_tp_kv_heads=kv_heads,
                d_rel=D_REL,
            )
            with self.subTest(name):
                self.assertTrue(
                    attn._xpu_rmsnorm_heads_inplace_supported(
                        row_width=width,
                        num_heads=heads,
                        head_dim=HEAD_DIM,
                        dtype=torch.bfloat16,
                    )
                )

    def test_norms_q_on_the_row_and_leaves_k_v_r(self):
        for name, (heads, kv_heads) in ADMITTED.items():
            for dtype in TOLERANCE:
                for num_tokens in (1, 8, 512):
                    with self.subTest(name, dtype=dtype, num_tokens=num_tokens):
                        a = _Attn(heads, kv_heads, dtype)
                        row, (q, k, v, r) = a.project(num_tokens, dtype)
                        before = row.clone()
                        self.assertTrue(a.norm_q_inplace(q))
                        q_width = HEAD_DIM * heads
                        ref = F.rms_norm(
                            before[:, :q_width].reshape(-1, HEAD_DIM).float(),
                            (HEAD_DIM,),
                            a.q_norm.weight.float(),
                            EPS,
                        ).to(dtype)
                        # q is still a view of the row, so the row holds the result.
                        torch.testing.assert_close(
                            q.reshape(-1, HEAD_DIM), ref, **TOLERANCE[dtype]
                        )
                        self.assertTrue(
                            torch.equal(row[:, q_width:], before[:, q_width:])
                        )

    def test_declines_q_that_is_not_on_the_row(self):
        heads, kv_heads = ADMITTED["swa_tp8"]
        a = _Attn(heads, kv_heads, torch.bfloat16)
        row, (q, _, _, _) = a.project(8, torch.bfloat16)
        before = row.clone()
        compact = q.contiguous()
        self.assertFalse(a.norm_q_inplace(compact))
        fp32_row, (fp32_q, _, _, _) = a.project(8, torch.float32)
        self.assertFalse(a.norm_q_inplace(fp32_q))
        self.assertTrue(torch.equal(row, before))


if __name__ == "__main__":
    unittest.main()
