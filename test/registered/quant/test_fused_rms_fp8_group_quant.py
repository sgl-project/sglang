import itertools
import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-a-test-1-amd")


def _fp8_available() -> bool:
    # requirement：1) GPU；2) ROCm；3) torch support float8_e4m3fn
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is None:
        return False
    return hasattr(torch, "float8_e4m3fn")


def _rmsnorm(x, weight, eps=1e-6):
    # row-wise RMSNorm
    row_norm = (x * x).sum(dim=-1)
    norm = torch.rsqrt(row_norm / x.shape[1] + eps)
    return x * norm[:, None] * weight[None, :]


def _per_token_fp8_group_quant(x, dtype_quant, group_size=128):
    """per token、group-size quant, return (quantized, scale)。"""
    DTYPE_MAX = torch.finfo(dtype_quant).max
    M, N = x.shape

    pad = (group_size - (N % group_size)) % group_size
    if pad:
        x_reshape = F.pad(x, (0, pad, 0, 0), "constant", 0)
    else:
        x_reshape = x

    G = (N + group_size - 1) // group_size
    x_reshape = x_reshape.view(M, G, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0].clamp_min_(1e-10)
    x_scale = x_max / DTYPE_MAX
    inv = 1.0 / x_scale

    x_q = torch.clamp(x_reshape * inv, -DTYPE_MAX, DTYPE_MAX).to(dtype_quant)
    x_q = x_q.view(M, G * group_size)
    if pad:
        x_q = x_q[:, :N]
    x_scale = x_scale.squeeze(-1)  # [M, G]
    return x_q, x_scale


def _upcast_fp8_group(x_q, x_s, out_dtype=torch.float32, group_size=128):
    """unqaunt"""
    M, N = x_q.shape
    G = (N + group_size - 1) // group_size
    pad = (group_size - (N % group_size)) % group_size

    if pad:
        x_q = F.pad(x_q, (0, pad, 0, 0), "constant", 0)

    x_q = x_q.view(M, G, group_size).to(torch.float32)
    x = x_q * x_s.view(M, G, 1)
    x = x.view(M, G * group_size)[:, :N]
    return x.to(out_dtype)


class TestFusedRMSFP8GroupQuant(CustomTestCase):
    #
    DTYPES = [torch.bfloat16, torch.float16]
    # (M, N1, N2)
    SHAPES = [(32, 128, 7168), (128, 7168, 7168)]
    GROUP_SIZE = [128]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not _fp8_available():
            raise unittest.SkipTest("Skip: ROCm/FP8 is not available")
        torch.set_default_device("cuda")

    def _run_ref(self, x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size):
        s = x1 + (res1 if res1 is not None else 0)
        y1 = _rmsnorm(s, w1, eps1)
        y2 = _rmsnorm(x2, w2, eps2) if x2 is not None else None
        y1_q, y1_s = _per_token_fp8_group_quant(y1, dtype_quant, group_size)
        return (
            (y1_q, y1_s),
            y1.to(x1.dtype),
            (y2.to(x1.dtype) if y2 is not None else None),
            (s.to(x1.dtype) if res1 is not None else None),
        )

    def _case(self, M, N1, N2, group_size, dtype, seed):
        torch.manual_seed(seed)
        fp8 = torch.float8_e4m3fn
        device = "cuda"

        x1 = torch.randn(M, N1, dtype=dtype, device=device) / 10
        x2 = torch.randn(M, N2, dtype=dtype, device=device) / 10
        w1 = torch.ones(N1, dtype=torch.float32, device=device)
        w2 = torch.ones(N2, dtype=torch.float32, device=device)
        res1 = torch.randn(M, N1, dtype=dtype, device=device) / 10

        # ref
        (y1_q_ref, y1_s_ref), y1_ref, y2_ref, s_ref = self._run_ref(
            x1, w1, 1e-6, x2, w2, 1e-6, res1, fp8, group_size
        )

        # be tested：aiter fused op
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

        (y1_q, y1_s), y1, y2, s = fused_rms_fp8_group_quant(
            x1,
            w1,
            1e-6,
            inp2=x2,
            inp2_weight=w2,
            inp2_epsilon=1e-6,
            group_size=group_size,
            dtype_quant=fp8,
            res1=res1,
            output_unquantized_inp1=True,  # get unqaunt y1
        )

        torch.testing.assert_close(y1, y1_ref, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y2, y2_ref, atol=0.1, rtol=0.1)
        torch.testing.assert_close(s, s_ref, atol=0.1, rtol=0.1)

        # check unquant
        y1_up_ref = _upcast_fp8_group(
            y1_q_ref, y1_s_ref, out_dtype=torch.float32, group_size=group_size
        )
        y1_up = _upcast_fp8_group(
            y1_q, y1_s, out_dtype=torch.float32, group_size=group_size
        )
        torch.testing.assert_close(y1_up, y1_up_ref, atol=0.1, rtol=0.1)

    def test_fused_rms_fp8_group_quant(self):
        for params in itertools.product(
            self.SHAPES, self.GROUP_SIZE, self.DTYPES, self.SEEDS
        ):
            (M, N1, N2), g, dtype, seed = params
            with self.subTest(M=M, N1=N1, N2=N2, group_size=g, dtype=dtype, seed=seed):
                self._case(M, N1, N2, g, dtype, seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
