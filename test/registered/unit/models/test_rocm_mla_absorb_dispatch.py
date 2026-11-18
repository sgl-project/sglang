"""Which kernel the ROCm MLA q and v absorb BMMs run.

The fp8 Triton kernel takes both absorbs on gfx950, and on gfx942 only under
CUDA graph capture (with aiter enabled). q and v must always take the same path.
"""

import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_KERNEL = "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
_HEADS, _KV_LORA_RANK, _QK_NOPE_HEAD_DIM, _V_HEAD_DIM = 4, 256, 128, 128
_W_SCALE = torch.tensor(0.25)
# The eager v absorb reads one exec flag; a unit test has no published exec config.
_EXEC = SimpleNamespace(kernel=SimpleNamespace(enable_amd_monokernel=False))


def _attn(w_dtype):
    gen = torch.Generator().manual_seed(0)

    def weight(*shape):
        return (torch.randn(*shape, generator=gen) * 0.5).to(w_dtype)

    return SimpleNamespace(
        w_kc=weight(_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK),
        w_vc=weight(_HEADS, _KV_LORA_RANK, _V_HEAD_DIM),
        w_scale=_W_SCALE,
        num_local_heads=_HEADS,
        o_proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)),
    )


def _fake_kernel(calls):
    """The kernel's math in fp32, without the activation quantization."""

    def kernel(*, X, WQ, w_scale, group_size, YQ, transpose_bm, transpose_bm_in, **_):
        calls.append(SimpleNamespace(YQ=YQ, transpose_bm=transpose_bm))
        x = X.float().transpose(0, 1) if transpose_bm_in else X.float()
        y = torch.einsum("bmk,bnk->bmn", x, WQ.float() * w_scale)
        y = y.transpose(0, 1) if transpose_bm else y
        if YQ is None:
            return y.to(torch.bfloat16)
        YQ.copy_(y)
        return YQ

    return kernel


class _Case(CustomTestCase):
    def patch(self, name, value):
        patcher = mock.patch.object(forward_mla_rocm, name, value, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def platform(self, *, aiter, gfx95):
        self.patch("_use_aiter", aiter)
        self.patch("_use_aiter_gfx95", gfx95)
        self.patch("_use_gfx1250_absorb_bmm", False)
        self.patch("_absorb_bmm_num_cu", 304)
        self.patch("get_exec", lambda: _EXEC)

    def kernel_calls(self):
        calls = []
        self.patch(_KERNEL, _fake_kernel(calls))
        return calls


class TestFp8AbsorbPredicate(_Case):
    def test_gfx942_only_under_capture(self):
        self.platform(aiter=True, gfx95=False)
        fnuz = _attn(torch.float8_e4m3fnuz)
        self.assertTrue(
            forward_mla_rocm._use_fp8_absorb_bmm(fnuz, is_capture_mode=True)
        )
        self.assertFalse(
            forward_mla_rocm._use_fp8_absorb_bmm(fnuz, is_capture_mode=False)
        )

    def test_gfx950_regardless_of_capture(self):
        self.platform(aiter=True, gfx95=True)
        for capture in (True, False):
            with self.subTest(capture=capture):
                self.assertTrue(
                    forward_mla_rocm._use_fp8_absorb_bmm(
                        _attn(torch.float8_e4m3fn), is_capture_mode=capture
                    )
                )

    def test_bf16_weights_never(self):
        for gfx95, capture in itertools.product((True, False), repeat=2):
            with self.subTest(gfx95=gfx95, capture=capture):
                self.platform(aiter=True, gfx95=gfx95)
                self.assertFalse(
                    forward_mla_rocm._use_fp8_absorb_bmm(
                        _attn(torch.bfloat16), is_capture_mode=capture
                    )
                )
            self.doCleanups()

    def test_without_aiter_never(self):
        """The kernel is only imported with aiter enabled."""
        self.platform(aiter=False, gfx95=False)
        self.assertFalse(
            forward_mla_rocm._use_fp8_absorb_bmm(
                _attn(torch.float8_e4m3fnuz), is_capture_mode=True
            )
        )


class TestAbsorbDispatch(_Case):
    def absorb_q(self, attn, capture, tokens=3):
        q_nope = torch.randn(tokens, _HEADS, _QK_NOPE_HEAD_DIM).to(torch.bfloat16)
        return forward_mla_rocm.rocm_absorb_q_bmm(attn, q_nope, is_capture_mode=capture)

    def absorb_v(self, attn, capture, tokens=3):
        gen = torch.Generator().manual_seed(1)
        attn_output = torch.randn(tokens, _HEADS, _KV_LORA_RANK, generator=gen).to(
            torch.bfloat16
        )
        return forward_mla_rocm.rocm_absorb_v_bmm(
            attn, attn_output, is_capture_mode=capture
        )

    def test_q_and_v_take_the_same_path(self):
        """The regression guard: every platform, dtype and capture state routes
        both absorbs to the same kernel."""
        for aiter, gfx95, w_dtype, capture in itertools.product(
            (True, False),
            (True, False),
            (torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.bfloat16),
            (True, False),
        ):
            if gfx95 and not aiter:
                continue
            with self.subTest(aiter=aiter, gfx95=gfx95, w=w_dtype, capture=capture):
                self.platform(aiter=aiter, gfx95=gfx95)
                calls = self.kernel_calls()
                attn = _attn(w_dtype)
                self.absorb_q(attn, capture)
                q_fp8 = bool(calls)
                calls.clear()
                self.absorb_v(attn, capture)
                self.assertEqual(bool(calls), q_fp8)
                self.assertEqual(
                    q_fp8,
                    forward_mla_rocm._use_fp8_absorb_bmm(attn, is_capture_mode=capture),
                )
            self.doCleanups()

    def test_gfx942_capture_v_writes_token_major(self):
        """Under capture the fp8 v absorb writes (tokens, heads, v_head_dim), so
        the flatten for o_proj is a view, and matches the bf16 path it replaces."""
        self.platform(aiter=True, gfx95=False)
        calls = self.kernel_calls()
        attn = _attn(torch.float8_e4m3fnuz)
        tokens = 5

        captured = self.absorb_v(attn, True, tokens)
        (call,) = calls
        self.assertEqual(tuple(call.YQ.shape), (tokens, _HEADS, _V_HEAD_DIM))
        self.assertTrue(call.transpose_bm)
        self.assertEqual(captured.data_ptr(), call.YQ.data_ptr())

        eager = self.absorb_v(attn, False, tokens)
        self.assertEqual(len(calls), 1)
        self.assertEqual(captured.shape, (tokens, _HEADS * _V_HEAD_DIM))
        torch.testing.assert_close(captured, eager, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
