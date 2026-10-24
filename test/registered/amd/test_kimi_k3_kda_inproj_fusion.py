"""Layout check for the ROCm fused Kimi-K3 KDA input projection.

Below SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS the whole in-proj is one GEMM
over ``[q,k,v,g | f_a | b | pad]`` instead of a wide GEMM plus a tiny [f_a|b]
GEMV. Both layouts are views over the same buffer, so the two paths have to
agree; this pins the slice offsets, the tail view the split path still reads,
and the fact that the strided f_a slice is a legal input to the f_b GEMM and
to the fused decode kernel's shape gate.

The two paths run different GEMM kernels (N=6288 has no tuned aiter config,
N=6144 does), so they agree to bf16 rounding, not bitwise.
"""

import unittest
import unittest.mock
from types import SimpleNamespace

import torch

from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    _merge_weights_as_views,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN = 7168
HEADS_TP = 12  # num_heads / tp8
HEAD_DIM = 128
PROJ_TP = HEADS_TP * HEAD_DIM  # 1536
WIDE = 4 * PROJ_TP  # 6144, [q,k,v,g]
MERGED = 6288  # + f_a(128) + b(12) + pad(4)
# bf16 carries ~8 mantissa bits, so one ULP is ~4e-3 relative.
TOL = 8e-3


class _Fake(torch.nn.Module):
    """Minimal stand-in for a linear layer: _merge_weights_as_views only
    touches .weight.data."""

    def __init__(self, rows, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3KDAInProjFusion(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = torch.device("cuda", 0)
        cls.qkvg = _Fake(WIDE, dev)
        cls.f_a = _Fake(HEAD_DIM, dev)
        cls.b = _Fake(HEADS_TP, dev)
        cls.pre = [m.weight.data.clone() for m in (cls.qkvg, cls.f_a, cls.b)]
        cls.f_b_w = (
            torch.randn(PROJ_TP, HEAD_DIM, dtype=torch.bfloat16, device=dev) * 0.02
        )
        cls.merged, cls.sizes = _merge_weights_as_views(
            [cls.qkvg, cls.f_a, cls.b], pad_rows_to=8
        )
        cls.split_sizes = [3 * PROJ_TP, PROJ_TP]
        cls.all_sizes = cls.split_sizes + [
            HEAD_DIM,
            HEADS_TP,
            MERGED - WIDE - HEAD_DIM - HEADS_TP,
        ]

    def test_merged_layout(self):
        self.assertEqual(self.sizes, [WIDE, HEAD_DIM, HEADS_TP])
        self.assertEqual(tuple(self.merged.shape), (MERGED, HIDDEN))
        self.assertEqual(sum(self.all_sizes), MERGED)

    def test_views_alias_and_preserve_values(self):
        """The merge must re-point, not reorder or copy-and-drop."""
        tail = self.merged[WIDE:]
        self.assertEqual(tail.data_ptr(), self.f_a.weight.data_ptr())
        self.assertTrue(tail.is_contiguous())
        self.assertTrue(self.qkvg.weight.is_contiguous())
        self.assertEqual(self.qkvg.weight.data_ptr(), self.merged.data_ptr())
        for got, want in zip((self.qkvg, self.f_a, self.b), self.pre):
            self.assertTrue(torch.equal(got.weight.data, want))

    def test_split_and_fused_paths_agree(self):
        tail = self.merged[WIDE:]
        for tokens in (1, 4, 8, 33, 128, 256):
            with self.subTest(tokens=tokens):
                x = torch.randn(
                    tokens, HIDDEN, dtype=torch.bfloat16, device=self.merged.device
                )

                wide = torch.nn.functional.linear(x, self.qkvg.weight)
                s_qkv, s_g = torch.split(wide, self.split_sizes, dim=-1)
                s_bfa = kimi_k3_tiny_gemm(x, tail)
                s_beta = s_bfa[..., HEAD_DIM : HEAD_DIM + HEADS_TP]
                s_fg = kimi_k3_tiny_gemm(s_bfa[..., :HEAD_DIM], self.f_b_w)

                allp = torch.nn.functional.linear(x, self.merged)
                f_qkv, f_g, f_fa, f_beta, _pad = torch.split(
                    allp, self.all_sizes, dim=-1
                )
                # The fused decode kernel's shape gate requires a unit last
                # stride but tolerates the wider row stride.
                self.assertEqual(f_fa.stride(-1), 1)
                self.assertEqual(f_qkv.stride(-1), 1)
                self.assertEqual(f_fa.stride(0), MERGED)
                f_fg = kimi_k3_tiny_gemm(f_fa, self.f_b_w)

                for name, got, want in (
                    ("qkv", f_qkv, s_qkv),
                    ("g", f_g, s_g),
                    ("beta", f_beta, s_beta),
                    ("forget_gate", f_fg, s_fg),
                ):
                    scale = want.float().abs().max().clamp_min(1e-6)
                    rel = ((got.float() - want.float()).abs().max() / scale).item()
                    self.assertLess(rel, TOL, f"{name} rel err {rel:.2e}")

    @unittest.skipUnless(torch.version.hip is not None, "ROCm-only fused path")
    def test_deferred_f_b_returns_raw_f_a(self):
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=self.merged.device)
        fused_states = torch.nn.functional.linear(x, self.merged)
        expected_f_a = torch.split(fused_states, self.all_sizes, dim=-1)[2]
        fake_attention = SimpleNamespace(
            use_full_rank_gate=True,
            _bfa_w=object(),
            _bfa_fa_size=HEAD_DIM,
            _bfa_b_size=HEADS_TP,
            _qkvgbfa_sizes=self.all_sizes,
            _qkvgbfa_bs_limit=256,
            _qkvgbfa_layer=None,
            _bfa_f_b_w=self.f_b_w,
            _bfa_alt_stream=None,
            fused_qkvg_proj=SimpleNamespace(
                quant_method=SimpleNamespace(
                    apply=lambda _layer, _hidden_states, _bias: fused_states
                )
            ),
        )

        _, _, deferred_f_a, _ = KimiK3DeltaAttention.forward_qkvbfg_fused(
            fake_attention, x, defer_f_b=True
        )
        self.assertEqual(deferred_f_a.shape[-1], HEAD_DIM)
        self.assertTrue(torch.equal(deferred_f_a, expected_f_a))

        expected_gate = kimi_k3_tiny_gemm(expected_f_a, self.f_b_w)
        _, _, eager_gate, _ = KimiK3DeltaAttention.forward_qkvbfg_fused(
            fake_attention, x, defer_f_b=False
        )
        torch.testing.assert_close(eager_gate, expected_gate)


class _Weight(torch.Tensor):
    """A weight whose .data is not a plain tensor, the way a quantized one
    that carries its scales in a subclass is not."""


class TestKimiK3MayFuseKDAInProj(CustomTestCase):
    """What _may_fuse_kda_inproj accepts, one test a condition.

    The merge cats .weight alone. Anything that carries a separate scale must
    stay unfused, or the scale is dropped and the layer reads wrong numbers
    without failing. Those checks are on the three weights.

    Which configurations reach them is a separate question. The fused GEMM is a
    different kernel from the split pair, so taking the merge where it was not
    taken before moves decode output by a bf16 rounding step. do_fuse_qkvbfg
    keeps its say over that, and a monokernel backend overrides it, because
    that backend reads the merged buffer as its in-projection weight.

    The weights are on the meta device. The predicate reads their dtype, rank
    and width and never their values, so these need no GPU; this file runs on
    the AMD CI machines, where the predicate's own ROCm gate is satisfied.
    """

    @staticmethod
    def _attention(*, dtypes=None, widths=None, cls=torch.Tensor, **over):
        dtypes = dtypes or [torch.bfloat16] * 3
        widths = widths or [HIDDEN] * 3
        rows = (WIDE, HEAD_DIM, HEADS_TP)
        mods = [
            SimpleNamespace(
                weight=torch.zeros(r, w, dtype=d, device="meta").as_subclass(cls)
            )
            for r, w, d in zip(rows, widths, dtypes)
        ]
        defaults = {
            "use_full_rank_gate": True,
            "attn_tp_size": 8,
            "tp_size": 8,
            # False on the Kimi-K3 checkpoint, which carries a
            # quantization_config for its experts.
            "do_fuse_qkvbfg": False,
            "_bfa_uses_block_fp8": False,
            "fused_qkvg_proj": mods[0],
            "f_a_proj": mods[1],
            "b_proj": mods[2],
        }
        return SimpleNamespace(**(defaults | over))

    def _may_fuse(self, *, monokernel_backend=None, **kw) -> bool:
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(monokernel_backend=monokernel_backend)
        )
        with unittest.mock.patch.object(kimi_k3, "get_exec", lambda: exec_):
            return KimiK3DeltaAttention._may_fuse_kda_inproj(self._attention(**kw))

    def test_quantized_checkpoint_keeps_the_split_path_on_its_own(self):
        """Main's behaviour. do_fuse_qkvbfg is False on the shipped
        checkpoint, and nothing else asks for the merge."""
        self.assertFalse(self._may_fuse())

    def test_a_monokernel_backend_takes_the_merge(self):
        """The served case: the backend reads the merged buffer as w_in."""
        self.assertTrue(self._may_fuse(monokernel_backend="amd"))

    def test_do_fuse_qkvbfg_still_takes_the_merge(self):
        """An unquantized checkpoint fuses with no backend, as before."""
        self.assertTrue(self._may_fuse(do_fuse_qkvbfg=True))

    def test_quantized_attention_does_not_fuse(self):
        """An fp8 weight carries a scale the merged buffer would not."""
        self.assertFalse(
            self._may_fuse(monokernel_backend="amd", dtypes=[torch.float8_e4m3fn] * 3)
        )

    def test_mixed_dtypes_do_not_fuse(self):
        mixed = [torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn]
        self.assertFalse(self._may_fuse(monokernel_backend="amd", dtypes=mixed))

    def test_mixed_widths_do_not_fuse(self):
        """The three read the same hidden states, so a mismatch is a bug."""
        self.assertFalse(
            self._may_fuse(
                monokernel_backend="amd", widths=[HIDDEN, HIDDEN, HIDDEN // 2]
            )
        )

    def test_a_weight_that_is_not_a_plain_tensor_does_not_fuse(self):
        self.assertFalse(self._may_fuse(monokernel_backend="amd", cls=_Weight))

    def test_block_fp8_in_proj_does_not_fuse(self):
        """Block-FP8 needs dequantized buffers; the split path keeps them."""
        self.assertFalse(
            self._may_fuse(monokernel_backend="amd", _bfa_uses_block_fp8=True)
        )

    def test_low_rank_gate_does_not_fuse(self):
        """Without the full-rank gate there is no wide projection to extend."""
        self.assertFalse(
            self._may_fuse(monokernel_backend="amd", use_full_rank_gate=False)
        )

    def test_attention_tp_below_tp_still_fuses(self):
        """DP attention. The three weights sit on attention-TP or are
        replicated, and the merge reads their shapes, so TP does not enter."""
        self.assertTrue(self._may_fuse(monokernel_backend="amd", attn_tp_size=4))


if __name__ == "__main__":
    unittest.main()
