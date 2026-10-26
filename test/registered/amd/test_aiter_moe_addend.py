"""Folding a shared-expert output into aiter's routed MoE (sglang.kernels.ops.moe.aiter_moe_addend)."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "gfx950 (MI35x) only"
)
class TestAiterMoeAddend(CustomTestCase):
    # MiniMax-M3.1 TP4 expert shape with fewer experts
    E, TOPK, DIM, INTER = 16, 4, 6144, 768

    @classmethod
    def setUpClass(cls):
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.ops.shuffle import shuffle_weight
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle

        from sglang.kernels.ops.moe.aiter_moe_addend import (
            apply_aiter_moe_addend_patch,
        )
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
            AiterRunnerCore,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        # MiniMaxM3MoEGfx950 applies it; AiterRunnerCore applies only the sort patch
        apply_aiter_moe_addend_patch()
        torch.manual_seed(0)
        dev = torch.device("cuda", 0)
        E, DIM, INTER = cls.E, cls.DIM, cls.INTER

        def mxfp4(rows, cols):
            q, s = dynamic_mxfp4_quant(
                (torch.randn(E * rows, cols, device=dev) * 0.02).to(torch.bfloat16)
            )
            s = e8m0_shuffle(s.view(torch.uint8)).view(torch.float8_e8m0fnu)
            q = shuffle_weight(q.view(torch.uint8).view(E, rows, cols // 2), (16, 16))
            q = q.view(torch.float4_e2m1fn_x2)
            q.is_shuffled = True
            return q, s.view(E, rows, -1)

        w13, s13 = mxfp4(2 * INTER, DIM)
        w2, s2 = mxfp4(DIM, INTER)
        cls.quant_info = AiterMoeQuantInfo(
            fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
            w13_weight=w13,
            w2_weight=w2,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=s13,
            w2_scale=s2,
            swiglu_limit=7.0,
        )
        cls.runner = AiterRunnerCore(
            MoeRunnerConfig(
                num_experts=E,
                num_local_experts=E,
                hidden_size=DIM,
                intermediate_size_per_partition=INTER,
                top_k=cls.TOPK,
                activation="silu",
                is_gated=True,
                gemm1_alpha=1.702,
                gemm1_beta=1.0,
                gemm1_clamp_limit=7.0,
                gate_up_interleaved=False,
            )
        )
        cls.dev = dev

    def _inputs(self, m):
        g = torch.Generator(device=self.dev).manual_seed(m)
        x = torch.randn(m, self.DIM, generator=g, device=self.dev).to(torch.bfloat16)
        ids = torch.rand(m, self.E, generator=g, device=self.dev).argsort(dim=1)
        ids = ids[:, : self.TOPK].to(torch.int32).contiguous()
        w = torch.rand(m, self.TOPK, generator=g, device=self.dev)
        addend = torch.randn(m, self.DIM, generator=g, device=self.dev) * 0.1
        return x, ids, w, addend.to(torch.bfloat16)

    def _routed(self, x, ids, w):
        from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerInput

        runner_input = AiterRunnerInput(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=w,
            quant_type=self.quant_info.quant_type,
        )
        return self.runner.run(runner_input, self.quant_info, {}).hidden_states

    def _fold(self, x, ids, w, addend):
        from sglang.kernels.ops.moe.aiter_moe_addend import moe_output_addend

        with moe_output_addend(addend) as slot:
            out = self._routed(x, ids, w)
        return out, slot.consumed

    def _assert_close(self, out, ref, msg):
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 1e-2, msg)

    def test_folded_output_is_routed_plus_addend(self):
        """A consumed addend comes out added exactly once: the sort must not zero it
        and a stage2 that reduces after the sort handed it out must not overwrite it."""
        cases = {
            "bf16 activations, accumulating stage2": (64, {}),
            "fp4 activations, accumulating stage2": (320, {}),
            "accumulate sort, reducing stage2": (
                320,
                {"AITER_FLYDSL_FORCE_REDUCE": "1"},
            ),
        }
        for name, (m, env) in cases.items():
            with self.subTest(name), patch.dict(os.environ, env):
                x, ids, w, addend = self._inputs(m)
                ref = self._routed(x, ids, w).float() + addend.float()
                out, consumed = self._fold(x, ids, w, addend.clone())
                self.assertTrue(consumed, name)
                self._assert_close(out, ref, name)

    def test_reduce_epilogue_adds_a_pending_addend(self):
        import aiter.ops.flydsl.moe_kernels as moe_kernels

        from sglang.kernels.ops.moe.aiter_moe_addend import moe_output_addend

        m = 300
        target = torch.randn(m, self.TOPK, self.DIM, device=self.dev).to(torch.bfloat16)
        _, _, _, addend = self._inputs(m)
        out = torch.empty_like(addend)
        with moe_output_addend(addend) as slot:
            moe_kernels._run_moe_reduction(target, out, m, self.TOPK, self.DIM)
        self.assertTrue(slot.consumed)
        self._assert_close(out, target.float().sum(1) + addend.float(), "reduce")

    def test_reduction_patch_names_what_aiter_passes(self):
        """The wrapper names aiter's positional arguments: a reordered _run_moe_reduction
        stays unpatched instead of misbinding them, and arguments past the named ones
        reach aiter unchanged."""
        from sglang.kernels.ops.moe.aiter_moe_addend import wrap_moe_reduction

        def reordered(out, target, token_num, topk, model_dim):
            pass

        self.assertIsNone(wrap_moe_reduction(reordered))
        calls = []

        def reduction(
            target,
            out,
            token_num,
            topk,
            model_dim,
            expert_mask=None,
            topk_ids=None,
            stream=None,
            is_fp8=False,
            topk_weights=None,
            fp8_scale_blk=None,
        ):
            calls.append((out, is_fp8, fp8_scale_blk))

        target = torch.zeros(2, self.TOPK, 8, device=self.dev)
        out = torch.empty(2, 8, device=self.dev)
        wrap_moe_reduction(reduction)(
            target, out, 2, self.TOPK, 8, None, None, None, True, None, 128
        )
        self.assertEqual(calls, [(out, True, 128)])


if __name__ == "__main__":
    unittest.main()
