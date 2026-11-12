# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Conv prologue (speculative/dflash_draft_conv_prologue.py, SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE): the draft grouped conv
(side 0) folded into the next split-K GEMM. Pre-registered (before data):
- B: for fused qkv_a (2112 x 7168), gate_up (1792 x 7168) and the fused gate_up + SiLU-mul, at the served <= 8-row tactic
  and its OV form, the conv-prologue GEMM of the RAW rows equals (served dflash_grouped_conv side 0 -> served split-K projection of
  the same tactic) BITWISE at 1 / 3 / 8 rows on normal / large / cancelling inputs. Independent reference: the served
  Triton conv kernel and the served SplitKGemm, not the code under test.
- C: positive control: the served projection of the UNCONVOLVED rows differs from the conv-prologue output in every case (the
  comparison can fail and the transform is engaged).
- Each mutant of the transform (tap shift, group index, swizzle, tap-1 delta) must make B fail.
"""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HIDDEN, BLOCK, TAPS, GROUP = 7168, 8, 2, 16
GROUPS = HIDDEN // GROUP


def _inputs(rows, gen, kind):
    x = torch.randn(rows, HIDDEN, device=DEV, generator=gen)
    if kind == "large":
        x = x * 40.0
    elif kind == "cancel":
        x = x.abs()
        x[:, 1::2] = -x[:, 0::2] * 1.001
    return x.to(torch.bfloat16)


def _conv_operands(rows, gen):
    projection = (torch.randn(rows, 2 * TAPS * GROUPS, device=DEV, generator=gen) * 0.1).to(torch.bfloat16)
    base = torch.randn(TAPS, HIDDEN, device=DEV, generator=gen) * 0.05
    base[0] += 1.0
    return projection, base.to(torch.bfloat16).contiguous()


def _served_conv(x, projection, base):
    from sglang.kernels.ops.speculative.dflash import dflash_grouped_conv

    delta = projection.reshape(x.shape[0], 2, TAPS, GROUPS)[:, 0]
    return dflash_grouped_conv(x, delta, base, BLOCK, GROUP, TAPS)


class TestDFlashDraftConvPrologue(unittest.TestCase):
    def test_B_bitwise_to_served_conv_then_projection(self):
        from sglang.srt.speculative.dflash_draft_conv_prologue import DraftConvPrologueProjection
        from sglang.srt.speculative.dflash_draft_splitk import OV_TACTICS, SPLITK_TACTICS, DraftSplitKProjection

        gen = torch.Generator(device=DEV).manual_seed(20261105)
        cases = (("qkv_a", 2112, False), ("gate_up", 1792, False), ("gate_up_act", 1792, True))
        failures, controls_equal, n = [], [], 0
        # Test hook for mutant runs: SGLANG_TEST_D2_CASES=label:form,... restricts the cases.
        only = {c for c in os.environ.get("SGLANG_TEST_D2_CASES", "").split(",") if c}
        for label, n_out, act in cases:
            weight = (torch.randn(n_out, HIDDEN, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)
            served_tactic = SPLITK_TACTICS[(n_out, HIDDEN)]
            for form, tactic, ov in (("served", served_tactic, None), ("ov", OV_TACTICS[(n_out, HIDDEN)], OV_TACTICS[(n_out, HIDDEN)])):
                if only and f"{label}:{form}" not in only:
                    continue
                ref_proj = DraftSplitKProjection(weight, served_tactic, act_fuse=act, ov=ov)
                d2 = DraftConvPrologueProjection(weight, tactic, BLOCK, act_fuse=act, oversized=ov is not None)
                for rows in (1, 3, 8):
                    for kind in ("normal", "large", "cancel"):
                        x = _inputs(rows, gen, kind)
                        projection, base = _conv_operands(rows, gen)
                        ref = ref_proj.run(_served_conv(x, projection, base))
                        raw = ref_proj.run(x)
                        out = torch.full_like(ref, float("nan"))
                        d2.run(x, projection, base, out)
                        torch.cuda.synchronize()
                        n += 1
                        same = torch.equal(out.view(torch.int16), ref.view(torch.int16))
                        bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
                        print(f"B {label} {form} {tactic} rows {rows} {kind}: bitwise {same} ({bad} of {out.numel()} differ)")
                        if not same:
                            failures.append((label, form, rows, kind, bad))
                        if torch.equal(raw.view(torch.int16), out.view(torch.int16)):
                            controls_equal.append((label, form, rows, kind))
        print(f"B cases {n}, failures {len(failures)}; C controls equal to the conv prologue {len(controls_equal)}")
        self.assertEqual(controls_equal, [], "positive control: unconvolved rows must differ")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
