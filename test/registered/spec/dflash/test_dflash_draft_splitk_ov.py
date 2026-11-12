# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV (speculative/dflash_draft_splitk.py OV_TACTICS): the <= 8-row draft GEMMs with every
weight tile issued before the PDL wait. Pre-registered (before data):
- B: each OV projection (conv kernel projection with trigger mode 2, gate_up, fused gate_up + SiLU-mul, qkv_a) is bitwise
  equal to the served tactic's projection of the same weight at 1-8 rows on normal / large / cancelling inputs (same split,
  same reduction order), and the non-fused ones stay within the FP64 error bounds of test_dflash_draft_splitk.py. Positive
  control: a projection with another split differs from the served one in at least one case (the comparison can fail).
- T: trigger_follows_wait at 8 rows, from SplitKGemm's trigger semantics (mode 0 triggers once the weight warp has issued
  its stages, i.e. before the wait when stages >= K tiles per split; mode 2 waits, fences and triggers): served conv tactic
  True, OV conv mode 2 True, OV conv mode 0 False, OV qkv_a / gate_up mode 0 False; the draft AR site keeps its early
  trigger for the OV conv projection (mode 2) and drops it for a mode-0 one.
- I: with the knob, install routes the conv kernel projections to OV mode 2 and qkv_a / gate_up to OV mode 0 (qkv_a's
  trigger no longer follows its wait: the OV form is engaged), q_b / o_proj / down keep the served tactics, and every
  installed projection is bitwise the served projection of its own weight at 8 rows."""

import os
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HIDDEN, Q_LORA, BLOCK, TAPS, GROUP = 7168, 1536, 8, 2, 16
# Same bounds as test_dflash_draft_splitk.py.
REL_F, ELEM_REL, ELEM_SCALE = 4e-3, 2.0**-7, 2.0**-13
KNOB = "SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV"


def _inputs(rows, k, gen, kind):
    x = torch.randn(rows, k, device=DEV, generator=gen)
    if kind == "large":
        x = x * 40.0
    elif kind == "cancel":
        x = x.abs()
        x[:, 1::2] = -x[:, 0::2] * 1.001
    return x.to(torch.bfloat16)


def _weight(n, k, gen):
    return (torch.randn(n, k, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)


def _errors(y, x, w):
    ref = x.double() @ w.double().T
    scale = x.double().abs() @ w.double().abs().T
    err = (y.double() - ref).abs()
    return (err.norm() / ref.norm()).item(), (err / (ELEM_REL * ref.abs() + ELEM_SCALE * scale)).max().item()


def _run(proj, x):
    out = torch.full((x.shape[0], proj.n), float("nan"), device=DEV, dtype=torch.bfloat16)
    return proj.run(x, out)


class TestDFlashDraftSplitKOV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def test_B_ov_bitwise_to_served(self):
        from sglang.srt.speculative.dflash_draft_splitk import OV_TACTICS, SPLITK_TACTICS, DraftSplitKProjection

        gen = torch.Generator(device=DEV).manual_seed(20261105)
        # (label, shape, act_fuse, OV trigger mode, other-split control tactic)
        cases = (
            ("conv_proj", (1792, HIDDEN), False, 2, (128, 8, 4, 6)),
            ("gate_up", (1792, HIDDEN), False, 0, None),
            ("gate_up_act", (1792, HIDDEN), True, 0, None),
            ("qkv_a", (2112, HIDDEN), False, 0, (64, 8, 7, 8)),
        )
        failures, checked, control_differs = [], 0, {}
        for label, (n, k), act, trig, other in cases:
            w = _weight(n, k, gen)
            served = DraftSplitKProjection(w, SPLITK_TACTICS[(n, k)], act_fuse=act)
            ov = DraftSplitKProjection(w, SPLITK_TACTICS[(n, k)], act_fuse=act, ov=OV_TACTICS[(n, k)], ov_trigger=trig)
            ctl = DraftSplitKProjection(w, other) if other is not None else None
            for rows in range(1, 9):
                for kind in ("normal", "large", "cancel"):
                    x = _inputs(rows, k, gen, kind)
                    ys, yo = _run(served, x), _run(ov, x)
                    tag = f"{label} rows={rows} {kind}"
                    if not torch.isfinite(yo).all():
                        failures.append(f"{tag}: unwritten / non-finite OV output")
                    if not torch.equal(ys, yo):
                        d = (ys.float() - yo.float()).abs().max().item()
                        failures.append(f"{tag}: OV differs from the served tactic (maxabs {d:.3e})")
                    if not act:
                        rel_f, worst = _errors(yo, x, w)
                        if rel_f > REL_F or worst > 1.0:
                            failures.append(f"{tag}: OV relF {rel_f:.3e}, element {worst:.2f}x bound")
                    if ctl is not None:
                        control_differs[label] = control_differs.get(label, False) or not torch.equal(_run(ctl, x), ys)
                    checked += 1
            print(f"B {label}: OV {OV_TACTICS[(n, k)]} trigger {trig} vs served {SPLITK_TACTICS[(n, k)]}", flush=True)
        print(f"B cases checked: {checked}; control differs: {control_differs}", flush=True)
        self.assertEqual(checked, len(cases) * 8 * 3)
        for label, differs in control_differs.items():
            self.assertTrue(differs, f"CONTROL {label}: another split was bitwise equal everywhere (the check cannot fail)")
        self.assertFalse(failures, "\n".join(failures))

    def test_T_trigger_follows_wait_and_site(self):
        from sglang.srt.speculative.dflash_draft_ar_site import DraftArSites
        from sglang.srt.speculative.dflash_draft_splitk import OV_TACTICS, SPLITK_TACTICS, DraftSplitKProjection

        gen = torch.Generator(device=DEV).manual_seed(5)
        conv_w, qkv_w = _weight(1792, HIDDEN, gen), _weight(2112, HIDDEN, gen)
        conv_t, qkv_t = SPLITK_TACTICS[(1792, HIDDEN)], SPLITK_TACTICS[(2112, HIDDEN)]
        projs = {
            "served conv": (DraftSplitKProjection(conv_w, conv_t), True),
            "OV conv mode 2": (DraftSplitKProjection(conv_w, conv_t, ov=OV_TACTICS[(1792, HIDDEN)], ov_trigger=2), True),
            "OV conv mode 0": (DraftSplitKProjection(conv_w, conv_t, ov=OV_TACTICS[(1792, HIDDEN)], ov_trigger=0), False),
            "OV gate_up act mode 0": (
                DraftSplitKProjection(conv_w, conv_t, act_fuse=True, ov=OV_TACTICS[(1792, HIDDEN)], ov_trigger=0), False
            ),
            "OV qkv_a mode 0": (DraftSplitKProjection(qkv_w, qkv_t, ov=OV_TACTICS[(2112, HIDDEN)], ov_trigger=0), False),
        }
        failures = []
        for label, (proj, want) in projs.items():
            got = proj.trigger_follows_wait(8)
            print(f"T {label}: trigger_follows_wait(8) = {got}", flush=True)
            if got != want:
                failures.append(f"{label}: trigger_follows_wait(8) {got}, want {want}")
        sites = types.SimpleNamespace(early_trigger=True)
        normed = torch.empty(8, HIDDEN, device=DEV, dtype=torch.bfloat16)
        for label, want in (("OV conv mode 2", True), ("OV conv mode 0", False), ("served conv", True)):
            succ = types.SimpleNamespace(split_k_projection=projs[label][0])
            got = DraftArSites._early_for(sites, succ, normed)
            print(f"T site early trigger before {label}: {got}", flush=True)
            if got != want:
                failures.append(f"site early trigger before {label}: {got}, want {want}")
        self.assertFalse(failures, "\n".join(failures))

    def test_I_install_routes_ov(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.models.dflash import DFlashGroupedConv, DFlashMLAAttention
        from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS, DraftSplitKProjection, install_draft_splitk

        gen = torch.Generator(device=DEV).manual_seed(11)

        def linear(n, k):
            m = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16).to(DEV)
            with torch.no_grad():
                m.weight.copy_(_weight(n, k, gen))
            return m

        def conv():
            c = DFlashGroupedConv(HIDDEN, BLOCK, TAPS, GROUP).to(DEV, torch.bfloat16)
            with torch.no_grad():
                c.kernel_projection.weight.copy_(_weight(2 * TAPS * HIDDEN // GROUP, HIDDEN, gen))
            return c

        def attention():
            a = DFlashMLAAttention.__new__(DFlashMLAAttention)
            nn.Module.__init__(a)
            a.q_lora_rank, a.num_local_heads, a.qk_head_dim = Q_LORA, 6, 192
            a.fused_qkv_a_proj_with_mqa = linear(2112, HIDDEN)
            a.q_b_proj = linear(1152, Q_LORA)
            a.o_proj = linear(HIDDEN, 768)
            a.has_fused_proj, a.is_packed_weight, a.fused_a_gemm_backend = True, False, "auto"
            a._q_b_proj_verified_shape = True
            a._use_min_latency_fused_a_gemm = None
            a._use_min_latency_q_b_gemm = None
            return a

        layer = types.SimpleNamespace(
            self_attn=attention(),
            mlp=types.SimpleNamespace(gate_up_proj=linear(1792, HIDDEN), down_proj=linear(HIDDEN, 896), act_fn=None),
            attention_conv=conv(),
            mlp_conv=conv(),
        )
        old = os.environ.get(KNOB)
        os.environ[KNOB] = "1"
        try:
            install_draft_splitk(types.SimpleNamespace(layers=[layer]))
        finally:
            if old is None:
                os.environ.pop(KNOB, None)
            else:
                os.environ[KNOB] = old
        a, m = layer.self_attn, layer.mlp
        # (label, installed projection, weight, want trigger_follows_wait(8): from the OV tactic table + trigger mode)
        routed = (
            ("conv_attn", layer.attention_conv.split_k_projection, layer.attention_conv.kernel_projection.weight, True),
            ("conv_mlp", layer.mlp_conv.split_k_projection, layer.mlp_conv.kernel_projection.weight, True),
            ("qkv_a", a.fused_qkv_a_proj_with_mqa.quant_method.projection, a.fused_qkv_a_proj_with_mqa.weight, False),
            ("gate_up", m.gate_up_proj.quant_method.projection, m.gate_up_proj.weight, False),
            ("q_b", a.q_b_proj.quant_method.projection, a.q_b_proj.weight, False),  # served (64, 8, 4, 3): 3 tiles, 3 stages
            ("o_proj", a.o_proj.quant_method.projection, a.o_proj.weight, False),  # served (64, 8, 1, 6): 6 tiles, 6 stages
            ("down", m.down_proj.quant_method.projection, m.down_proj.weight, False),  # served (64, 8, 1, 7): 7 tiles, 7 stages
        )
        failures = []
        for label, proj, w, want in routed:
            w = w.detach()
            got = proj.trigger_follows_wait(8)
            if got != want:
                failures.append(f"{label}: trigger_follows_wait(8) {got}, want {want}")
            x = _inputs(8, w.shape[1], gen, "normal")
            ref = DraftSplitKProjection(w, SPLITK_TACTICS[tuple(w.shape)]).run(x)
            if not torch.equal(_run(proj, x), ref):
                failures.append(f"{label}: not bitwise the served projection of its own weight")
            print(f"I {label}: trigger_follows_wait(8) {got}", flush=True)
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
