# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE_MLP_ONLY (speculative/dflash_draft_conv_prologue.py install_draft_conv_prologue):
The conv prologue routes only the MLP conv into gate_up (fused gate_up + SiLU-mul, or the plain gate_up projection); the
attention conv and fused qkv_a stay served. Pre-registered (before data), on a 2-layer stand-in draft model with the served
draft weight shapes (layer 0: fused gate_up + SiLU-mul, layer 1: plain gate_up):
- W (wiring): with the switch on, install returns 2 (one GEMM per layer); every attention conv keeps conv_prologue None and
  every qkv_a keeps its original quant_method object; every MLP conv gets a conv-prologue form and its gate_up path is wrapped.
- N (numerics through the installed objects): for each layer, raw rows left as the MLP conv's pending record and passed to the
  wrapped gate_up path equal (served dflash_grouped_conv side 0 -> the original served split-K projection) BITWISE at 1 / 3 / 8
  rows on normal / large / cancelling inputs, and the record is consumed. Independent reference: the served Triton conv kernel
  and the served projection objects the stand-in was built with, not the code under test.
- C (control): with the switch off, install routes 4 GEMMs and sets the attention conv's conv-prologue form (the W checks can fail).
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HIDDEN, BLOCK, TAPS, GROUP = 7168, 8, 2, 16
GROUPS = HIDDEN // GROUP
QKV_A, GATE_UP = 2112, 1792
KNOBS = ("SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE_MLP_ONLY", "SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV")


def _inputs(rows, gen, kind):
    x = torch.randn(rows, HIDDEN, device=DEV, generator=gen)
    if kind == "large":
        x = x * 40.0
    elif kind == "cancel":
        x = x.abs()
        x[:, 1::2] = -x[:, 0::2] * 1.001
    return x.to(torch.bfloat16)


def _projection(rows, gen):
    return (torch.randn(rows, 2 * TAPS * GROUPS, device=DEV, generator=gen) * 0.1).to(torch.bfloat16)


def _base(gen):
    base = torch.randn(2, TAPS, HIDDEN, device=DEV, generator=gen) * 0.05
    base[:, 0] += 1.0
    return torch.nn.Parameter(base.to(torch.bfloat16).contiguous(), requires_grad=False)


def _served_conv(x, projection, base0):
    from sglang.kernels.ops.speculative.dflash import dflash_grouped_conv

    delta = projection.reshape(x.shape[0], 2, TAPS, GROUPS)[:, 0]
    return dflash_grouped_conv(x, delta, base0, BLOCK, GROUP, TAPS)


def _model(gen):
    """Two stand-in draft layers carrying exactly the attributes install_draft_conv_prologue reads."""
    from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS, DraftSplitKLinearMethod, DraftSplitKProjection

    def weight(n):
        return (torch.randn(n, HIDDEN, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)

    def conv():
        return SimpleNamespace(block_size=BLOCK, base_kernel=_base(gen), conv_prologue=None, conv_pending=None)

    layers = []
    for fused in (True, False):
        w_qkv, w_gu = weight(QKV_A), weight(GATE_UP)
        qkv_a = SimpleNamespace(weight=w_qkv)
        qkv_a.quant_method = DraftSplitKLinearMethod(DraftSplitKProjection(w_qkv, SPLITK_TACTICS[(QKV_A, HIDDEN)]))
        gate_up = SimpleNamespace(weight=w_gu, quant_method=None)
        tactic = SPLITK_TACTICS[(GATE_UP, HIDDEN)]
        if fused:
            act = DraftSplitKProjection(w_gu, tactic, act_fuse=True)
        else:
            act = None
            gate_up.quant_method = DraftSplitKLinearMethod(DraftSplitKProjection(w_gu, tactic))
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(fused_qkv_a_proj_with_mqa=qkv_a),
                mlp=SimpleNamespace(split_k_gate_up_act=act, gate_up_proj=gate_up),
                attention_conv=conv(),
                mlp_conv=conv(),
            )
        )
    return SimpleNamespace(layers=layers)


class TestDFlashDraftConvPrologueMlpOnly(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in KNOBS}
        os.environ.pop("SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _install(self, mlp_only, gen):
        from sglang.srt.speculative.dflash_draft_conv_prologue import install_draft_conv_prologue

        os.environ["SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE_MLP_ONLY"] = "1" if mlp_only else "0"
        model = _model(gen)
        served = [
            (l.self_attn.fused_qkv_a_proj_with_mqa.quant_method, l.mlp.split_k_gate_up_act, l.mlp.gate_up_proj.quant_method)
            for l in model.layers
        ]
        return model, served, install_draft_conv_prologue(model)

    def test_C_switch_off_routes_qkv_a_too(self):
        gen = torch.Generator(device=DEV).manual_seed(20261105)
        model, served, installed = self._install(False, gen)
        print(f"C installed {installed}")
        self.assertEqual(installed, 4)
        for layer, (qkv_m, _, _) in zip(model.layers, served):
            self.assertIsNotNone(layer.attention_conv.conv_prologue)
            self.assertIsNot(layer.self_attn.fused_qkv_a_proj_with_mqa.quant_method, qkv_m)

    def test_W_N_mlp_only(self):
        from sglang.srt.speculative.dflash_draft_conv_prologue import DraftConvPrologueFused, DraftConvPrologueLinearMethod

        gen = torch.Generator(device=DEV).manual_seed(20261106)
        model, served, installed = self._install(True, gen)
        print(f"W installed {installed}")
        # W
        self.assertEqual(installed, 2)
        for layer, (qkv_m, act, gu_m) in zip(model.layers, served):
            self.assertIsNone(layer.attention_conv.conv_prologue)
            self.assertIs(layer.self_attn.fused_qkv_a_proj_with_mqa.quant_method, qkv_m)
            self.assertIsNotNone(layer.mlp_conv.conv_prologue)
            if act is not None:
                self.assertIsInstance(layer.mlp.split_k_gate_up_act, DraftConvPrologueFused)
                self.assertIs(layer.mlp.gate_up_proj.quant_method, gu_m)
            else:
                self.assertIsNone(layer.mlp.split_k_gate_up_act)
                self.assertIsInstance(layer.mlp.gate_up_proj.quant_method, DraftConvPrologueLinearMethod)
        # N
        failures, controls_equal, n = [], [], 0
        for li, (layer, (qkv_m, act, gu_m)) in enumerate(zip(model.layers, served)):
            conv = layer.mlp_conv
            base0 = conv.base_kernel.detach()[0]
            for rows in (1, 3, 8):
                for kind in ("normal", "large", "cancel"):
                    x = _inputs(rows, gen, kind)
                    projection = _projection(rows, gen)
                    convolved = _served_conv(x, projection, base0)
                    if act is not None:
                        ref = act.run(convolved)
                        raw = act.run(x)
                        conv.conv_pending = (x, projection)
                        out = layer.mlp.split_k_gate_up_act.run(x)
                    else:
                        gate_up = layer.mlp.gate_up_proj
                        ref = gu_m.apply(gate_up, convolved)
                        raw = gu_m.apply(gate_up, x)
                        conv.conv_pending = (x, projection)
                        out = gate_up.quant_method.apply(gate_up, x)
                    torch.cuda.synchronize()
                    n += 1
                    self.assertIsNone(conv.conv_pending, "conv prologue record not consumed")
                    bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
                    print(f"N layer {li} {'fused' if act is not None else 'plain'} rows {rows} {kind}: {bad} of {out.numel()} differ")
                    if bad:
                        failures.append((li, rows, kind, bad))
                    if torch.equal(raw.view(torch.int16), out.view(torch.int16)):
                        controls_equal.append((li, rows, kind))
        print(f"N cases {n}, failures {len(failures)}; controls equal to the conv prologue {len(controls_equal)}")
        self.assertEqual(controls_equal, [], "unconvolved rows must differ (transform engaged)")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
