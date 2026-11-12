# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash draft split-K GEMMs (speculative/dflash_draft_splitk.py): each GEMM stays within a stated error of an
FP64 reference and no worse than the stock GEMM, and the install routes every covered draft linear to the
split-K projection of its own weight, leaving other row counts on the stock path."""

import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HIDDEN, Q_LORA, BLOCK, TAPS, GROUP = 7168, 1536, 8, 2, 16
LAYERS = 2
# Pre-registered bounds: Frobenius error vs FP64, ratio to stock, per-element bound.
REL_F, STOCK_RATIO, STOCK_FLOOR = 4e-3, 1.25, 2e-4
ELEM_REL, ELEM_SCALE = 2.0**-7, 2.0**-13


def _inputs(rows, k, gen, kind):
    x = torch.randn(rows, k, device=DEV, generator=gen)
    if kind == "large":
        x = x * 40.0
    elif kind == "cancel":
        # Opposite-sign pairs: partial sums cancel, so a dropped or doubled slice shows.
        x = x.abs()
        x[:, 1::2] = -x[:, 0::2] * 1.001
    return x.to(torch.bfloat16)


def _weight(n, k, gen):
    return (torch.randn(n, k, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)


def _errors(y, x, w):
    """(relative Frobenius error vs FP64, worst element error over its bound)."""
    ref = x.double() @ w.double().T
    scale = x.double().abs() @ w.double().abs().T
    err = (y.double() - ref).abs()
    rel_f = (err.norm() / ref.norm()).item()
    worst = (err / (ELEM_REL * ref.abs() + ELEM_SCALE * scale)).max().item()
    return rel_f, worst


class TestDFlashDraftSplitK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def test_projection_error_vs_fp64(self):
        from sglang.srt.speculative.dflash_draft_splitk import (
            SPLITK_TACTICS,
            DraftSplitKProjection,
        )

        gen = torch.Generator(device=DEV).manual_seed(20261030)
        failures, checked = [], 0
        for (n, k), tactic in SPLITK_TACTICS.items():
            w = _weight(n, k, gen)
            proj = DraftSplitKProjection(w, tactic)
            for rows in range(1, 9):
                for kind in ("normal", "large", "cancel"):
                    x = _inputs(rows, k, gen, kind)
                    out = torch.full((rows, n), float("nan"), device=DEV, dtype=torch.bfloat16)
                    y = proj.run(x, out)
                    stock = F.linear(x, w)
                    rel_f, worst = _errors(y, x, w)
                    stock_f, stock_worst = _errors(stock, x, w)
                    tag = f"{(n, k)} rows={rows} {kind}"
                    if not torch.isfinite(y).all():
                        failures.append(f"{tag}: unwritten/non-finite output")
                    if rel_f > REL_F:
                        failures.append(f"{tag}: relF {rel_f:.3e} > {REL_F}")
                    if rel_f > STOCK_RATIO * stock_f + STOCK_FLOOR:
                        failures.append(f"{tag}: relF {rel_f:.3e} vs stock {stock_f:.3e}")
                    if worst > 1.0:
                        failures.append(f"{tag}: element error {worst:.2f}x bound")
                    if stock_worst > 1.0:
                        failures.append(f"{tag}: CONTROL stock element error {stock_worst:.2f}x bound")
                    checked += 1
        print(f"projection cases checked: {checked}")
        self.assertEqual(checked, len(SPLITK_TACTICS) * 8 * 3)
        self.assertFalse(failures, "\n".join(failures))

    def test_install_routes_each_linear_to_its_own_weight(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.dflash import DFlashGroupedConv, DFlashMLAAttention
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
        from sglang.srt.speculative.dflash_draft_splitk import (
            SPLITK_TACTICS,
            DraftSplitKProjection,
            install_draft_splitk,
        )

        gen = torch.Generator(device=DEV).manual_seed(7)

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
            # Without the install's routing, qkv_a would take fused_a_gemm (the stock min-latency path).
            a.has_fused_proj, a.is_packed_weight, a.fused_a_gemm_backend = True, False, "auto"
            a._q_b_proj_verified_shape = True
            a._use_min_latency_fused_a_gemm = None
            a._use_min_latency_q_b_gemm = None
            return a

        layers = [
            types.SimpleNamespace(
                self_attn=attention(),
                mlp=types.SimpleNamespace(gate_up_proj=linear(1792, HIDDEN), down_proj=linear(HIDDEN, 896)),
                attention_conv=conv(),
                mlp_conv=conv(),
            )
            for _ in range(LAYERS)
        ]
        model = types.SimpleNamespace(layers=layers)

        def named(layer):
            a, m = layer.self_attn, layer.mlp
            return {
                "qkv_a": a.fused_qkv_a_proj_with_mqa,
                "q_b": a.q_b_proj,
                "o_proj": a.o_proj,
                "gate_up": m.gate_up_proj,
                "down": m.down_proj,
            }

        # Independent references from each layer's own weight, taken before install.
        refs = []
        for layer in layers:
            r = {name: lin.weight.detach().clone() for name, lin in named(layer).items()}
            r["conv_attn"] = layer.attention_conv.kernel_projection.weight.detach().clone()
            r["conv_mlp"] = layer.mlp_conv.kernel_projection.weight.detach().clone()
            refs.append(r)
        covered = lambda w: tuple(w.shape) in SPLITK_TACTICS
        expected_count = sum(covered(w) for r in refs for w in r.values())
        self.assertEqual(install_draft_splitk(model), expected_count)

        failures = []

        def check(tag, y, x, w):
            # Shapes without a tactic stay on the stock GEMM (F.linear here: no bf16 backend is initialised).
            if covered(w):
                expected = DraftSplitKProjection(w, SPLITK_TACTICS[tuple(w.shape)]).run(x)
            else:
                expected = F.linear(x, w)
            if not torch.equal(y, expected):
                failures.append(f"{tag}: not the {'split-K projection' if covered(w) else 'stock GEMM'} of its own weight")
            rel_f, worst = _errors(y, x, w)
            if rel_f > REL_F or worst > 1.0:
                failures.append(f"{tag}: relF {rel_f:.3e}, element {worst:.2f}x bound")

        for i, (layer, r) in enumerate(zip(layers, refs)):
            for name, lin in named(layer).items():
                x = _inputs(BLOCK, lin.weight.shape[1], gen, "normal")
                check(f"layer {i} {name} forward", lin(x)[0], x, r[name])
                if not covered(r[name]):
                    continue
                out = torch.full((BLOCK, lin.weight.shape[0]), float("nan"), device=DEV, dtype=torch.bfloat16)
                y = lin.quant_method.apply_into(lin, x, out)
                if y.data_ptr() != out.data_ptr():
                    failures.append(f"layer {i} {name} apply_into: did not write the caller's output")
                check(f"layer {i} {name} apply_into", out, x, r[name])
                # Uncovered row counts keep the stock method bitwise.
                for rows in (9, 16):
                    x_big = _inputs(rows, lin.weight.shape[1], gen, "normal")
                    stock = UnquantizedLinearMethod().apply(lin, x_big)
                    if not torch.equal(lin(x_big)[0], stock):
                        failures.append(f"layer {i} {name} rows={rows}: not the stock path")
            a = layer.self_attn
            h = _inputs(BLOCK, HIDDEN, gen, "normal")
            check(
                f"layer {i} prepare_qkv_latent",
                DeepseekV2AttentionMLA.prepare_qkv_latent(a, h, None),
                h,
                r["qkv_a"],
            )
            q_lora = _inputs(BLOCK, Q_LORA, gen, "normal")
            q = DeepseekV2AttentionMLA.q_b_proj_forward(a, q_lora)
            check(f"layer {i} q_b_proj_forward", q.reshape(BLOCK, -1), q_lora, r["q_b"])
            for side, c in (("conv_attn", layer.attention_conv), ("conv_mlp", layer.mlp_conv)):
                h = _inputs(BLOCK, HIDDEN, gen, "normal")
                _, coefficients = c.prepare(h)
                full = c._project(h)
                check(f"layer {i} {side}", full, h, r[side])
                if not torch.equal(coefficients, full.reshape(BLOCK, 2, TAPS, -1)[:, 1]):
                    failures.append(f"layer {i} {side}: prepare does not use the projection")
        self.assertFalse(failures, "\n".join(failures))


def _bitwise_or_both_nan(got, ref):
    """(number of non-NaN reference elements whose BF16 bits differ, NaN masks equal)."""
    nan_ref = torch.isnan(ref)
    diff = (got.view(torch.int16) != ref.view(torch.int16)) & ~nan_ref
    return int(diff.sum()), bool(torch.equal(nan_ref, torch.isnan(got)))


class TestDFlashDraftSplitKAct(unittest.TestCase):
    """SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT: the fused gate_up + SiLU-mul projection is bitwise equal to the served pair it
    replaces (the split-K gate_up projection, then sglang's SiluAndMul act_and_mul kernel); the reference is those served
    kernels, not the fused code. Catches a swapped gate / up, a wrong up row, a different SiLU / division form, a missing
    BF16 rounding of the gate_up tile, and stores past the last row."""

    I, K = 896, HIDDEN

    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def _pair(self, w):
        from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS, DraftSplitKProjection

        tactic = SPLITK_TACTICS[tuple(w.shape)]
        return DraftSplitKProjection(w, tactic), DraftSplitKProjection(w, tactic, act_fuse=True)

    def test_every_bf16_gate_value_bitwise(self):
        from sglang.srt.layers.activation import SiluAndMul

        act, I = SiluAndMul(), self.I
        w = torch.zeros(2 * I, self.K, dtype=torch.bfloat16, device=DEV)
        unfused, fused = self._pair(w)
        # One-hot activation rows: out[r, j] = act(w[j, r], w[I + j, r]) exactly (the other K terms are zero).
        x = torch.zeros(BLOCK, self.K, dtype=torch.bfloat16, device=DEV)
        x[torch.arange(BLOCK), torch.arange(BLOCK)] = 1.0
        patterns = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16)
        ups = [1.0, -1.0, 0.0, -0.0, float("inf"), float("nan"), 3.3895e38, 1.1755e-38, -2.75]
        per, failures, checked = I * BLOCK, [], 0
        gen = torch.Generator(device=DEV).manual_seed(5)
        for u in ups + ["random"]:
            for start in range(0, 65536, per):
                gate = torch.zeros(per, dtype=torch.bfloat16, device=DEV)
                chunk = patterns[start : start + per]
                gate[: chunk.numel()] = chunk
                w.zero_()
                w[:I, :BLOCK] = gate.view(BLOCK, I).t()
                if u == "random":
                    w[I:, :BLOCK] = (torch.randn(I, BLOCK, generator=gen, device=DEV) * 3).to(torch.bfloat16)
                else:
                    w[I:, :BLOCK] = u
                ref = act(unfused.run(x))
                got = fused.run(x)
                n_diff, nan_ok = _bitwise_or_both_nan(got, ref)
                if n_diff or not nan_ok:
                    failures.append(f"up={u} gates from {start}: {n_diff} differ, NaN masks equal {nan_ok}")
                checked += 1
        self.assertEqual(checked, (len(ups) + 1) * -(-65536 // per))
        self.assertFalse(failures, "\n".join(failures))

    def test_random_rows_bitwise_and_tail_untouched(self):
        from sglang.srt.layers.activation import SiluAndMul

        act, I, failures = SiluAndMul(), self.I, []
        gen = torch.Generator(device=DEV).manual_seed(20261031)
        for seed_round in range(3):
            w = _weight(2 * I, self.K, gen)
            unfused, fused = self._pair(w)
            for rows in range(1, BLOCK + 1):
                for kind in ("normal", "large", "cancel"):
                    x = _inputs(rows, self.K, gen, kind)
                    ref = act(unfused.run(x))
                    canvas = torch.full((BLOCK, I), -7.0, dtype=torch.bfloat16, device=DEV)
                    fused.run(x, canvas[:rows])
                    n_diff, nan_ok = _bitwise_or_both_nan(canvas[:rows], ref)
                    tag = f"round {seed_round} rows={rows} {kind}"
                    if n_diff or not nan_ok:
                        failures.append(f"{tag}: {n_diff} elements differ from the served pair")
                    if not bool((canvas[rows:] == -7.0).all()):
                        failures.append(f"{tag}: wrote past the last row")
        self.assertFalse(failures, "\n".join(failures))

    def test_mlp_forward_takes_the_fused_projection_for_covered_rows(self):
        """DFlashMLP.forward reads mlp.split_k_gate_up_act for <= 8 rows and the unfused path otherwise. The fused projection
        here is built from a DIFFERENT weight than gate_up_proj, so each path is identified by its output."""
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.models.dflash import DFlashMLP
        from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS, DraftSplitKProjection

        I, gen = self.I, torch.Generator(device=DEV).manual_seed(11)

        def linear(n, k):
            m = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16).to(DEV)
            with torch.no_grad():
                m.weight.copy_(_weight(n, k, gen))
            return m

        mlp = types.SimpleNamespace(
            gate_up_proj=linear(2 * I, self.K), down_proj=linear(self.K, I), act_fn=SiluAndMul(), _draft_all_reduce=None
        )
        other = _weight(2 * I, self.K, gen)
        mlp.split_k_gate_up_act = DraftSplitKProjection(other, SPLITK_TACTICS[tuple(other.shape)], act_fuse=True)
        failures = []
        for rows in (1, BLOCK, 9, 16):
            x = _inputs(rows, self.K, gen, "normal")
            got = DFlashMLP.forward(mlp, x)
            if rows <= BLOCK:
                want = mlp.down_proj(mlp.split_k_gate_up_act.run(x))[0]
            else:
                want = mlp.down_proj(mlp.act_fn(mlp.gate_up_proj(x)[0]))[0]
            if not torch.equal(got, want):
                failures.append(f"rows={rows}: not the {'fused' if rows <= BLOCK else 'unfused'} path")
        self.assertFalse(failures, "\n".join(failures))

    def test_install_fuses_only_the_default_silu(self):
        """With the knob, install_draft_splitk gives a default-dispatch SiluAndMul MLP the fused projection (its forward then
        equals the served pair bitwise) and leaves an MLP pinned to forward_native (exact torch SiLU, not reproduced) alone."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.models.dflash import DFlashMLP
        from sglang.srt.speculative import dflash_draft_splitk as dsk

        I, gen = self.I, torch.Generator(device=DEV).manual_seed(13)

        def linear(n, k):
            m = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16).to(DEV)
            with torch.no_grad():
                m.weight.copy_(_weight(n, k, gen))
            return m

        def mlp(pin_native):
            act = SiluAndMul()
            if pin_native:
                act._forward_method = act.forward_native
            return types.SimpleNamespace(
                gate_up_proj=linear(2 * I, self.K), down_proj=linear(self.K, I), act_fn=act, _draft_all_reduce=None,
                split_k_gate_up_act=None,
            )

        default, pinned = mlp(False), mlp(True)
        self.assertEqual(dsk._install_mlp_act(default), 0, "installed with the knob off")
        with envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT.override(True):
            self.assertEqual(dsk._install_mlp_act(default), 1)
            self.assertEqual(dsk._install_mlp_act(pinned), 0)
        self.assertIsNotNone(default.split_k_gate_up_act, "default-dispatch SiluAndMul MLP was not fused")
        self.assertIsNone(pinned.split_k_gate_up_act, "an MLP pinned to forward_native was fused")
        x = _inputs(BLOCK, self.K, gen, "normal")
        want = default.down_proj(default.act_fn(dsk.DraftSplitKProjection(
            default.gate_up_proj.weight, dsk.SPLITK_TACTICS[tuple(default.gate_up_proj.weight.shape)]).run(x)))[0]
        self.assertTrue(torch.equal(DFlashMLP.forward(default, x), want))


if __name__ == "__main__":
    unittest.main()
