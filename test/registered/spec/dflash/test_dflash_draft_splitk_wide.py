# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash draft split-K GEMMs at 9-64 rows (SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE, speculative/dflash_draft_splitk.py
WIDE_TACTICS / WIDE_ACT_TACTICS): each wide entry stays within the module's stated error of an FP64 reference and no
worse than the stock GEMM, stores no row past the input's last, the fused SiLU-mul stays bitwise equal to its unfused
split-K pair, and the knob routes exactly the listed row counts (every other row count keeps the stock path)."""

import types
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HIDDEN, INTER = 7168, 896
# The module's bounds (test_dflash_draft_splitk.py, pre-registered), unchanged.
REL_F, STOCK_RATIO, STOCK_FLOOR = 4e-3, 1.25, 2e-4
ELEM_REL, ELEM_SCALE = 2.0**-7, 2.0**-13
KINDS = ("normal", "large", "cancel")
CANVAS = -7.0


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


def _direct(weight, rows, tactic, act_fuse=False):
    """The plugin projection built straight from one (rows, tactic) policy: the reference for the module's routing."""
    from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import PreparedProjection

    return PreparedProjection(weight, {rows: tactic}, proxy_fences=True, act_fuse=act_fuse).prepare()


def _bitwise_or_both_nan(got, ref):
    nan_ref = torch.isnan(ref)
    diff = (got.view(torch.int16) != ref.view(torch.int16)) & ~nan_ref
    return int(diff.sum()), bool(torch.equal(nan_ref, torch.isnan(got)))


class TestDFlashDraftSplitKWide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def test_tables_are_wide_and_nonempty(self):
        from sglang.srt.speculative.dflash_draft_splitk import (
            MAX_SPLITK_ROWS,
            SPLITK_TACTICS,
            WIDE_ACT_TACTICS,
            WIDE_TACTICS,
        )

        self.assertTrue(WIDE_TACTICS and WIDE_ACT_TACTICS, "empty wide tables: the knob would route nothing")
        for shape, table in WIDE_TACTICS.items():
            self.assertIn(shape, SPLITK_TACTICS)
            self.assertTrue(all(MAX_SPLITK_ROWS < r <= 64 for r in table), f"{shape}: rows {sorted(table)}")
        self.assertTrue(all(MAX_SPLITK_ROWS < r <= 64 for r in WIDE_ACT_TACTICS))

    def test_wide_error_vs_fp64_and_tail(self):
        from sglang.srt.speculative.dflash_draft_splitk import (
            SPLITK_TACTICS,
            WIDE_TACTICS,
            DraftSplitKProjection,
        )

        gen = torch.Generator(device=DEV).manual_seed(20261102)
        failures, checked = [], 0
        for (n, k), table in WIDE_TACTICS.items():
            for seed_round in range(2):
                w = _weight(n, k, gen)
                proj = DraftSplitKProjection(w, SPLITK_TACTICS[(n, k)], wide=table)
                for rows in sorted(table):
                    for kind in KINDS:
                        x = _inputs(rows, k, gen, kind)
                        canvas = torch.full((rows + 16, n), CANVAS, device=DEV, dtype=torch.bfloat16)
                        y = proj.run(x, canvas[:rows])
                        stock = F.linear(x, w)
                        rel_f, worst = _errors(y, x, w)
                        stock_f, stock_worst = _errors(stock, x, w)
                        tag = f"{(n, k)} rows={rows} {table[rows]} {kind} round {seed_round}"
                        if not torch.isfinite(y).all():
                            failures.append(f"{tag}: non-finite output")
                        if rel_f > REL_F:
                            failures.append(f"{tag}: relF {rel_f:.3e} > {REL_F}")
                        if rel_f > STOCK_RATIO * stock_f + STOCK_FLOOR:
                            failures.append(f"{tag}: relF {rel_f:.3e} vs stock {stock_f:.3e}")
                        if worst > 1.0:
                            failures.append(f"{tag}: element error {worst:.2f}x bound")
                        if stock_worst > 1.0:
                            failures.append(f"{tag}: CONTROL stock element error {stock_worst:.2f}x bound")
                        if not bool((canvas[rows:] == CANVAS).all()):
                            failures.append(f"{tag}: wrote past the last row")
                        print(f"WIDE {n}x{k} rows={rows} {kind} relF {rel_f:.3e} stock {stock_f:.3e} worst {worst:.3f}")
                        checked += 1
        self.assertEqual(checked, 2 * len(KINDS) * sum(len(t) for t in WIDE_TACTICS.values()))
        self.assertFalse(failures, "\n".join(failures))

    def test_wide_act_bitwise_to_its_unfused_pair(self):
        """Fused gate_up + SiLU-mul at each WIDE_ACT_TACTICS row count == the same tactic's unfused split-K gate_up, then
        sglang's SiluAndMul (bitwise); that gate_up meets the FP64 bounds; no row past the last is written."""
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.speculative.dflash_draft_splitk import WIDE_ACT_TACTICS

        act, gen, failures, checked = SiluAndMul(), torch.Generator(device=DEV).manual_seed(31), [], 0
        for seed_round in range(2):
            w = _weight(2 * INTER, HIDDEN, gen)
            for rows, tactic in sorted(WIDE_ACT_TACTICS.items()):
                unfused, fused = _direct(w, rows, tactic), _direct(w, rows, tactic, act_fuse=True)
                for kind in KINDS:
                    x = _inputs(rows, HIDDEN, gen, kind)
                    gate_up = torch.empty(rows, 2 * INTER, device=DEV, dtype=torch.bfloat16)
                    unfused.run(x, gate_up)
                    ref = act(gate_up)
                    canvas = torch.full((rows + 16, INTER), CANVAS, device=DEV, dtype=torch.bfloat16)
                    fused.run(x, canvas[:rows])
                    tag = f"act rows={rows} {tactic} {kind} round {seed_round}"
                    n_diff, nan_ok = _bitwise_or_both_nan(canvas[:rows], ref)
                    if n_diff or not nan_ok:
                        failures.append(f"{tag}: {n_diff} elements differ from the unfused pair")
                    if not bool((canvas[rows:] == CANVAS).all()):
                        failures.append(f"{tag}: wrote past the last row")
                    rel_f, worst = _errors(gate_up, x, w)
                    stock_f, _ = _errors(F.linear(x, w), x, w)
                    if rel_f > REL_F or worst > 1.0 or rel_f > STOCK_RATIO * stock_f + STOCK_FLOOR:
                        failures.append(f"{tag}: gate_up relF {rel_f:.3e} (stock {stock_f:.3e}), element {worst:.2f}x")
                    checked += 1
        self.assertEqual(checked, 2 * len(KINDS) * len(WIDE_ACT_TACTICS))
        self.assertFalse(failures, "\n".join(failures))

    def test_knob_routes_exactly_the_listed_rows(self):
        """Knob off: every shape covers 1..8 rows only and a wide row count takes the stock method bitwise. Knob on: the
        listed rows equal the plugin projection of the table's tactic built directly (not through the module), the rest of
        9..64 stay stock, and at least one routed output differs from the stock GEMM (so routing is observable)."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.speculative import dflash_draft_splitk as dsk

        gen, failures, differs = torch.Generator(device=DEV).manual_seed(41), [], 0
        for (n, k), table in dsk.WIDE_TACTICS.items():
            lin = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16).to(DEV)
            with torch.no_grad():
                lin.weight.copy_(_weight(n, k, gen))
            w = lin.weight
            off = dsk._projection(w)
            if off.rows != frozenset(range(1, 9)):
                failures.append(f"{(n, k)} knob off covers {sorted(off.rows)}")
            with envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE.override(True):
                on = dsk._projection(w)
            if on.rows != frozenset(range(1, 9)) | frozenset(table):
                failures.append(f"{(n, k)} knob on covers {sorted(on.rows)}, table {sorted(table)}")
            for proj, label in ((off, "off"), (on, "on")):
                method = dsk.DraftSplitKLinearMethod(proj)
                for rows in range(9, 65):
                    x = _inputs(rows, k, gen, "normal")
                    try:
                        got = method.apply(lin, x)
                    except Exception as e:  # a routed row count without a prepared kernel
                        failures.append(f"{(n, k)} knob {label} rows={rows}: {type(e).__name__}: {str(e)[:80]}")
                        continue
                    stock = UnquantizedLinearMethod().apply(lin, x)
                    if label == "on" and rows in table:
                        want = torch.empty(rows, n, device=DEV, dtype=torch.bfloat16)
                        _direct(w, rows, table[rows]).run(x, want)
                        differs += int(not torch.equal(want, stock))
                    else:
                        want = stock
                    if not torch.equal(got, want):
                        failures.append(f"{(n, k)} knob {label} rows={rows}: not the expected path")
        self.assertGreater(differs, 0, "no routed output differs from stock: routing not observable")
        self.assertFalse(failures, "\n".join(failures[:40]))

    def test_mlp_forward_takes_the_fused_wide_projection(self):
        """With _WIDE and _ACT, DFlashMLP.forward takes the fused projection at each WIDE_ACT_TACTICS row count and the
        unfused path at other rows. The fused projection is built from a DIFFERENT weight than gate_up_proj, so each path
        is identified by its output."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.models.dflash import DFlashMLP
        from sglang.srt.speculative import dflash_draft_splitk as dsk

        gen = torch.Generator(device=DEV).manual_seed(53)

        def linear(n, k):
            m = ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16).to(DEV)
            with torch.no_grad():
                m.weight.copy_(_weight(n, k, gen))
            return m

        mlp = types.SimpleNamespace(
            gate_up_proj=linear(2 * INTER, HIDDEN), down_proj=linear(HIDDEN, INTER), act_fn=SiluAndMul(),
            _draft_all_reduce=None,
        )
        other = _weight(2 * INTER, HIDDEN, gen)
        with envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE.override(True):
            mlp.split_k_gate_up_act = dsk._projection(other, act_fuse=True)
        failures = []
        wide_rows = sorted(dsk.WIDE_ACT_TACTICS)
        others = [r for r in (9, 10, 15, 17, 63) if r not in dsk.WIDE_ACT_TACTICS]
        for rows in [8] + wide_rows + others:
            x = _inputs(rows, HIDDEN, gen, "normal")
            got = DFlashMLP.forward(mlp, x)
            fused_rows = rows <= 8 or rows in dsk.WIDE_ACT_TACTICS
            if fused_rows:
                want = mlp.down_proj(mlp.split_k_gate_up_act.run(x))[0]
            else:
                want = mlp.down_proj(mlp.act_fn(mlp.gate_up_proj(x)[0]))[0]
            if not torch.equal(got, want):
                failures.append(f"rows={rows}: not the {'fused' if fused_rows else 'unfused'} path")
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
