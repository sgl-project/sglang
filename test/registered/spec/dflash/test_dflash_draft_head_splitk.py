#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS (models/dflash.py _project_candidate_logits, _HeadSplitK).

Claim under test: with rows "21,28" the draft head projects 21- and 28-row blocks (batches of 3 / 4 requests at 7 drafted) on the plugin
split-K projection (T64s1) with an error vs FP64 no worse than torch.matmul's, every other row count stays bitwise
torch.matmul, the knob off is bitwise torch.matmul everywhere, and the projection replays in a CUDA graph. Reference =
an FP64 product of the same BF16 operands and torch.matmul (the served path), not the code under test.

Checks (1 GPU, weight-free, TP16 per-rank head shape [10240, 7168] BF16):
  H0  knob unset: rows 7 / 14 / 21 / 28 bitwise torch.matmul, nothing prepared.
  H1  knob "21,28": rows 7 / 14 bitwise torch.matmul (row gate); rows 21 / 28 engaged (logged row set) and, per row count,
      mean |err| <= 1.01 x torch.matmul's and max |err| <= 1.5 x torch.matmul's vs FP64 (tolerances fixed here before any
      run; a first measurement read 1.000 / <= 1.003), no non-finite value; top-16 id-set agreement with torch.matmul printed (INFO).
  H2  CUDA graph: after the eager first use, a captured 21-row projection replays bitwise equal to the eager projection of
      the rewritten input.
  H3  a 21-row input whose row pitch is not a multiple of 8 elements falls back to bitwise torch.matmul.
  H4  first use inside a CUDA graph capture raises (the projection compiles eagerly only).
Mutants (K31_MUTATION on a copy of models/dflash.py; each must fail >= 1 check):
  head_rows (the gate drops the row-set test: 14 rows reach the projection), head_perm (rows reach the kernel reversed).
Output: 'CHECK <name> ok=0|1' lines, 'INFO ...', then 'RESULT mutant=<m> checks=N failures=F'; rc 1 iff F > 0.
"""

import importlib.util
import os
import sys
import tempfile
import types

import torch

MUT = os.environ.get("K31_MUTATION", "")
_MOD = "sglang.srt.models.dflash"
MUTATIONS = {
    "head_rows": ("            and x.shape[0] in self.rows\n", "            and x.shape[0] > 0\n"),
    "head_perm": ("return self._prepared.run(x, x.new_empty((rows, self.n)))",
                  "return self._prepared.run(x.flip(0).contiguous(), x.new_empty((rows, self.n)))"),
}
FAILS, CHECKS = [], [0]


def check(name, ok):
    CHECKS[0] += 1
    print(f"CHECK {name} ok={int(bool(ok))}", flush=True)
    if not ok:
        FAILS.append(name)


if MUT:
    if MUT not in MUTATIONS:
        print(f"unknown K31_MUTATION {MUT}")
        sys.exit(3)
    old, new = MUTATIONS[MUT]
    spec = importlib.util.find_spec(_MOD)
    src = open(spec.origin).read()
    if src.count(old) != 1:
        print(f"MUTATION {MUT} anchor count {src.count(old)}")
        sys.exit(3)
    path = os.path.join(tempfile.mkdtemp(prefix="k31mut_"), "dflash.py")
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    mspec = importlib.util.spec_from_file_location(_MOD, path)
    D = importlib.util.module_from_spec(mspec)
    sys.modules[_MOD] = D
    mspec.loader.exec_module(D)
    print(f"K31_MUTATION {MUT} applied", flush=True)
else:
    import sglang.srt.models.dflash as D

dev = torch.device("cuda")
torch.manual_seed(1108)
V, H = 10240, 7168
W = (torch.randn(V, H, device=dev) * 0.02).to(torch.bfloat16)
HEAD = types.SimpleNamespace(weight=W)
ENV = "SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS"


def proj(x):
    return D._project_candidate_logits(x, HEAD, num_org=V, use_quant_head=False)


def stock(x):
    return torch.matmul(x, W[:V].T)


def run_case(name, fn):
    try:
        fn()
    except Exception as e:  # a crash fails its check
        check(f"{name} raised {type(e).__name__}: {str(e)[:200]}", False)


X = {r: torch.randn(r, H, device=dev).to(torch.bfloat16) for r in (7, 14, 21, 28)}

# ---- H0
os.environ.pop(ENV, None)
D._HEAD_SPLITK.clear()


def h0():
    for r, x in X.items():
        check(f"H0 rows{r} bitwise matmul", torch.equal(proj(x), stock(x)))
    check("H0 nothing prepared", len(D._HEAD_SPLITK) == 0)


run_case("H0", h0)

# ---- H1
os.environ[ENV] = "21,28"
D._HEAD_SPLITK.clear()


def h1():
    for r in (7, 14):
        check(f"H1 rows{r} bitwise matmul (gated off)", torch.equal(proj(X[r]), stock(X[r])))
    for r in (21, 28):
        x = X[r]
        out, ref = proj(x), stock(x)
        f64 = x.double() @ W.double().T
        e_new, e_ref = (out.double() - f64).abs(), (ref.double() - f64).abs()
        mr = e_new.mean().item() / e_ref.mean().item()
        xr = e_new.max().item() / e_ref.max().item()
        a = torch.topk(out.float(), 16, dim=-1).indices.sort(dim=-1).values
        b = torch.topk(ref.float(), 16, dim=-1).indices.sort(dim=-1).values
        print(f"INFO H1 rows{r} mean_ratio {mr:.4f} max_ratio {xr:.4f} bitwise_to_matmul {int(torch.equal(out, ref))} "
              f"top16_sets_equal {int((a == b).all(dim=-1).sum())}/{r}", flush=True)
        check(f"H1 rows{r} finite", bool(torch.isfinite(out).all()))
        check(f"H1 rows{r} mean err ratio {mr:.4f} <= 1.01", mr <= 1.01)
        check(f"H1 rows{r} max err ratio {xr:.4f} <= 1.5", xr <= 1.5)
    p = D._HEAD_SPLITK.get((W.data_ptr(), V))
    check("H1 engaged rows == {21, 28}", p is not None and p.engaged_rows == {21, 28})


run_case("H1", h1)


# ---- H2
def h2():
    x = X[21].clone()
    proj(x)  # eager first use (already prepared by H1; harmless)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        proj(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = proj(x)
    x.copy_(torch.randn(21, H, device=dev).to(torch.bfloat16))
    g.replay()
    torch.cuda.synchronize()
    check("H2 graph replay == eager", torch.equal(out, proj(x)))


run_case("H2", h2)


# ---- H3
def h3():
    big = torch.randn(21, H + 1, device=dev).to(torch.bfloat16)
    x = big[:, :H]  # row pitch 7169 elements: not a multiple of 8
    check("H3 odd pitch bitwise matmul", torch.equal(proj(x), stock(x)))


run_case("H3", h3)


# ---- H4
def h4():
    D._HEAD_SPLITK.clear()
    x = X[21]
    g = torch.cuda.CUDAGraph()
    raised = False
    try:
        with torch.cuda.graph(g):
            proj(x)
    except Exception as e:  # capture_end may raise its own error on top of this one: walk the chain
        while e is not None and not raised:
            raised = "first head projection inside a CUDA graph capture" in str(e)
            e = e.__context__
    check("H4 first use under capture raises", raised)


run_case("H4", h4)

print(f"RESULT mutant={MUT or 'none'} checks={CHECKS[0]} failures={len(FAILS)}", flush=True)
for f in FAILS[:20]:
    print(f"  failed: {f}")
sys.exit(1 if FAILS else 0)
