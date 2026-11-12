#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_VERIFY_PROLOGUE_FUSED (speculative/dflash_verify_prologue.py + the sparse verify binding's forward).

Claim under test: with the knob on, the sparse TARGET_VERIFY binding hands the megakernel bitwise the same KDA valid mask,
stable-rows scratch indices, per-request page table and score contexts as the served torch ops, in eager and CUDA-graph
replay. Reference = the served torch ops (the knob-off code path of the same binding), not the code under test.

Checks (1 GPU, weight-free):
  P1  fused_verify_prologue vs the torch ops: valid / table / contexts bitwise, b 1..8, q 8 / 5, pages 513 / 16384 / 16385,
      slots with -1 and 0, table rows past b untouched (sentinel).
  P2  K31SparseVerifyHostBinding.forward knob on vs off (stubbed backend metadata, recorded KdaVerifyStep /
      SparseVerifyStep arguments): scratch, valid, table, score contexts bitwise; with stable rows off AND on
      (pp_spec_stable_rows_enabled patched True: the callable mask must give the same scratch); ENGAGED logged.
  P3  the fused call captured in a CUDA graph, inputs rewritten in place, replayed: outputs equal the torch ops on the new inputs.
  P4  knob on with an int64 page table: falls back (NOT ENGAGED) and is still bitwise the served path.
Mutants (K31_MUTATION, applied to a copy of the source under test; each must fail >= 1 check):
  vpre_stride (kernel reads request rows at stride q - 1), vpre_valid (s > 0), vpre_wire (binding passes req_pool_indices as
  slots to the fused call).
Output: one 'CHECK <name> ok=0|1' line per check, then 'RESULT mutant=<m> checks=N failures=F'; rc 1 iff F > 0.
"""

import importlib.util
import logging
import os
import sys
import tempfile
import types

import torch

MUT = os.environ.get("K31_MUTATION", "")
_PRO = "sglang.srt.speculative.dflash_verify_prologue"
_BIND = "sglang.srt.kernel_plugins.nvidia.kimi_k31_verify_dsa_binding"
MUTATIONS = {
    "vpre_stride": (_PRO, "src_row = (r * q).to(tl.int64)", "src_row = (r * (q - 1)).to(tl.int64)"),
    "vpre_valid": (_PRO, "tl.store(valid_ptr + r, (s >= 0).to(tl.int32))", "tl.store(valid_ptr + r, (s > 0).to(tl.int32))"),
    "vpre_wire": (
        _BIND,
        "slots, full.real_page_table, full.cache_seqlens_int32, self.request_page_table, b, q\n            )\n        else:",
        "forward_batch.req_pool_indices[:b], full.real_page_table, full.cache_seqlens_int32, self.request_page_table, b, q\n"
        "            )\n        else:",
    ),
}
FAILS, CHECKS = [], [0]


def check(name, ok):
    CHECKS[0] += 1
    print(f"CHECK {name} ok={int(bool(ok))}", flush=True)
    if not ok:
        FAILS.append(name)


def load_mutated(modname, old, new):
    """Import ``modname`` from a mutated copy of its source file (a real file, so triton.jit can read the source)."""
    spec = importlib.util.find_spec(modname)
    src = open(spec.origin).read()
    if src.count(old) != 1:
        print(f"MUTATION {MUT} anchor count {src.count(old)} in {spec.origin}", flush=True)
        sys.exit(3)
    d = tempfile.mkdtemp(prefix="k31mut_")
    path = os.path.join(d, os.path.basename(spec.origin))
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    mspec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(mspec)
    sys.modules[modname] = mod
    parent, _, leaf = modname.rpartition(".")
    mspec.loader.exec_module(mod)
    setattr(importlib.import_module(parent), leaf, mod)
    print(f"K31_MUTATION {MUT} applied to {modname}", flush=True)


if MUT:
    if MUT not in MUTATIONS:
        print(f"unknown K31_MUTATION {MUT}")
        sys.exit(3)
    load_mutated(*MUTATIONS[MUT])

import sglang.srt.layers.attention.linear.utils as lin_utils  # noqa: E402
from sglang.srt.speculative import dflash_verify_prologue as pro  # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(1107)


def served_ops(slots, rpt, lens, table_buf, b, q):
    valid = (slots[:b] >= 0).to(torch.int32)
    pages = rpt.shape[1]
    table = table_buf[: b * pages].view(b, pages)
    table.copy_(rpt[: b * q : q])
    contexts = lens[:b].view(b, 1).expand(b, q).contiguous()
    return valid, table, contexts


def make(b, q, pages, nreq=64):
    slots = torch.randint(0, 1000, (nreq,), dtype=torch.int32, device=dev)
    slots[1::3] = -1
    slots[2::5] = 0
    rpt = torch.randint(0, 2**31 - 1, (nreq * 8, pages), dtype=torch.int32, device=dev)
    lens = torch.randint(1, 1 << 20, (nreq,), dtype=torch.int32, device=dev)
    return slots, rpt, lens


# ---- P1
for pages in (513, 16384, 16385):
    for q in (8, 5):
        for b in (1, 2, 3, 4, 5, 8):
            slots, rpt, lens = make(b, q, pages)
            ref_buf = torch.full((16 * pages,), -7, dtype=torch.int32, device=dev)
            got_buf = torch.full((16 * pages,), -7, dtype=torch.int32, device=dev)
            rv, rt, rc = served_ops(slots, rpt, lens, ref_buf, b, q)
            check(f"P1 supported b{b} q{q} p{pages}", pro.unsupported_reason(slots[:b], rpt, lens, got_buf, b, q) is None)
            gv, gt, gc = pro.fused_verify_prologue(slots[:b], rpt, lens, got_buf, b, q)
            torch.cuda.synchronize()
            tag = f"b{b} q{q} p{pages}"
            check(f"P1 valid {tag}", torch.equal(gv, rv))
            check(f"P1 table {tag}", torch.equal(gt, rt) and torch.equal(got_buf, ref_buf))
            check(f"P1 contexts {tag}", torch.equal(gc, rc) and gc.shape == rc.shape and gc.is_contiguous())


# ---- P2 binding wiring
def binding_run(knob, b, q, pages, stable_rows, page_dtype=torch.int32):
    import sglang.srt.model_executor.forward_context as fctx
    import sglang.srt.model_executor.runner_utils.capture_mode as cmode
    import sgl_nvidia_mega.kimi_k31.verify_binding as kvb
    import sgl_nvidia_mega.kimi_k31.dsa.verify as dver

    bind = importlib.import_module(_BIND)
    torch.manual_seed(1000 + b * 10 + q)
    rows = b * q
    slots, rpt, lens = make(b, q, pages)
    rpt = rpt.to(page_dtype)
    full = types.SimpleNamespace(
        real_page_table=rpt, cache_seqlens_int32=lens, dsa_seqlens_expanded=torch.arange(rows, dtype=torch.int32, device=dev),
        paged_mqa_schedule_metadata=None, topk_v2_plan=None,
    )
    linear = types.SimpleNamespace(
        forward_metadata=types.SimpleNamespace(mamba_cache_indices=slots),
        verify_intermediate_state_indices=torch.arange(64, dtype=torch.int32, device=dev) + 100,
        req_to_token_pool=types.SimpleNamespace(size=64),
    )
    backend = types.SimpleNamespace(full_attn_backend=types.SimpleNamespace(forward_metadata=full), linear_attn_backend=linear)
    rec = {}
    saved = (fctx.get_attn_backend, cmode.get_capture_attention_variant, kvb.KdaVerifyStep, dver.SparseVerifyStep,
             lin_utils.pp_spec_stable_rows_enabled)
    fctx.get_attn_backend = lambda: backend
    cmode.get_capture_attention_variant = lambda: None
    kvb.KdaVerifyStep = lambda *a: ("KDA", a)
    dver.SparseVerifyStep = lambda *a, **k: types.SimpleNamespace(args=a, kwargs=k)
    lin_utils.pp_spec_stable_rows_enabled = (lambda: True) if stable_rows else (lambda: False)
    try:
        binding = bind.K31SparseVerifyHostBinding.__new__(bind.K31SparseVerifyHostBinding)
        binding._prologue_knob = knob
        binding.request_page_table = torch.full((16 * pages,), -7, dtype=torch.int32, device=dev)
        binding.aux = torch.zeros(256, dtype=torch.int32, device=dev)
        binding.graph_policy = types.SimpleNamespace(capture_labels=("dense", "sparse"), select=lambda fb: "sparse")
        lm = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=lambda ids: ids),
            logits_processor=lambda ids, hidden, head, fb, aux: hidden, lm_head=None,
        )
        binding.model = types.SimpleNamespace(language_model=lm)

        def run_verify(emb, view):
            rec["view"] = view
            return emb, None

        built = types.SimpleNamespace(run_verify=run_verify)
        fb = types.SimpleNamespace(
            batch_size=b, req_pool_indices=torch.arange(64, dtype=torch.int64, device=dev) * 3 + 1,
            out_cache_loc=torch.arange(rows, dtype=torch.int64, device=dev), spec_info=None,
        )
        binding.forward(built, types.SimpleNamespace(rows=rows), torch.arange(rows, device=dev), None, fb)
        torch.cuda.synchronize()
    finally:
        (fctx.get_attn_backend, cmode.get_capture_attention_variant, kvb.KdaVerifyStep, dver.SparseVerifyStep,
         lin_utils.pp_spec_stable_rows_enabled) = saved
    view = rec["view"]
    kda = view.args[3][1]
    out = dict(scratch=kda[1].clone(), valid=kda[2].clone(), table=view.args[0].clone(),
               contexts=view.kwargs["score_contexts"].clone(), table_buf=binding.request_page_table.clone(),
               slots_same=kda[0].data_ptr() == slots.data_ptr(), lens_rows=view.args[1].shape[0],
               logged=sorted(binding.__dict__.get("_prologue_logged", ())))
    return out


for stable in (False, True):
    for b, q, pages in ((1, 8, 16385), (3, 8, 16385), (4, 5, 513), (8, 8, 16384)):
        tag = f"b{b} q{q} p{pages} stable{int(stable)}"
        try:
            off = binding_run(False, b, q, pages, stable)
            on = binding_run(True, b, q, pages, stable)
        except Exception as e:  # a crash fails the check (it is not a pass)
            check(f"P2 run {tag} raised {type(e).__name__}: {str(e)[:160]}", False)
            continue
        for k in ("scratch", "valid", "table", "contexts", "table_buf"):
            check(f"P2 {k} {tag}", off[k].shape == on[k].shape and torch.equal(off[k], on[k]))
        check(f"P2 engaged {tag}", on["logged"] == ["ENGAGED"] and off["logged"] == [])
        check(f"P2 slots/lens {tag}", on["slots_same"] and off["slots_same"] and on["lens_rows"] == b)

# ---- P3 graph replay
for b, q, pages in ((3, 8, 16385), (4, 8, 16385)):
    slots, rpt, lens = make(b, q, pages)
    buf = torch.full((16 * pages,), -7, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pro.fused_verify_prologue(slots[:b], rpt, lens, buf, b, q)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gv, gt, gc = pro.fused_verify_prologue(slots[:b], rpt, lens, buf, b, q)
    slots2, rpt2, lens2 = make(b, q, pages)
    slots.copy_(slots2)
    rpt.copy_(rpt2)
    lens.copy_(lens2)
    g.replay()
    torch.cuda.synchronize()
    ref = torch.full((16 * pages,), -7, dtype=torch.int32, device=dev)
    rv, rt, rc = served_ops(slots, rpt, lens, ref, b, q)
    tag = f"b{b} q{q} p{pages}"
    check(f"P3 replay valid {tag}", torch.equal(gv, rv))
    check(f"P3 replay table {tag}", torch.equal(gt, rt))
    check(f"P3 replay contexts {tag}", torch.equal(gc, rc))

# ---- P4 unsupported dtype falls back and stays the served path
try:
    off = binding_run(False, 3, 8, 513, False, page_dtype=torch.int64)
    on = binding_run(True, 3, 8, 513, False, page_dtype=torch.int64)
    for k in ("scratch", "valid", "table", "contexts", "table_buf"):
        check(f"P4 {k}", torch.equal(off[k], on[k]))
    check("P4 not engaged", len(on["logged"]) == 1 and on["logged"][0].startswith("NOT ENGAGED"))
except Exception as e:
    check(f"P4 raised {type(e).__name__}: {str(e)[:160]}", False)

print(f"RESULT mutant={MUT or 'none'} checks={CHECKS[0]} failures={len(FAILS)}", flush=True)
for f in FAILS[:20]:
    print(f"  failed: {f}")
sys.exit(1 if FAILS else 0)
