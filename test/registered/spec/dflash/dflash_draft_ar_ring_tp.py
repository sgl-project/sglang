# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""torchrun check: the ring o_proj / down_proj of the DFlash draft all-reduce sites (SGLANG_OPT_DFLASH_DRAFT_AR_RING,
speculative/dflash_draft_ar_ring.py) against the served pair it replaces (split-K projection storing the partial, then
the site convolving and publishing it), on any TP >= 2 (multicast memory required).

    torchrun --nproc-per-node 4 test/registered/spec/dflash/dflash_draft_ar_ring_tp.py

Reference = the served pair (DraftSplitKProjection.run + DraftArSites.run without the ring), whose outputs the AR-site
test checks against FP64. The ring pair must reproduce it BITWISE (same tiles, same K order, same casts, same conv
arithmetic, same sums); no tolerance.
  R1  rows 1 / 3 / 8, the o_proj shape (K 768, parity 0) and the down_proj shape (K 896, parity 1): new residual and
      normed row bitwise the served pair's, every rank the same bits, the mailbox re-armed after each site.
  R2  wiring (DraftRingLinearMethod + DraftArSites.ring_scope): inside a scope of its parity the method publishes and
      the next site polls (bitwise R1); outside any scope, in a scope of the other parity, or at 9 rows it returns
      the split-K partial (bitwise DraftSplitKProjection.run) and the site publishes as served.
  R3  a CUDA graph of 10 ring + polled sites (alternating parities, residual chained) replayed with new inputs equals
      10 eager served pairs bitwise.
  S   stress: ITERS rounds of 10 chained sites; before each ring projection and between it and its site every rank
      sleeps a random 0-50 us (its own generator), so ranks drift; each round's outputs vs the served chain (eager, no
      sleeps) on the same inputs, bitwise. Any mismatched round fails.
Mutants (K31_MUTATION; block4 and noparity are applied by this file, the others to a patched copy of the module source under test,
see MUTATIONS): block4 = the ring kernels built for draft block 4 instead of 8, so rows 4-7
of a block convolve without their predecessor (R1 at 8 rows must fail); noparity = every site of S on parity 0 (the
alternation the protocol needs; S must fail or trap); gate_rearm = with SGLANG_OPT_DFLASH_DRAFT_AR_GATE=1, the gate lane
re-arms the fragment it waited for, so its owner's poll never sees it (R1 must trap at the poll deadline; it also shows the
gated kernel is the one that ran). With SGLANG_OPT_DFLASH_DRAFT_AR_GATE=1 every polled site of <= 8 rows is gated and must
stay bitwise the (ungated) served pair. With SGLANG_OPT_DFLASH_DRAFT_AR_DEFERRED=1 the ring projections trigger after their
wait and every polled site of <= 8 rows runs deferred (before its wait, static output buffer, outputs cloned here because
this test's 10-site chains share 2 norms); it must stay bitwise the served pair, and at least one deferred launch must
happen; S then alternates rounds without sleeps (the site right behind its producer, as served) and with the pre-producer
sleep only. With SGLANG_OPT_DFLASH_DRAFT_AR_VALID=1 every polled site of <= 8 rows lands its slots with validity-reporting
bulk copies (Rubin); it must stay bitwise the served pair and at least one such launch must happen. Mutants for it:
valid_dup = slot 1 lands rank 0's slot (R1 must fail bitwise, deterministic); valid_once = one copy
round whatever its report (S must fail: a rank that reaches its site before a sleeping peer publishes reads armed words).
With SGLANG_OPT_DFLASH_DRAFT_RING_EARLY_COEF=1 the ring kernels load the conv coefficients during the main loop; they must stay
bitwise the served pair (each rank prints which form it built). Mutant for it: coef_row = each thread loads the coefficients
of the row after its own (R1 at 3 or 8 rows must fail bitwise).
RING_PARTS (default R1,R2,R3,S) and RING_ROWS (default 1,3,8)
trim a run; RING_STRESS_ITERS sets S's rounds.
Prints FAIL lines and one TP_RESULT line per rank; exits nonzero on any failure.
"""

import importlib
import importlib.util
import os
import sys
import tempfile
import types

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

HIDDEN, TAPS, GROUP, BLOCK, EPS = 7168, 2, 16, 8, 1e-6
GROUPS = HIDDEN // GROUP
DEADLINE_NS = 5_000_000_000  # test-only: ranks reach a site well within this
ITERS = int(os.environ.get("RING_STRESS_ITERS", "300"))
PARTS = set(os.environ.get("RING_PARTS", "R1,R2,R3,S").split(","))
R1_ROWS = tuple(int(r) for r in os.environ.get("RING_ROWS", "1,3,8").split(","))
SHAPES = ((768, 0), (896, 1))  # (K, parity): o_proj feeds the attention site, down_proj the MLP site
_SITE = "sglang.srt.speculative.dflash_draft_ar_site"
_RING = "sglang.srt.speculative.dflash_draft_ar_ring"
# K31_MUTATION -> (module, [(old, new), ...]): each a plausible bug in a patched copy of the module source.
MUTATIONS = {
    # the gate lane re-arms the fragment it waited for: the owner's poll can never see it
    "gate_rearm": (_SITE, [(
        "                poll(last + tid * slot, slot, 1, self.deadline_ns)\n",
        "                poll(last + tid * slot, slot, 1, self.deadline_ns)\n                rearm(last + tid * slot)\n",
    )]),
    # slot 1 lands rank 0's slot (rank 0 twice, rank 1 never)
    "valid_dup": (_SITE, [("                    source = s\n", "                    source = s\n                    source = 0 if s == 1 else s\n")]),
    # ignores the armed report (one round, whatever landed)
    "valid_once": (_SITE, [("            rounds = rounds + 1\n", "            rounds = rounds + 1\n            armed = Int32(0)\n")]),
    # each thread loads the conv coefficients of the row after its own
    "coef_row": (_RING, [(
        "                nc, mc = cutlass.min(n, rows - 1), cutlass.min(m, width - 8)\n",
        "                nc, mc = cutlass.min(n, rows - 1), cutlass.min(m, width - 8)\n                nc = cutlass.min(n + 1, rows - 1)\n",
    )]),
}


def load_mutated(modname, edits):
    """Import ``modname`` from a patched copy of its source file (a real file, so the CuTe DSL can read the kernels' source):
    each (old, new) of ``edits`` must occur exactly once. Later imports of ``modname`` get the copy."""
    spec = importlib.util.find_spec(modname)
    src = open(spec.origin).read()
    for old, new in edits:
        if src.count(old) != 1:
            print(f"K31_MUTATION anchor count {src.count(old)} in {spec.origin}: {old[:60]!r}", flush=True)
            sys.exit(3)
        src = src.replace(old, new)
    path = os.path.join(tempfile.mkdtemp(prefix="k31mut_"), os.path.basename(spec.origin))
    with open(path, "w") as f:
        f.write(src)
    mspec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(mspec)
    sys.modules[modname] = mod
    parent, _, leaf = modname.rpartition(".")
    mspec.loader.exec_module(mod)
    setattr(importlib.import_module(parent), leaf, mod)
    return mod


def main() -> int:
    local_rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    mutation = os.environ.get("K31_MUTATION", "")
    if mutation in MUTATIONS:
        load_mutated(*MUTATIONS[mutation])
    elif mutation not in ("", "block4", "noparity"):
        print(f"unknown K31_MUTATION {mutation}", flush=True)
        return 3
    if mutation:
        print(f"K31_MUTATION {mutation} applied", flush=True)

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashGroupedConv
    from sglang.srt.speculative import dflash_draft_ar_site as site_mod
    from sglang.srt.speculative import dflash_draft_ar_ring as ring_mod
    from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS, DraftSplitKLinearMethod, DraftSplitKProjection

    group = types.SimpleNamespace(world_size=world, rank_in_group=rank, device_group=dist.group.WORLD)
    sites = site_mod.DraftArSites(group, device, DEADLINE_NS)
    sites.rings = True
    print(f"[rank{rank}] gated poll {sites.gate} deferred {sites.deferred} valid {sites.valid}", flush=True)
    failures, checks = 0, 0

    def fail(msg):
        nonlocal failures
        failures += 1
        print(f"[rank{rank}] FAIL {msg}", flush=True)

    def check(ok, msg):
        nonlocal checks
        checks += 1
        if not ok:
            fail(msg)

    def gather(t):
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous())
        return out

    def bits_equal(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    def same_everywhere(t):
        return all(bits_equal(t, o) for o in gather(t))

    def armed():
        torch.cuda.synchronize()
        return bool((sites.mailbox.buffer.view(torch.int32) == site_mod.SENTINEL).all())

    shared = torch.Generator(device=device).manual_seed(17)
    own = torch.Generator(device=device).manual_seed(1000 + rank)

    def randn(*shape, gen=shared, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(torch.bfloat16)

    def make_conv():
        conv = DFlashGroupedConv(HIDDEN, BLOCK, TAPS, GROUP).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            base = torch.randn(2, TAPS, HIDDEN, generator=shared, device=device) * 0.3
            base[:, 0] += 1.0
            conv.base_kernel.copy_(base)
        return conv

    def make_norm():
        norm = RMSNorm(HIDDEN, eps=EPS).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1.0 + torch.randn(HIDDEN, generator=shared, device=device) * 0.1)
        return norm

    convs = (make_conv(), make_conv())  # attention conv (parity 0), MLP conv (parity 1)
    norms = (make_norm(), make_norm())
    served, rings = {}, {}
    for k, parity in SHAPES:
        weight = randn(HIDDEN, k, gen=own, scale=0.03)  # this rank's row-parallel shard
        tactic = SPLITK_TACTICS[(HIDDEN, k)]
        served[parity] = DraftSplitKProjection(weight, tactic)
        rings[parity] = ring_mod.DraftRingProjection(
            weight, tactic, 4 if mutation == "block4" else BLOCK, trigger_after_wait=sites.deferred
        )
    in_k = {parity: k for k, parity in SHAPES}
    print(f"[rank{rank}] ring early_coef {rings[0].early_coef} {rings[1].early_coef}", flush=True)

    def served_pair(parity, x, projection, residual):
        partial = served[parity].run(x)
        return sites.run(parity, partial, projection, convs[parity], residual, norms[parity])

    def ring_pair(parity, x, projection, residual, poll_parity=None, sleep=None):
        poll_parity = parity if poll_parity is None else poll_parity
        if sleep is not None and sleep[0] > 0:
            torch.cuda._sleep(sleep[0])
        out = rings[parity].run(x, projection, convs[parity].base_kernel.detach()[1], sites.ring_address(poll_parity))
        sites.ring_published(poll_parity, x.shape[0], rings[parity].trigger_after_wait)
        if sleep is not None and sleep[1] > 0:
            torch.cuda._sleep(sleep[1])
        normed = sites.run(poll_parity, out, projection, convs[parity], residual, norms[parity])
        # Deferred sites write one static buffer per (norm, rows): copy before the next site of this norm reuses it.
        return normed.clone() if sites.deferred else normed

    def inputs(parity, rows):
        x = randn(rows, in_k[parity], gen=own, scale=1.0)
        projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3)
        return x, projection

    with torch.inference_mode():
        # R1
        for k, parity in (SHAPES if "R1" in PARTS else ()):
            for rows in R1_ROWS:
                x, projection = inputs(parity, rows)
                res0 = randn(rows, HIDDEN)
                r_ref, r_ring = res0.clone(), res0.clone()
                n_ref = served_pair(parity, x, projection, r_ref)
                check(armed(), f"R1 K {k} rows {rows}: mailbox not re-armed after the served site")
                dist.barrier()
                n_ring = ring_pair(parity, x, projection, r_ring)
                check(armed(), f"R1 K {k} rows {rows}: mailbox not re-armed after the ring site")
                dist.barrier()
                nd = int((n_ring.view(torch.int16) != n_ref.view(torch.int16)).sum())
                rd = int((r_ring.view(torch.int16) != r_ref.view(torch.int16)).sum())
                check(nd == 0 and rd == 0, f"R1 K {k} rows {rows}: {nd} normed / {rd} residual cells differ from the served pair")
                check(same_everywhere(n_ring) and same_everywhere(r_ring), f"R1 K {k} rows {rows}: ranks differ")
                if rank == 0:
                    print(f"R1 K {k} rows {rows}: normed diff {nd} residual diff {rd}", flush=True)

        # R2: the linear method's routing
        k, parity = SHAPES[0]
        rows = 8
        method = ring_mod.DraftRingLinearMethod(DraftSplitKLinearMethod(served[parity]), rings[parity], sites, parity)
        layer = types.SimpleNamespace(weight=served[parity]._prepared.weight)
        check(not rings[parity].covers(inputs(parity, 9)[0]), "R2: the ring covers 9 rows")
        cases = ((parity, "own scope", True), (None, "no scope", False), (1 - parity, "other parity", False))
        for scope, label, want_ring in (cases if "R2" in PARTS else ()):
            x, projection = inputs(parity, rows)
            sites._published = None
            if scope is None:
                got = method.apply(layer, x)
            else:
                with sites.ring_scope(scope, projection, convs[scope]):
                    got = method.apply(layer, x)
            published = sites._published is not None
            check(published == want_ring, f"R2 {label}: published {published}, want {want_ring}")
            if want_ring:
                res0 = randn(rows, HIDDEN)
                r_ring, r_ref = res0.clone(), res0.clone()
                n_ring = sites.run(parity, got, projection, convs[parity], r_ring, norms[parity])
                check(armed(), f"R2 {label}: mailbox not re-armed")
                dist.barrier()
                n_ref = served_pair(parity, x, projection, r_ref)
                check(armed(), f"R2 {label}: mailbox not re-armed (served)")
                dist.barrier()
                check(bits_equal(n_ring, n_ref) and bits_equal(r_ring, r_ref), f"R2 {label}: not the served pair's bits")
            else:
                check(bits_equal(got, served[parity].run(x)), f"R2 {label}: not the split-K partial")
            sites._published = None

        # R3: graph replay of 10 ring + polled sites == 10 eager served pairs
        rows = 8
        xs = [inputs(i % 2, rows) for i in range(10)]
        residual = randn(rows, HIDDEN)

        def ten_ring():
            return [ring_pair(i % 2, xs[i][0], xs[i][1], residual) for i in range(10)]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ten_ring() if "R3" in PARTS else None
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        if "R3" in PARTS:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outs = ten_ring()
            torch.cuda.synchronize()
            dist.barrier()
        for replay in range(3 if "R3" in PARTS else 0):
            fresh = [inputs(i % 2, rows) for i in range(10)]
            new_residual = randn(rows, HIDDEN)
            for (x, p), (fx, fp) in zip(xs, fresh):
                x.copy_(fx)
                p.copy_(fp)
            residual.copy_(new_residual)
            graph.replay()
            torch.cuda.synchronize()
            got = [o.clone() for o in outs] + [residual.clone()]
            check(armed(), f"R3 replay {replay}: mailbox not re-armed")
            dist.barrier()
            r_eager = new_residual.clone()
            want = [served_pair(i % 2, fresh[i][0], fresh[i][1], r_eager) for i in range(10)] + [r_eager]
            diff = sum(int((g.view(torch.int16) != w.view(torch.int16)).sum()) for g, w in zip(got, want))
            check(diff == 0, f"R3 replay {replay}: {diff} cells differ from the eager served pairs")
            check(armed(), f"R3 replay {replay}: mailbox not re-armed (served)")
            dist.barrier()

        # S: stress with per-rank random sleeps
        delays = torch.Generator().manual_seed(7000 + rank)
        cycles_max = 100_000  # tens of microseconds
        bad_rounds = 0
        for it in range(ITERS if "S" in PARTS else 0):
            xs = [inputs(i % 2, rows) for i in range(10)]
            res0 = randn(rows, HIDDEN)
            r_ref = res0.clone()
            want = [served_pair(i % 2, x, p, r_ref) for i, (x, p) in enumerate(xs)] + [r_ref]
            torch.cuda.synchronize()
            dist.barrier()
            r_ring = res0.clone()
            got = []
            for i, (x, p) in enumerate(xs):
                sleep = torch.randint(0, cycles_max, (2,), generator=delays).tolist()
                if sites.deferred:  # even rounds: no sleeps (site right behind its producer); odd: pre-producer sleep only
                    sleep = [0, 0] if it % 2 == 0 else [sleep[0], 0]
                poll = 0 if mutation == "noparity" else i % 2
                got.append(ring_pair(i % 2, x, p, r_ring, poll_parity=poll, sleep=sleep))
            got.append(r_ring)
            torch.cuda.synchronize()
            diff = sum(int((g.view(torch.int16) != w.view(torch.int16)).sum()) for g, w in zip(got, want))
            if diff:
                bad_rounds += 1
                if bad_rounds <= 3:
                    print(f"[rank{rank}] S round {it}: {diff} cells differ", flush=True)
            dist.barrier()
        if "S" in PARTS:
            check(bad_rounds == 0, f"S: {bad_rounds} / {ITERS} rounds differ from the served chain")
            check(armed(), "S: mailbox not re-armed at the end")
            if rank == 0:
                print(f"S rounds {ITERS} bad {bad_rounds} (rank 0)", flush=True)

    if sites.deferred and PARTS & {"R1", "R3", "S"} and mutation != "block4":
        check(sites.deferred_runs > 0, "deferred: no deferred site ran")
        if rank == 0:
            print(f"deferred site launches {sites.deferred_runs}", flush=True)
    if sites.valid and PARTS & {"R1", "R3", "S"} and not mutation:
        check(sites.valid_runs > 0, "valid: no validity-landed site ran")
        if rank == 0:
            print(f"valid site launches {sites.valid_runs}", flush=True)
    dist.barrier()
    print(f"[rank{rank}] TP_RESULT world={world} checks={checks} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
