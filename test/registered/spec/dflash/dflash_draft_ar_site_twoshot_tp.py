# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""torchrun check: the two-shot DFlash draft all-reduce site (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS,
speculative/dflash_draft_ar_site.py DraftArSiteTwoShot) against the one-shot site it replaces, on any TP >= 2 whose
HIDDEN / TP is a multiple of 8 (multicast memory required).

    torchrun --nproc-per-node 4 test/registered/spec/dflash/dflash_draft_ar_site_twoshot_tp.py

The reference is the served one-shot site (its own accuracy vs the stock chain and FP64 is dflash_draft_ar_site_tp.py's
check A). The two-shot site sums the same TP convolved BF16 words in the same rank order and rounds at the same points,
so the claim is bit equality. Checks; fixed here before any run of this file:
  T1 rows 8 / 16 / 24 / 32 / 64, partial scales 0.5 and 8.0: the two-shot site's new residual and normed row equal the one-shot
     site's bitwise on every rank; every rank holds the same bits; the two-shot launch counter rose by one per call and
     the one-shot instance never launched a two-shot site.
  T2 after every two-shot call (followed by a collective before the next site), every word of the two-shot box (receive
     slots and gather rows, both parities) is the sentinel again.
  T3 a CUDA graph of 10 chained two-shot sites at 64 rows (alternating parities), replayed 3 times with new inputs,
     equals 10 eager two-shot sites and 10 eager one-shot sites bitwise.
  T4 a NaN partial element on the last rank: the NaN masks equal the one-shot site's; all partials -0.0 with negative
     tap-1 coefficients: new residual and normed row equal the one-shot site's bitwise.
  T5 (informative, no pass line): per-site time, one-shot vs two-shot, rows 16 / 32 / 64: a graph of 20 chained sites
     per form, replayed in the order A B B A (A = one-shot), 5 warm-up + 30 timed replays each; the max over ranks of the
     per-rank median replay time / 20.
Mutants (K31_MUTATION, applied to a patched copy of the module source; they change only the two-shot kernel): ts_noconv (raw partial instead of its side-1 convolution)
must fail T1; ts_norearm (gather row not re-armed) must fail T2.
Batched owner polls (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED=1 in the environment of the whole run): the two-shot instance is
  built batched, so T1-T4 check the batched kernel against the one-shot site; B0: the batched launch counter equals the two-shot
  launch counter (0 when the knob is unset).
Prints FAIL lines and one TP_RESULT line per rank; exits nonzero on any failure.
"""

import importlib
import importlib.util
import os
import statistics
import sys
import tempfile

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

HIDDEN, TAPS, GROUP, BLOCK, EPS = 7168, 2, 16, 8, 1e-6
GROUPS = HIDDEN // GROUP
DEADLINE_NS = 20_000_000_000  # a test-only bound: ranks reach a site seconds apart at most

_SITE = "sglang.srt.speculative.dflash_draft_ar_site"
# K31_MUTATION -> (module, [(old, new), ...]): each a plausible bug in a patched copy of the module source.
MUTATIONS = {
    # the two-shot site pushes the raw partial instead of its side-1 convolution
    "ts_noconv": (_SITE, [(
        "            words = self.contribution(partial, proj, base, row, col)\n"
        "            twoshot_push(recv, peers, rank, row, col, words)\n        if tid < self.share:\n",
        "            raw = load_words(partial.iterator + row * HIDDEN + col, count=4, mutable=True)\n"
        "            words = tuple(pack_pair(bf16(raw[k], False), bf16(raw[k], True), policy=\"latent\") for k in range(4))\n"
        "            twoshot_push(recv, peers, rank, row, col, words)\n        if tid < self.share:\n",
    )]),
    # the two-shot site leaves its gather row published (no re-arm)
    "ts_norearm": (_SITE, [("            if cutlass.const_expr(not self.drr_defer):\n                rearm(source)\n",
                            "            if cutlass.const_expr(False):\n                rearm(source)\n")]),
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
    elif mutation:
        print(f"unknown K31_MUTATION {mutation}", flush=True)
        return 3
    if mutation:
        print(f"K31_MUTATION {mutation} applied", flush=True)

    import types

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashGroupedConv
    from sglang.srt.speculative import dflash_draft_ar_site as site_mod

    group = types.SimpleNamespace(world_size=world, rank_in_group=rank, device_group=dist.group.WORLD)
    one = site_mod.DraftArSites(group, device, DEADLINE_NS, twoshot_rows=0)
    two = site_mod.DraftArSites(group, device, DEADLINE_NS, twoshot_rows=1)
    assert one.twoshot is None and two.twoshot is not None
    want_batched = os.environ.get("SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED", "0").lower() in ("1", "true")
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

    def same_everywhere(t):
        return all(torch.equal(t.view(torch.int16), o.view(torch.int16)) for o in gather(t))

    def bits_equal(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    def box_armed():
        torch.cuda.synchronize()
        box = two.twoshot
        return all(bool((t.view(torch.int32) == site_mod.SENTINEL).all()) for t in (box.recv, box.gather))

    shared = torch.Generator(device=device).manual_seed(17)  # replicated tensors: the same on every rank
    own = torch.Generator(device=device).manual_seed(1000 + rank)  # this rank's partials

    def randn(*shape, gen=shared, scale=1.0):
        return torch.randn(*shape, generator=gen, device=device) * scale

    conv = DFlashGroupedConv(HIDDEN, BLOCK, TAPS, GROUP).to(device=device, dtype=torch.bfloat16)
    norm = RMSNorm(HIDDEN, eps=EPS).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        base = randn(2, TAPS, HIDDEN, scale=0.3)
        base[:, 0] += 1.0
        conv.base_kernel.copy_(base)
        conv.kernel_projection.weight.copy_(randn(2 * TAPS * GROUPS, HIDDEN, scale=0.003))
        norm.weight.copy_(1.0 + randn(HIDDEN, scale=0.1))

    parity = {"one": 0, "two": 0}

    def run(form, partial, projection, residual):
        sites = one if form == "one" else two
        p = parity[form]
        parity[form] ^= 1
        return sites.run(p, partial, projection, conv, residual, norm)

    with torch.inference_mode():
        # T1, T2
        for rows in (8, 16, 24, 32, 64):
            for kind, scale in (("normal", 0.5), ("large", 8.0)):
                partial = randn(rows, HIDDEN, gen=own, scale=scale).to(torch.bfloat16)
                projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
                res0 = randn(rows, HIDDEN).to(torch.bfloat16)
                r_one, r_two = res0.clone(), res0.clone()
                n_one = run("one", partial, projection, r_one)
                dist.barrier()
                before = two.twoshot_runs
                n_two = run("two", partial, projection, r_two)
                check(two.twoshot_runs == before + 1 and one.twoshot_runs == 0, f"T1 rows {rows} {kind}: launch counters")
                # B0 (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED set for the run): every two-shot launch took the batched owner polls.
                check(two.twoshot_batched_runs == (two.twoshot_runs if want_batched else 0), f"B0 rows {rows} {kind}: batched counter")
                check(box_armed(), f"T2 rows {rows} {kind}: two-shot box not re-armed")
                dist.barrier()
                dr = int((r_one.view(torch.int16) != r_two.view(torch.int16)).sum())
                dn = int((n_one.view(torch.int16) != n_two.view(torch.int16)).sum())
                check(dr == 0 and dn == 0, f"T1 rows {rows} {kind}: {dr} residual / {dn} normed cells differ from one-shot")
                check(same_everywhere(r_two) and same_everywhere(n_two), f"T1 rows {rows} {kind}: ranks differ")
                if rank == 0:
                    print(f"T1 rows {rows} {kind}: residual diff {dr} normed diff {dn}", flush=True)

        # T3: graph replay of 10 chained two-shot sites at 64 rows
        rows = 64
        partial = randn(rows, HIDDEN, gen=own, scale=0.5).to(torch.bfloat16)
        projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
        residual = randn(rows, HIDDEN).to(torch.bfloat16)

        def ten(sites, part, proj, res):
            return [sites.run(i % 2, part, proj, conv, res, norm) for i in range(10)]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ten(two, partial, projection, residual)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = ten(two, partial, projection, residual)
        torch.cuda.synchronize()
        dist.barrier()
        for replay in range(3):
            new_partial = randn(rows, HIDDEN, gen=own, scale=0.5).to(torch.bfloat16)
            new_projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
            new_residual = randn(rows, HIDDEN).to(torch.bfloat16)
            partial.copy_(new_partial)
            projection.copy_(new_projection)
            residual.copy_(new_residual)
            graph.replay()
            torch.cuda.synchronize()
            got = [o.clone() for o in outs] + [residual.clone()]
            check(box_armed(), f"T2/T3 replay {replay}: two-shot box not re-armed")
            dist.barrier()
            r_eager = new_residual.clone()
            want = ten(two, new_partial, new_projection, r_eager) + [r_eager]
            torch.cuda.synchronize()
            dist.barrier()
            r_one = new_residual.clone()
            ref = ten(one, new_partial, new_projection, r_one) + [r_one]
            torch.cuda.synchronize()
            dist.barrier()
            d_eager = sum(int((g.view(torch.int16) != w.view(torch.int16)).sum()) for g, w in zip(got, want))
            d_one = sum(int((g.view(torch.int16) != w.view(torch.int16)).sum()) for g, w in zip(got, ref))
            check(d_eager == 0, f"T3 replay {replay}: {d_eager} cells differ from the eager two-shot sites")
            check(d_one == 0, f"T3 replay {replay}: {d_one} cells differ from the eager one-shot sites")

        # T4: NaN and -0.0 partials at 16 rows
        rows = 16
        projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
        res0 = randn(rows, HIDDEN).to(torch.bfloat16)
        partial = randn(rows, HIDDEN, gen=own, scale=0.5).to(torch.bfloat16)
        if rank == world - 1:
            partial[3, 1234] = float("nan")
        r_one, r_two = res0.clone(), res0.clone()
        n_one = run("one", partial, projection, r_one)
        dist.barrier()
        n_two = run("two", partial, projection, r_two)
        check(box_armed(), "T2/T4 NaN: two-shot box not re-armed")
        dist.barrier()
        check(
            torch.equal(torch.isnan(r_one), torch.isnan(r_two)) and torch.equal(torch.isnan(n_one), torch.isnan(n_two)),
            "T4 NaN masks differ from the one-shot site",
        )
        negative = projection.clone().view(rows, 2, TAPS, GROUPS)
        negative[:, 1, 1] = -negative[:, 1, 1].abs() - 1.0
        negative = negative.view(rows, -1).contiguous()
        partial = torch.full((rows, HIDDEN), -0.0, device=device, dtype=torch.bfloat16)
        r_one, r_two = res0.clone(), res0.clone()
        n_one = run("one", partial, negative, r_one)
        dist.barrier()
        n_two = run("two", partial, negative, r_two)
        check(box_armed(), "T2/T4 -0.0: two-shot box not re-armed")
        dist.barrier()
        check(bits_equal(r_one, r_two) and bits_equal(n_one, n_two), "T4 -0.0 partials: differs from the one-shot site")

        # T5: timing (informative)
        if not mutation and os.environ.get("TS_TIMING", "1") == "1":
            for rows in (16, 32, 64):
                part = randn(rows, HIDDEN, gen=own, scale=0.5).to(torch.bfloat16)
                proj = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
                res = randn(rows, HIDDEN).to(torch.bfloat16)
                graphs = {}
                for form, sites in (("one", one), ("two", two)):
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        [sites.run(i % 2, part, proj, conv, res, norm) for i in range(20)]
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()
                    dist.barrier()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=s):
                        [sites.run(i % 2, part, proj, conv, res, norm) for i in range(20)]
                    torch.cuda.synchronize()
                    dist.barrier()
                    graphs[form] = g
                med = {"one": [], "two": []}
                for form in ("one", "two", "two", "one"):
                    times = []
                    for i in range(35):
                        dist.barrier()
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        graphs[form].replay()
                        b.record()
                        b.synchronize()
                        if i >= 5:
                            times.append(a.elapsed_time(b) * 1000.0 / 20)
                    med[form].append(statistics.median(times))
                out = torch.tensor([med["one"][0], med["one"][1], med["two"][0], med["two"][1]], device=device)
                dist.all_reduce(out, op=dist.ReduceOp.MAX)
                if rank == 0:
                    o1, o2, t1, t2 = out.tolist()
                    print(
                        f"T5 rows {rows} TP {world}: us/site one-shot A1 {o1:.2f} A2 {o2:.2f} two-shot B1 {t1:.2f} B2 {t2:.2f} "
                        f"two - one {((t1 + t2) - (o1 + o2)) / 2:+.2f} (A/A {o2 - o1:+.2f}, B/B {t2 - t1:+.2f})",
                        flush=True,
                    )

    print(f"TP_RESULT rank {rank} failures={failures} checks={checks}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
