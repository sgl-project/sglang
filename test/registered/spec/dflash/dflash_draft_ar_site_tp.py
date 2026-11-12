# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""torchrun check: the DFlash draft all-reduce sites (SGLANG_OPT_DFLASH_DRAFT_AR_SITE, speculative/dflash_draft_ar_site.py)
and their wiring in models/dflash.py, on any TP >= 2 (multicast memory required).

    torchrun --nproc-per-node 4 test/registered/spec/dflash/dflash_draft_ar_site_tp.py

Checks; the tolerances are fixed here, before any run of this file:
  A  one site vs the chain it replaces (FP32 rank-order all-reduce rounded once -> dflash_grouped_conv side 1 ->
     sgl_kernel fused add-RMSNorm), both against an FP64 reference, rows 8 / 16 / 32 / 64: the site's relative
     Frobenius error of the new residual and of the normed row is at most 1.03x the chain's (the site rounds each
     rank's convolved partial before the sum: same error size, different rounding points).
  B  every rank holds the same bits (new residual and normed row).
  C  every mailbox word is the sentinel again after each call. The read must come before any peer can start the next
     site (a peer one site ahead legitimately pushes into this rank's slots), so a collective follows every read that
     is not already followed by one before the next site.
  D  a CUDA graph of 10 sites (alternating parities, residual chained) replayed with new inputs equals the same 10
     eager calls bitwise.
  E  a NaN partial element on the last rank: the site finishes and its NaN masks equal the chain's; every partial
     -0.0 (convolved halves -0.0, the slots' sentinel half): it finishes, its new residual equals the chain's up to the
     sign of zero and its normed row is within relF 2^-8 of the chain's (two norm kernels on the same input).
  F  wiring: a 3-layer DFlash stack (real grouped convs and RMSNorms; stub attention / MLP whose row-parallel partials
     are per-rank column scalings) through DFlashDraftModel.forward with and without the sites, each against an FP64
     reference of the stack: relF(sites) <= 1.25x relF(stock), every rank the same bits; at 72 rows (above the sites'
     64) the sites stay out and the output equals the stock path bitwise.
  G  SGLANG_OPT_DFLASH_DRAFT_AR_SITE_EARLY_TRIGGER, on the F stack with split-K conv projections: (G1) the early
     release is chosen only with the knob on AND a next conv whose split-K projection covers the site's output (not
     without a next conv, nor for a stock projection, nor at rows or a base address the projection does not cover); (G2) at 8 rows the
     stack's output with the early release equals the output without it bitwise, every rank the same bits, the
     mailbox re-armed (the release carries no data: the projection reads the site's output after its own wait);
     (G3) the projection must itself trigger only after its wait (more K tiles per split than stages): every served
     conv-projection tactic (<= 8 rows and the WIDE rows) meets it, and a covering projection with 7 tiles per split
     gets the early release with 6 stages but not with 7.
Prints FAIL lines and one TP_RESULT line per rank; exits nonzero on any failure.
"""

import functools
import os
import sys
import types

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

HIDDEN, TAPS, GROUP, BLOCK, EPS = 7168, 2, 16, 8, 1e-6
GROUPS = HIDDEN // GROUP
SITE_RATIO, STACK_RATIO = 1.03, 1.25
NORM_ONLY_RELF = 2.0**-8  # two RMSNorm kernels on the same input differ by output roundings only
DEADLINE_NS = 20_000_000_000  # a test-only bound: ranks reach a site seconds apart at most


def relf(y, ref):
    return float((y.double() - ref).norm() / ref.norm())


def main() -> int:
    local_rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if os.environ.get("K31_MUTATION"):
        print(f"K31_MUTATION {os.environ['K31_MUTATION']} applied", flush=True)

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashDraftModel, DFlashGroupedConv
    from sglang.srt.speculative import dflash_draft_ar_site as site_mod

    group = types.SimpleNamespace(world_size=world, rank_in_group=rank, device_group=dist.group.WORLD)
    sites = site_mod.DraftArSites(group, device, DEADLINE_NS)
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

    def armed():
        torch.cuda.synchronize()
        return bool((sites.mailbox.buffer.view(torch.int32) == site_mod.SENTINEL).all())

    def reduce_rank_order(p):
        parts = gather(p)
        acc = parts[0].float()
        for q in parts[1:]:
            acc = acc + q.float()
        return acc.to(torch.bfloat16)

    shared = torch.Generator(device=device).manual_seed(17)  # replicated tensors: the same on every rank
    own = torch.Generator(device=device).manual_seed(1000 + rank)  # this rank's partials

    def randn(*shape, gen=shared, scale=1.0):
        return torch.randn(*shape, generator=gen, device=device) * scale

    def make_conv():
        conv = DFlashGroupedConv(HIDDEN, BLOCK, TAPS, GROUP).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            base = randn(2, TAPS, HIDDEN, scale=0.3)
            base[:, 0] += 1.0
            conv.base_kernel.copy_(base)
            conv.kernel_projection.weight.copy_(randn(2 * TAPS * GROUPS, HIDDEN, scale=0.003))
        return conv

    def make_norm():
        norm = RMSNorm(HIDDEN, eps=EPS).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1.0 + randn(HIDDEN, scale=0.1))
        return norm

    def conv64(conv, x, projection, side):
        """FP64 DFlash grouped conv: out[t] = sum_tap (base[side, tap] + delta[t, tap]) * x[t - tap], within a block."""
        rows = x.shape[0]
        delta = projection.double().view(rows, 2, TAPS, GROUPS)[:, side].repeat_interleave(GROUP, dim=2)
        coef = conv.base_kernel.double()[side].unsqueeze(0) + delta  # [rows, taps, hidden]
        prev = torch.zeros_like(x)
        prev[1:] = x[:-1]
        prev[torch.arange(rows, device=device) % BLOCK == 0] = 0
        return coef[:, 0] * x + coef[:, 1] * prev

    def norm64(norm, r):
        return r * torch.rsqrt((r * r).mean(-1, keepdim=True) + EPS) * norm.weight.double()

    parity = [0]

    def site(partial, projection, conv, residual, norm):
        p = parity[0]
        parity[0] ^= 1
        return sites.run(p, partial, projection, conv, residual, norm)

    def chain(partial, projection, conv, residual, norm):
        rows = partial.shape[0]
        y = conv.finish(reduce_rank_order(partial), projection.view(rows, 2, TAPS, GROUPS)[:, 1])
        out, _ = norm(y, residual)  # residual += y in place; out = norm(residual)
        return out

    conv, norm = make_conv(), make_norm()
    with torch.inference_mode():
        # A, B, C
        for rows in (8, 16, 32, 64):
            site_mod.compile_site(rows, world, BLOCK, DEADLINE_NS)
            dist.barrier()
            for kind, scale in (("normal", 0.5), ("large", 8.0)):
                partial = randn(rows, HIDDEN, gen=own, scale=scale).to(torch.bfloat16)
                projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
                res0 = randn(rows, HIDDEN).to(torch.bfloat16)
                r_site = res0.clone()
                n_site = site(partial, projection, conv, r_site, norm)
                check(armed(), f"C rows {rows} {kind}: mailbox not re-armed")
                r_chain = res0.clone()
                n_chain = chain(partial, projection, conv, r_chain, norm)
                x64 = sum(p.double() for p in gather(partial))
                r64 = res0.double() + conv64(conv, x64, projection, 1)
                n64 = norm64(norm, r64)
                e = [relf(r_site, r64), relf(r_chain, r64), relf(n_site, n64), relf(n_chain, n64)]
                check(e[0] <= SITE_RATIO * e[1], f"A rows {rows} {kind}: residual relF {e[0]:.4e} > {SITE_RATIO} x chain {e[1]:.4e}")
                check(e[2] <= SITE_RATIO * e[3], f"A rows {rows} {kind}: norm relF {e[2]:.4e} > {SITE_RATIO} x chain {e[3]:.4e}")
                check(same_everywhere(r_site) and same_everywhere(n_site), f"B rows {rows} {kind}: ranks differ")
                if rank == 0:
                    print(f"A rows {rows} {kind}: relF residual site {e[0]:.4e} chain {e[1]:.4e}; norm site {e[2]:.4e} chain {e[3]:.4e}", flush=True)

        # D: graph replay == eager, 10 chained sites at 16 rows
        rows = 16
        partial = randn(rows, HIDDEN, gen=own, scale=0.5).to(torch.bfloat16)
        projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
        residual = randn(rows, HIDDEN).to(torch.bfloat16)

        def ten():
            return [sites.run(i % 2, partial, projection, conv, residual, norm) for i in range(10)]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ten()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = ten()
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
            check(armed(), f"D replay {replay}: mailbox not re-armed")
            dist.barrier()  # no peer pushes its next (eager) site into this rank's slots before every rank has read
            r_eager = new_residual.clone()
            want = [sites.run(i % 2, new_partial, new_projection, conv, r_eager, norm) for i in range(10)] + [r_eager]
            diff = sum(int((g.view(torch.int16) != w.view(torch.int16)).sum()) for g, w in zip(got, want))
            check(diff == 0, f"D replay {replay}: {diff} cells differ from the eager sites")

        # E: NaN and -0.0 partials at 8 rows
        rows = 8
        projection = randn(rows, 2 * TAPS * GROUPS, scale=0.3).to(torch.bfloat16)
        res0 = randn(rows, HIDDEN).to(torch.bfloat16)
        partial = randn(rows, HIDDEN, gen=own, scale=0.5).to(torch.bfloat16)
        if rank == world - 1:
            partial[3, 1234] = float("nan")
        r_site, r_chain = res0.clone(), res0.clone()
        n_site = site(partial, projection, conv, r_site, norm)
        n_chain = chain(partial, projection, conv, r_chain, norm)
        check(
            torch.equal(torch.isnan(r_site), torch.isnan(r_chain)) and torch.equal(torch.isnan(n_site), torch.isnan(n_chain)),
            "E NaN masks differ from the chain",
        )
        check(armed(), "E NaN: mailbox not re-armed")
        dist.barrier()  # as in D: the next site follows without a collective
        negative = projection.clone().view(rows, 2, TAPS, GROUPS)
        negative[:, 1, 1] = -negative[:, 1, 1].abs() - 1.0  # tap-1 coefficients below zero: -0.0 products
        negative = negative.view(rows, -1).contiguous()
        partial = torch.full((rows, HIDDEN), -0.0, device=device, dtype=torch.bfloat16)
        r_site, r_chain = res0.clone(), res0.clone()
        n_site = site(partial, negative, conv, r_site, norm)
        n_chain = chain(partial, negative, conv, r_chain, norm)
        check(torch.equal(r_site.float(), r_chain.float()), "E -0.0 partials: residual differs beyond the sign of zero")
        check(relf(n_site, n_chain.double()) <= NORM_ONLY_RELF, "E -0.0 partials: normed row off the chain's")
        check(armed(), "E -0.0: mailbox not re-armed")

        # F: wiring through DFlashDraftModel.forward
        class Projection:
            def __init__(self):
                self.reduce_results = True

        class StubAttention(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale, self.o_proj = scale, Projection()

            def forward(self, positions, hidden_states, forward_batch):
                partial = (hidden_states.float() * self.scale.float()).to(torch.bfloat16)
                return reduce_rank_order(partial) if self.o_proj.reduce_results else partial

        class StubMLP(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale, self.down_proj = scale, Projection()

            def forward(self, x):
                partial = (x.float() * self.scale.float()).to(torch.bfloat16)
                return reduce_rank_order(partial) if self.down_proj.reduce_results else partial

        def rank_scales():
            scales = [(0.5 + torch.rand(HIDDEN, generator=shared, device=device)) / world for _ in range(world)]
            return [s.to(torch.bfloat16) for s in scales]

        layers = []
        for _ in range(3):
            layer = DFlashDecoderLayer.__new__(DFlashDecoderLayer)
            nn.Module.__init__(layer)
            attention_scales, mlp_scales = rank_scales(), rank_scales()
            layer.input_layernorm, layer.post_attention_layernorm = make_norm(), make_norm()
            layer.attention_conv, layer.mlp_conv = make_conv(), make_conv()
            layer.self_attn, layer.mlp = StubAttention(attention_scales[rank]), StubMLP(mlp_scales[rank])
            layer.scales64 = (
                sum(s.double() for s in attention_scales),
                sum(s.double() for s in mlp_scales),
            )
            layers.append(layer)
        model = types.SimpleNamespace(layers=layers, norm=make_norm(), ar_sites=None)
        model._forward_ar_sites = functools.partial(DFlashDraftModel._forward_ar_sites, model)

        def forward(x, with_sites):
            model.ar_sites = sites if with_sites else None
            return DFlashDraftModel.forward(model, None, None, None, input_embeds=x.clone()).hidden_states

        def stack64(x):
            res = x.double()
            h = norm64(layers[0].input_layernorm, res)
            next_norms = [layer.input_layernorm for layer in layers[1:]] + [model.norm]
            for layer, next_norm in zip(layers, next_norms):
                for conv_, scale, after in (
                    (layer.attention_conv, layer.scales64[0], layer.post_attention_layernorm),
                    (layer.mlp_conv, layer.scales64[1], next_norm),
                ):
                    projection = h @ conv_.kernel_projection.weight.double().T
                    h = conv64(conv_, h, projection, 0)
                    res = res + conv64(conv_, h * scale, projection, 1)
                    h = norm64(after, res)
            return h

        for rows in (8, 24, 64):
            x = randn(rows, HIDDEN).to(torch.bfloat16)
            ref = stack64(x)
            stock, with_sites = forward(x, False), forward(x, True)
            e_stock, e_sites = relf(stock, ref), relf(with_sites, ref)
            check(e_sites <= STACK_RATIO * e_stock, f"F rows {rows}: relF {e_sites:.4e} > {STACK_RATIO} x stock {e_stock:.4e}")
            check(same_everywhere(with_sites), f"F rows {rows}: ranks differ")
            check(all(l.self_attn.o_proj.reduce_results and l.mlp.down_proj.reduce_results for l in layers), f"F rows {rows}: reduce_results not restored")
            check(armed(), f"F rows {rows}: mailbox not re-armed")
            if rank == 0:
                print(f"F rows {rows}: relF sites {e_sites:.4e} stock {e_stock:.4e}", flush=True)
        x = randn(72, HIDDEN).to(torch.bfloat16)
        stock, with_sites = forward(x, False), forward(x, True)
        check(torch.equal(stock.view(torch.int16), with_sites.view(torch.int16)), "F rows 72: not the stock path")

        from sglang.srt.speculative.dflash_draft_splitk import DraftSplitKProjection

        for layer in layers:
            for conv_ in (layer.attention_conv, layer.mlp_conv):
                conv_.split_k_projection = DraftSplitKProjection(conv_.kernel_projection.weight.detach(), (128, 8, 7, 5))
        covered, bare = layers[0].mlp_conv, make_conv()
        for rows, successor, label, knob, want in (
            (8, covered, "split-K", False, False),
            (8, None, "none", True, False),
            (8, bare, "stock projection", True, False),
            (24, covered, "split-K", True, False),
            (8, covered, "split-K", True, True),
        ):
            sites.early_trigger = knob
            got = sites._early_for(successor, torch.empty(rows, HIDDEN, device=device, dtype=torch.bfloat16))
            check(got == want, f"G1 rows {rows}, successor {label}, knob {knob}: early {got}, want {want}")
        sites.early_trigger = True
        misaligned = torch.empty(8 * HIDDEN + 1, device=device, dtype=torch.bfloat16)[1:].view(8, HIDDEN)  # TMA needs 16 B
        got = sites._early_for(covered, misaligned)
        check(not got, f"G1 rows 8, split-K, knob True, output base not 16-byte aligned: early {got}, want False")
        from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS, WIDE_TACTICS

        def shallow(policies):  # the real class and methods, no kernel build: only the tactic table matters here
            p = object.__new__(DraftSplitKProjection)
            p.k, p.rows, p._policies = HIDDEN, frozenset(policies), dict(policies)
            # the row set the constructor sets; these tables have no SK3B rows
            p._sk3b_rows = frozenset()
            return p

        conv_tactics = {**{r: SPLITK_TACTICS[(1792, HIDDEN)] for r in range(1, 9)}, **WIDE_TACTICS[(1792, HIDDEN)]}
        served = shallow(conv_tactics)
        for rows in sorted(conv_tactics):
            check(served.trigger_follows_wait(rows), f"G3 rows {rows}: served conv tactic {conv_tactics[rows]} lets the projection trigger before its wait")
        sites.early_trigger = True
        for stages, want in ((7, False), (6, True)):  # 7168 / (128 x 8) = 7 K tiles per split
            succ = types.SimpleNamespace(split_k_projection=shallow({8: (64, 8, 8, stages)}))
            got = sites._early_for(succ, torch.empty(8, HIDDEN, device=device, dtype=torch.bfloat16))
            check(got == want, f"G3 rows 8, 7 tiles per split, {stages} stages: early {got}, want {want}")
        x = randn(8, HIDDEN).to(torch.bfloat16)
        sites.early_trigger = False
        late = forward(x, True)
        sites.early_trigger = True
        early = forward(x, True)
        sites.early_trigger = False
        check(torch.equal(late.view(torch.int16), early.view(torch.int16)), "G2 rows 8: the early release changed the output")
        check(same_everywhere(early), "G2 rows 8: ranks differ")
        check(armed(), "G2: mailbox not re-armed")

    dist.barrier()
    print(f"[rank{rank}] TP_RESULT world={world} checks={checks} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
