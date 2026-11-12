# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_K31_VP_GATHER=multimem / SGLANG_OPT_K31_VP_SPLIT on a real TP group (torchrun --nproc-per-node N, one
NVLink domain). Forms: payload transport nccl | multimem (the served triton_symm_mem_ag kernel on a fresh symmetric
buffer) x column split 1 | auto. Checks per rank, per round (bs 1 / 2 / 8, two logit kinds: FP32 + softcap 30, and
BF16 without softcap = served K3.1), stated before data:
  (a) target argmax == torch.argmax over the gathered (softcapped) logits, bitwise, every form;
  (b) every rank holds the same (accept result, target argmax), every form;
  (c) multimem == nccl at the same split, bitwise (the transport moves the same payload bytes);
  (c_skip, r2) multimem with the served entry rule (skip_entry_sync=True, own buffer, a cross-rank collective between
      consecutive calls as in serving) == nccl at split auto, bitwise;
  (d) split auto == split 1: target argmax bitwise; accept results equal (the split changes only the FP32 order of
      the softmax denominator; a coin within ~1e-6 relative of p could flip, expected 0 on these inputs).
Prints VPMM_CHECK lines and VPMM_OK / VPMM_FAIL per rank; exit != 0 on failure. K31_VPMM_MUT=stale makes the multimem
gather skip its kernel after the first call (the reader sees a stale buffer): (a)-(c) must fail.
Then times (single-node proxy, TP = this world, not TP16) each form against the served exchange it replaces: the multimem
gather of the BF16 verify-logit shard (served gatherer form, entry sync skipped) + torch.argmax over the gathered rows.
Every op is captured in a CUDA graph (20 calls), replayed in ABBA order on the same GPUs; per-call p50 / p10 / p90.
"""

import os
import sys

import torch
import torch.distributed as dist

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.kernels.ops.speculative import vocab_parallel_accept as VP
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as AG


class _Group:
    def __init__(self, rank, world):
        self.world_size, self.rank_in_group = world, rank

    def all_gather_into_tensor(self, out, inp):
        dist.all_gather_into_tensor(out, inp)


def _case(bs, block, top_k, vocab, world, seed, dev, dtype, softcap):
    g = torch.Generator(device=dev).manual_seed(seed)
    shard = -(-vocab // world)
    raw = torch.randn((bs * block, world * shard), generator=g, device=dev) * 3.0
    raw[:, vocab:] = 1.0e4
    raw[0, 100] = raw[0, vocab - 5] = 40.0  # cross-shard tie
    raw = raw.to(dtype)
    gamma = block - 1
    cid = torch.rand((bs * gamma, vocab), generator=g, device=dev).topk(top_k, -1).indices.view(bs, gamma, top_k)
    q = torch.rand((bs, gamma, top_k), generator=g, device=dev)
    q = (q / q.sum(-1, keepdim=True) * 0.95).contiguous()
    pick = torch.randint(0, top_k, (bs, gamma), generator=g, device=dev)
    drafts = cid.gather(-1, pick[..., None]).squeeze(-1)
    # drafts get real target mass so the walk accepts often (boundaries exercised)
    rows = (torch.arange(bs, device=dev)[:, None] * block + torch.arange(gamma, device=dev)[None, :]).reshape(-1)
    raw[rows, drafts.reshape(-1)] = (raw[rows, :vocab].float().max(-1).values - 0.5).to(dtype)
    cand = torch.cat([torch.randint(0, vocab, (bs, 1), generator=g, device=dev), drafts], 1).contiguous()
    full = torch.empty((bs * block, vocab), dtype=torch.float32, device=dev)
    shards = raw.float().view(bs * block, world, shard).permute(1, 0, 2).contiguous()
    gathered_softcap_logits(shards, softcap, full)
    return raw, cid.contiguous(), q, cand, shard, full.argmax(-1).view(bs, block)


def _mm_state(world, rank, max_rows, width_fp32):
    st = AG.create_state(group=dist.group.WORLD, rank_in_group=rank, max_tokens=max_rows, hidden_size=2 * width_fp32 * world)
    return st if st.symm_mem_hdl.multicast_ptr != 0 else None


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    mut = os.environ.get("K31_VPMM_MUT", "")
    vocab, block, top_k = 163840, 8, 16
    grp = _Group(rank, world)
    width_max = VP._MAX_SPLITS * VP.payload_width(top_k)
    st = _mm_state(world, rank, 128, width_max)
    if st is None:
        print(f"VPMM_NO_MULTICAST rank={rank}", flush=True)
        sys.exit(3)
    # The skip-entry gatherer gets its OWN symmetric buffer (as in serving, where VP's buffer is not shared),
    # so a stale skip-entry read cannot pass by returning the entry-sync form's bytes for the same inputs.
    st_skip = _mm_state(world, rank, 128, width_max)
    if st_skip is None:
        print(f"VPMM_NO_MULTICAST rank={rank}", flush=True)
        sys.exit(3)
    gat = VP.VpPayloadGather(st, grp, skip_entry_sync=False)
    gat_skip = VP.VpPayloadGather(st_skip, grp, skip_entry_sync=True)
    if mut == "stale":

        def _stale(g, buf, skip):
            real = g.__call__
            seen = {"n": 0}

            class _Stale(VP.VpPayloadGather):
                def __call__(self, local):
                    seen["n"] += 1
                    if seen["n"] == 1:
                        return real(local)
                    rows, width = local.shape
                    return buf.comm_buff[:rows, : 2 * width * world].view(torch.float32), True

            return _Stale(buf, grp, skip_entry_sync=skip)

        gat, gat_skip = _stale(gat, st, False), _stale(gat_skip, st_skip, True)
        print("K31_MUTATION vpmm_stale applied", flush=True)
    # ("multimem_skip", 0): the served gatherer's entry rule (skip_entry_sync=True) with the served spacing (a cross-rank
    # NCCL collective between consecutive calls, as the TP forward provides between two verify rounds).
    forms = (("nccl", 1), ("nccl", 0), ("multimem", 1), ("multimem", 0), ("multimem_skip", 0))
    gathers = {"nccl": None, "multimem": gat, "multimem_skip": gat_skip}
    bad = 0
    n_checks = 0
    for kind, dtype, softcap in (("f32sc", torch.float32, 30.0), ("bf16", torch.bfloat16, None)):
        for bs in (1, 2, 8):
            raw, cid, q, cand, shard, ref_tp = _case(bs, block, top_k, vocab, world, 100 + bs, dev, dtype, softcap)
            temps = torch.ones((bs, 1), device=dev)
            sl = raw[:, rank * shard : (rank + 1) * shard].contiguous()
            e = {"a": 0, "b": 0, "c": 0, "c_skip": 0, "d_tp": 0, "d_res": 0}
            for it in range(10):
                u = torch.rand((bs, block - 1), device=dev, generator=torch.Generator(device=dev).manual_seed(it))
                uf = torch.rand((bs,), device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + it))
                out = {}
                for tr, sp in forms:
                    res, tp = VP.vp_selector_accept(
                        logits_shard=sl, group=grp, vocab_size=vocab, softcap=softcap, candidates=cand,
                        candidate_ids=cid, q_rows=q, uniform_samples=u, uniform_samples_final=uf, temperatures=temps,
                        gather=gathers[tr], split=sp,
                    )  # fmt: skip
                    out[(tr, sp)] = (res.clone(), tp.clone())
                    e["a"] += int((tp != ref_tp).sum())
                    mine = torch.cat([res.view(-1), tp.view(-1)])
                    allr = torch.empty((world, mine.numel()), dtype=mine.dtype, device=dev)
                    dist.all_gather_into_tensor(allr, mine)
                    e["b"] += int((allr != allr[0:1]).any(dim=1).sum())
                for sp in (1, 0):
                    (r0, t0), (r1, t1) = out[("nccl", sp)], out[("multimem", sp)]
                    e["c"] += int((r0 != r1).sum() + (t0 != t1).sum())
                (r0, t0), (r1, t1) = out[("nccl", 0)], out[("multimem_skip", 0)]
                e["c_skip"] += int((r0 != r1).sum() + (t0 != t1).sum())
                for tr in ("nccl", "multimem"):
                    (r0, t0), (r1, t1) = out[(tr, 1)], out[(tr, 0)]
                    e["d_tp"] += int((t0 != t1).sum())
                    e["d_res"] += int((r0 != r1).any(-1).sum())
            for key, val in e.items():
                n_checks += 1
                ok = val == 0
                bad += 0 if ok else 1
                print(f"VPMM_CHECK rank={rank} kind={kind} bs={bs} check={key} errors={val} ok={int(ok)}", flush=True)
    print(f"{'VPMM_OK' if bad == 0 else 'VPMM_FAIL'} rank={rank} checks={n_checks} bad={bad}", flush=True)
    if mut:
        _finish(rank, 0 if bad == 0 else 1)

    # ---- timing (single-node proxy): graph-replayed, ABBA over forms, same GPUs ----
    # the served verify-logit gatherer's form: its own symmetric buffer of vocab width, entry sync skipped
    served_st = AG.create_state(group=dist.group.WORLD, rank_in_group=rank, max_tokens=128, hidden_size=world * (-(-vocab // world)))
    n_in_graph, reps = 20, 30
    for bs in (1, 2, 8):
        rows = bs * block
        raw, cid, q, cand, shard, _ = _case(bs, block, top_k, vocab, world, 7 + bs, dev, torch.bfloat16, None)
        temps = torch.ones((bs, 1), device=dev)
        sl = raw[:, rank * shard : (rank + 1) * shard].contiguous()
        u = torch.rand((bs, block - 1), device=dev)
        uf = torch.rand((bs,), device=dev)

        def served():
            full = AG.all_gather_inner(served_st, sl, tp_hidden_dim=sl.shape[1] * world, skip_entry_sync=True, safe=False)
            return full.argmax(-1)

        def vp(tr, sp, g):
            return VP.vp_selector_accept(
                logits_shard=sl, group=grp, vocab_size=vocab, softcap=None, candidates=cand, candidate_ids=cid,
                q_rows=q, uniform_samples=u, uniform_samples_final=uf, temperatures=temps,
                gather=g if tr == "multimem" else None, split=sp,
            )  # fmt: skip

        ops = {
            "served_mm_gather_argmax": served,
            "vp_nccl_s1": lambda: vp("nccl", 1, None),
            "vp_nccl_auto": lambda: vp("nccl", 0, None),
            "vp_mm_s1": lambda: vp("multimem", 1, gat),
            "vp_mm_auto": lambda: vp("multimem", 0, gat),
            "vp_mm_auto_skipentry": lambda: vp("multimem", 0, gat_skip),
        }
        graphs = {}
        for name, fn in ops.items():
            for _ in range(3):  # eager warm-up (Triton compiles)
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            dist.barrier()
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                for _ in range(n_in_graph):
                    fn()
            torch.cuda.synchronize()
            dist.barrier()
            graphs[name] = gph
        names = list(ops)
        order = (names + names[::-1]) * (reps // 2)
        ts = {n: [] for n in names}
        for i, name in enumerate(order):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[name].replay()
            b.record()
            b.synchronize()
            ts[name].append(a.elapsed_time(b) * 1000.0 / n_in_graph)
        for name in names:
            v = sorted(ts[name][2:])
            tmax = torch.tensor(ts[name][2:], device=dev)  # per replay, then the slowest rank
            dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
            if rank == 0:
                vm = sorted(tmax.tolist())
                print(f"VPMM_US world={world} bs={bs} rows={rows} form={name} p50={v[len(v) // 2]:.2f} "
                      f"p10={v[len(v) // 10]:.2f} p90={v[(9 * len(v)) // 10]:.2f} maxrank_p50={vm[len(vm) // 2]:.2f} n={len(v)}",
                      flush=True)  # fmt: skip
    del graphs
    _finish(rank, 0 if bad == 0 else 1)


def _finish(rank, code):
    """Teardown: destroy_process_group can hang after every check and timing line has printed. Serving never destroys
    its process group, so the teardown is harness-only: sync, one barrier, try destroy_process_group for at most 60 s on
    a daemon thread (the outcome is printed as a diagnostic, not a gate), then exit with the real verdict via os._exit."""
    import threading

    torch.cuda.synchronize()
    dist.barrier()
    print(f"VPMM_TIMING_DONE rank={rank}", flush=True)
    th = threading.Thread(target=dist.destroy_process_group, daemon=True)
    th.start()
    th.join(60)
    print(f"VPMM_DESTROY rank={rank} done={int(not th.is_alive())}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
