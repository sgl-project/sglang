# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_K31_VP_ACCEPT on a real TP group (torchrun --nproc-per-node N): every rank reduces its own vocab shard,
the payloads go through torch.distributed.all_gather_into_tensor, and every rank must get (a) the same accept result
and target argmax as every other rank, (b) bit for bit the single-process emulation that stacks all shards, and
(c) torch.argmax over the gathered softcapped logits. K31_VP_TP_MUT=rank0 makes every rank claim rank 0's column
offset (the shard-offset bug in a real group): (b) and (c) must fail. Cross-rank guard: in
extra rounds the LAST rank perturbs one request's seed (one ulp of uniform_samples_final) or one candidate id; every
rank's mismatch counter must read exactly (1, 0) / (0, 1), and (0, 0) on every unperturbed round. Sync fold: rank 0
writes its values, the others zeros, dist.all_reduce sums them, and commit_from_fp32_sync must equal the served
commit_selector_block on every rank's own int64 result (VP_TP_FOLD). Prints VP_TP_OK / VP_TP_FAIL per rank and exits
non-zero on failure. Also times (proxy, plain NCCL in a 4-GPU world, not the served collective) the payload all-gather
vs the full-logit shard all-gather at the same rows.
"""

import os
import sys

import torch
import torch.distributed as dist

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.kernels.ops.speculative.selector_sampled_accept import commit_selector_block
from sglang.kernels.ops.speculative.vocab_parallel_accept import (
    commit_from_fp32_sync,
    vp_combine,
    vp_local_payload,
    vp_selector_accept,
)


class _Group:
    def __init__(self, rank, world, mut):
        self.world_size, self.rank_in_group = world, (0 if mut == "rank0" else rank)

    def all_gather_into_tensor(self, out, inp):
        dist.all_gather_into_tensor(out, inp)


def _case(bs, block, top_k, vocab, world, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    shard = -(-vocab // world)
    raw = torch.randn((bs * block, world * shard), generator=g, device=dev) * 3.0
    raw[:, vocab:] = 1.0e4
    raw[0, 100] = raw[0, vocab - 5] = 40.0  # cross-shard tie
    gamma = block - 1
    cid = torch.rand((bs * gamma, vocab), generator=g, device=dev).topk(top_k, -1).indices.view(bs, gamma, top_k)
    q = torch.rand((bs, gamma, top_k), generator=g, device=dev)
    q = (q / q.sum(-1, keepdim=True) * 0.95).contiguous()
    pick = torch.randint(0, top_k, (bs, gamma), generator=g, device=dev)
    drafts = cid.gather(-1, pick[..., None]).squeeze(-1)
    cand = torch.cat([torch.randint(0, vocab, (bs, 1), generator=g, device=dev), drafts], 1).contiguous()
    return raw, cid.contiguous(), q, cand, shard


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    mut = os.environ.get("K31_VP_TP_MUT", "")
    if mut:
        print(f"K31_MUTATION tp_{mut} applied", flush=True)
    grp = _Group(rank, world, mut)
    vocab, block, top_k, softcap = 163840, 8, 16, 30.0
    bad = 0
    for bs in (1, 2, 8):
        raw, cid, q, cand, shard = _case(bs, block, top_k, vocab, world, 100 + bs, dev)
        temps = torch.ones((bs, 1), device=dev)
        rows = bs * block
        full = torch.empty((rows, vocab), dtype=torch.float32, device=dev)
        gathered_softcap_logits(raw.view(rows, world, shard).permute(1, 0, 2).contiguous(), softcap, full)
        ref_tp = full.argmax(-1).view(bs, block)
        for it in range(20):
            u = torch.rand((bs, block - 1), device=dev, generator=torch.Generator(device=dev).manual_seed(it))
            uf = torch.rand((bs,), device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + it))
            cnt = torch.zeros((2,), dtype=torch.int32, device=dev)
            res, tp = vp_selector_accept(
                logits_shard=raw[:, rank * shard : (rank + 1) * shard].contiguous(), group=grp, vocab_size=vocab,
                softcap=softcap, candidates=cand, candidate_ids=cid, q_rows=q, uniform_samples=u,
                uniform_samples_final=uf, temperatures=temps, mismatch_counter=cnt,
            )  # fmt: skip
            pays = [
                vp_local_payload(
                    logits_shard=raw[:, w * shard : (w + 1) * shard].contiguous(), temperatures=temps,
                    uniform_samples_final=uf, candidate_ids=cid, rows_per_request=block,
                    valid_width=max(0, min(shard, vocab - w * shard)), col_offset=w * shard, vocab_size=vocab,
                    softcap=softcap, candidates=cand, q_rows=q, uniform_samples=u,
                )
                for w in range(world)
            ]  # fmt: skip
            res_e, tp_e = vp_combine(
                gathered=torch.stack(pays).contiguous(), candidates=cand, candidate_ids=cid, q_rows=q,
                uniform_samples=u, uniform_samples_final=uf, temperatures=temps, vocab_size=vocab, shard_width=shard,
            )  # fmt: skip
            mine = torch.cat([res.view(-1), tp.view(-1)])
            allr = torch.empty((world, mine.numel()), dtype=mine.dtype, device=dev)
            dist.all_gather_into_tensor(allr, mine)
            e_cross = int((allr != allr[0:1]).any(dim=1).sum())
            e_emul = int((res != res_e).sum() + (tp != tp_e).sum())
            e_ref = int((tp != ref_tp).sum())
            e_cnt = int(cnt.ne(0).sum())
            if e_cross or e_emul or e_ref or e_cnt:
                bad += 1
                if it < 3:
                    print(f"rank {rank} bs={bs} it={it} cross={e_cross} emul={e_emul} ref={e_ref} cnt={cnt.tolist()}", flush=True)
        # sync fold on a real all-reduce (round 0's uniforms)
        u = torch.rand((bs, block - 1), device=dev, generator=torch.Generator(device=dev).manual_seed(0))
        uf = torch.rand((bs,), device=dev, generator=torch.Generator(device=dev).manual_seed(1000))
        sync = torch.full(((2 * bs + 3) // 4 * 4,), float("nan"), device=dev)
        res, _ = vp_selector_accept(
            logits_shard=raw[:, rank * shard : (rank + 1) * shard].contiguous(), group=grp, vocab_size=vocab,
            softcap=softcap, candidates=cand, candidate_ids=cid, q_rows=q, uniform_samples=u,
            uniform_samples_final=uf, temperatures=temps, sync_out=sync, sync_values=(rank == 0),
        )  # fmt: skip
        dist.all_reduce(sync)
        prefix = torch.arange(50, 50 + bs, device=dev, dtype=torch.int64)
        got = commit_from_fp32_sync(candidates=cand, summed=sync, prefix_lens=prefix)
        ref = commit_selector_block(candidates=cand, accept_result=res.contiguous(), prefix_lens=prefix)
        fold_ok = all(torch.equal(a, b) for a, b in zip(got, ref))
        print(f"VP_TP_FOLD rank={rank} bs={bs} ok={int(fold_ok)}", flush=True)
        if not fold_ok:
            bad += 1
        # cross-rank guard: the last rank alone perturbs request bs - 1 (seed) or row (bs - 1, 2)'s first candidate id.
        for field, want in (("seed", [1, 0]), ("cid", [0, 1])):
            u = torch.rand((bs, block - 1), device=dev, generator=torch.Generator(device=dev).manual_seed(77))
            uf = torch.rand((bs,), device=dev, generator=torch.Generator(device=dev).manual_seed(78))
            uf_r, cid_r = uf.clone(), cid.clone()
            if rank == world - 1:
                if field == "seed":
                    uf_r[bs - 1] = torch.nextafter(uf_r[bs - 1], torch.tensor(2.0, device=dev))
                else:
                    cid_r[bs - 1, 2, 0] = (cid_r[bs - 1, 2, 0] + 1) % vocab
            cnt = torch.zeros((2,), dtype=torch.int32, device=dev)
            vp_selector_accept(
                logits_shard=raw[:, rank * shard : (rank + 1) * shard].contiguous(), group=grp, vocab_size=vocab,
                softcap=softcap, candidates=cand, candidate_ids=cid_r, q_rows=q, uniform_samples=u,
                uniform_samples_final=uf_r, temperatures=temps, mismatch_counter=cnt,
            )  # fmt: skip
            got = cnt.tolist()
            print(f"VP_TP_GUARD rank={rank} bs={bs} field={field} counter={got} want={want}", flush=True)
            if got != want:
                bad += 1
        print(f"VP_TP rank={rank} world={world} bs={bs} rounds=20+2 bad_rounds_so_far={bad}", flush=True)
    # timing proxy: payload all-gather vs shard all-gather (fp32), same rows, median of 200
    for bs in (1, 2, 8):
        rows = bs * block
        for name, width in (("payload", 8 + top_k), ("shard", -(-vocab // world))):
            x = torch.randn((rows, width), device=dev)
            out = torch.empty((world * rows, width), device=dev)
            ts = []
            for i in range(230):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                dist.barrier()
                a.record()
                dist.all_gather_into_tensor(out, x)
                b.record()
                b.synchronize()
                if i >= 30:
                    ts.append(a.elapsed_time(b) * 1000.0)
            ts.sort()
            if rank == 0:
                print(f"VP_TP_GATHER_US world={world} bs={bs} {name} width={width} p50={ts[len(ts) // 2]:.2f} p10={ts[len(ts) // 10]:.2f}", flush=True)
    print(f"{'VP_TP_OK' if bad == 0 else 'VP_TP_FAIL'} rank={rank} bad_rounds={bad}", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
