# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Single-GPU cost of the parts of SGLANG_OPT_K31_VP_ACCEPT without collectives, at the TP16 shape (vocab 163840, shard 10240, 8 verify
rows per request): the served path's non-collective work after the logit all-gather (gathered softcap, row stats,
selector_logit_accept, the simulator's argmax over the forced row) vs the vocab-parallel local kernel + combine.
Collectives are timed separately (a TP16 profile and the TP test). Same GPU, alternated A/B/B/A
blocks; CUDA-graph replay time (GPU only) and eager time (with launch gaps, as the seam runs). Prints one line per cell.
"""

import torch

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.kernels.ops.speculative.selector_sampled_accept import selector_logit_accept
from sglang.kernels.ops.speculative.verify_row_stats import verify_row_stats
from sglang.kernels.ops.speculative.vocab_parallel_accept import vp_combine, vp_local_payload

DEV = torch.device("cuda")
VOCAB, WORLD, BLOCK = 163840, 16, 8
SHARD = VOCAB // WORLD


def _setup(bs, top_k):
    g = torch.Generator(device=DEV).manual_seed(bs * 31 + top_k)
    rows = bs * BLOCK
    gathered = torch.randn((WORLD, rows, SHARD), generator=g, device=DEV) * 3.0
    cid = torch.rand((bs * (BLOCK - 1), VOCAB), generator=g, device=DEV).topk(top_k, -1).indices
    cid = cid.view(bs, BLOCK - 1, top_k).contiguous()
    q = torch.rand((bs, BLOCK - 1, top_k), generator=g, device=DEV)
    q = (q / q.sum(-1, keepdim=True) * 0.9).contiguous()
    cand = torch.cat([torch.randint(0, VOCAB, (bs, 1), generator=g, device=DEV), cid[:, :, 0]], 1).contiguous()
    temps = torch.ones((bs, 1), device=DEV)
    u = torch.rand((bs, BLOCK - 1), generator=g, device=DEV)
    uf = torch.rand((bs,), generator=g, device=DEV)
    full = torch.empty((rows, VOCAB), dtype=torch.float32, device=DEV)
    cnt = torch.zeros((2,), dtype=torch.int32, device=DEV)  # the always-on cross-rank guard counter, as served
    shard0 = gathered[3].contiguous()
    pays = torch.stack(
        [
            vp_local_payload(
                logits_shard=gathered[w].contiguous(), temperatures=temps, uniform_samples_final=uf,
                candidate_ids=cid, rows_per_request=BLOCK, valid_width=SHARD, col_offset=w * SHARD,
                vocab_size=VOCAB, softcap=30.0, candidates=cand, q_rows=q, uniform_samples=u,
            )
            for w in range(WORLD)
        ]
    ).contiguous()  # fmt: skip

    def served():
        gathered_softcap_logits(gathered, 30.0, full)
        lg, stats = verify_row_stats(logits=full, temperatures=temps, rows_per_request=BLOCK)
        selector_logit_accept(
            candidates=cand, target_logits=lg, logit_stats=stats, candidate_ids=cid, q_rows=q,
            uniform_samples=u, uniform_samples_final=uf,
        )  # fmt: skip
        torch.argmax(full.view(bs, BLOCK, VOCAB)[:, 3], dim=-1)

    def vp():
        vp_local_payload(
            logits_shard=shard0, temperatures=temps, uniform_samples_final=uf, candidate_ids=cid,
            rows_per_request=BLOCK, valid_width=SHARD, col_offset=3 * SHARD, vocab_size=VOCAB, softcap=30.0,
            candidates=cand, q_rows=q, uniform_samples=u,
        )  # fmt: skip
        vp_combine(
            gathered=pays, candidates=cand, candidate_ids=cid, q_rows=q, uniform_samples=u,
            uniform_samples_final=uf, temperatures=temps, vocab_size=VOCAB, shard_width=SHARD, mismatch_counter=cnt,
        )  # fmt: skip

    return served, vp


def _graph(fn):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def _time(fn, n=20):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / n


def main():
    print(f"VP_BENCH gpu={torch.cuda.get_device_name()} vocab={VOCAB} world={WORLD} shard={SHARD}")
    for top_k in (8, 32):
        for bs in (1, 2, 8):
            served, vp = _setup(bs, top_k)
            for mode in ("graph", "eager"):
                fs = _graph(served) if mode == "graph" else served
                fv = _graph(vp) if mode == "graph" else vp
                for _ in range(5):
                    fs(), fv()
                ts, tv = [], []
                for blk in range(40):  # A B B A
                    order = (fs, fv, fv, fs) if blk % 2 == 0 else (fv, fs, fs, fv)
                    for f in order:
                        (ts if f is fs else tv).append(_time(f))
                ts.sort()
                tv.sort()
                ms, mv = ts[len(ts) // 2], tv[len(tv) // 2]
                print(f"VP_BENCH top_k={top_k} bs={bs} mode={mode} served_local_us={ms:.2f} vp_local_us={mv:.2f} delta_us={mv - ms:.2f}")


if __name__ == "__main__":
    main()
