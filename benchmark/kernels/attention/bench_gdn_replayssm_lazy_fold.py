"""GDN ReplaySSM verify path of one decode step with a bf16 checkpoint, folding every
commit (SGLANG_OPT_GDN_REPLAYSSM_LAZY_FOLD off) or lazily.

A step is the target verify of every GDN layer (one CUDA graph of
`gdn_replayssm_spec_decode` calls) and one `commit_gdn_replayssm_after_verify`, on a
MambaPool laid out as the server builds it. Per request count: warm-up steps, then
sampled steps; each sampled step times the verify graph and the commit at that step's
ring state (512 MB written before each repetition to evict the caches, median of the
repetitions; the commit's state is restored after each), then commits. Accept lengths
come from a chain draft accepted token by token at a constant rate, the same sequence
for every arm. Defaults: Qwen3.8-Flash-Next at TP1 (36 GDN layers, 16 key and 48 value
heads of 128, 4 draft tokens, a 16-entry ring); random states and inputs.

Arms: off (fold every commit); on (the opt-in: the lazy fold with compensated history in
batches where replayssm_lazy_fold_pays, every commit folding in smaller ones);
on-always-lazy (the lazy fold at any batch size); on-uncompensated (the lazy fold at any
batch size, history read without the ring's low parts; for reference).

    python benchmark/kernels/attention/bench_gdn_replayssm_lazy_fold.py --out bench.json
"""

import argparse
import json
import statistics

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_after_verify,
    gdn_replayssm_spec_decode,
    replayssm_lazy_fold_pays,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool

# arm: (lazy fold: "never", "gated" as the opt-in or "always", compensated history)
ARMS = {
    "off": ("never", False),
    "on": ("gated", True),
    "on-always-lazy": ("always", True),
    "on-uncompensated": ("always", False),
}


def make_pool(a):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=a.v_heads * a.head_dim,
        n_groups=a.k_heads,
        num_heads=a.v_heads,
        head_dim=a.head_dim,
        state_size=a.head_dim,
        conv_kernel=4,
        conv_shard_groups=[a.k_heads * a.head_dim] * 2 + [a.v_heads * a.head_dim],
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16)
    params = Mamba2CacheParams(shape=shape, layers=list(range(a.layers)), dtype=dtype)
    return MambaPool(
        size=a.state_slots,
        spec_state_size=a.request_rows,
        cache_params=params,
        mamba_layer_ids=list(range(a.layers)),
        device="cuda",
        speculative_num_draft_tokens=a.drafts,
        speculative_eagle_topk=1,
        enable_linear_replayssm_spec=True,
        linear_replayssm_cache_len=a.ring,
    )


def time_us(fn, reps, flush, after=lambda: None):
    times = []
    for i in range(reps + 3):
        flush.fill_(1.0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        after()
        if i >= 3:
            times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def run_point(a, num_requests, flush):
    torch.manual_seed(num_requests)
    pool = make_pool(a)
    cache = pool.mamba_cache
    hk, hv, d, n = a.k_heads, a.v_heads, a.head_dim, num_requests
    slots = torch.randperm(a.state_slots, device="cuda")[:n].int()
    rows = torch.randperm(a.request_rows - 1, device="cuda")[:n] + 1
    cache.temporal[:, slots.long()] = (
        torch.randn(a.layers, n, hv, d, d, device="cuda") * 0.05
    ).bfloat16()
    bases = torch.randint(0, a.ring, (n,), device="cuda", dtype=torch.int32)
    tokens = n * a.drafts

    def randn(*shape, scale=0.5):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    q, k, v = randn(tokens, hk, d), randn(tokens, hk, d), randn(tokens, hv, d)
    gate_a, gate_b = randn(tokens, hv, scale=1.0), randn(tokens, hv, scale=1.0)
    A_log = torch.empty(hv, device="cuda").uniform_(-4.0, 1.0)
    dt_bias = randn(hv, scale=0.2)
    qsl = torch.arange(0, tokens + 1, a.drafts, device="cuda", dtype=torch.int32)
    out = torch.empty(tokens, hv, d, device="cuda", dtype=torch.bfloat16)
    accept = torch.ones(n, device="cuda", dtype=torch.int32)

    def verify(compensated):
        for layer in range(a.layers):
            lc = pool.mamba2_layer_cache(layer)
            gdn_replayssm_spec_decode(
                q=q,
                k=k,
                v=v,
                a=gate_a,
                b=gate_b,
                A_log=A_log,
                dt_bias=dt_bias,
                checkpoint_state=lc.temporal,
                d_cache=lc.replayssm_d,
                k_cache=lc.replayssm_k,
                g_cache=lc.replayssm_g,
                rawv_cache=lc.replayssm_rawv,
                rawk_cache=lc.replayssm_rawk,
                beta_cache=lc.replayssm_beta,
                out=out,
                query_start_loc=qsl,
                ssm_state_indices=slots,
                replay_indices=rows,
                write_pos=pool.replayssm_spec_write_pos,
                cache_base=pool.replayssm_cache_base,
                is_flush=pool.replayssm_is_flush,
                max_cache_len=a.ring,
                max_spec_len=a.drafts,
                scale=d**-0.5,
                use_qk_l2norm_in_kernel=True,
                null_block_id=-1,
                launch_mode="verify",
                compensated_history=compensated,
            )

    graphs = {}
    for compensated in {ARMS[arm][1] for arm in a.arms}:
        verify(compensated)
        torch.cuda.synchronize()
        graphs[compensated] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[compensated]):
            verify(compensated)

    cursors = (
        pool.replayssm_spec_write_pos,
        pool.replayssm_cache_base,
        pool.replayssm_is_flush,
    )
    temporal0 = cache.temporal[:, slots.long()].clone()
    p = a.accept_rate
    accept_probs = torch.tensor([(1 - p) * p**i for i in range(a.drafts - 1)])
    accept_probs = torch.cat([accept_probs, torch.tensor([p ** (a.drafts - 1)])])
    result = {}
    for i, arm in enumerate(a.arms):
        lazy, compensated = ARMS[arm]
        every = lazy == "never" or (
            lazy == "gated" and not replayssm_lazy_fold_pays(n, a.v_heads)
        )
        cache.temporal[:, slots.long()] = temporal0
        for cursor in cursors:
            cursor.zero_()
        pool.replayssm_cache_base[rows] = bases
        generator = torch.Generator().manual_seed(7)

        def commit():
            commit_gdn_replayssm_after_verify(
                spec_state=pool.get_speculative_mamba2_params_all_layers(),
                write_pos=pool.replayssm_spec_write_pos,
                cache_base=pool.replayssm_cache_base,
                is_flush=pool.replayssm_is_flush,
                state_batch_indices=slots,
                replay_indices=rows,
                accept_lens=accept,
                last_correct_step_indices=(accept - 1).long(),
                max_spec_len=a.drafts,
                fold_every_commit=every,
                null_block_id=-1,
            )

        steps = []
        for step in range(a.warmup + a.samples):
            draws = torch.multinomial(accept_probs, n, True, generator=generator)
            accept.copy_(draws + 1)
            if step >= a.warmup:
                history = pool.replayssm_spec_write_pos[rows]
                saved = [c.clone() for c in cursors]
                temporal = cache.temporal[:, slots.long()].clone()

                def restore():
                    cache.temporal[:, slots.long()] = temporal
                    for cursor, value in zip(cursors, saved):
                        cursor.copy_(value)

                steps.append(
                    dict(
                        verify=time_us(graphs[compensated].replay, a.reps, flush),
                        commit=time_us(commit, a.reps, flush, restore),
                        rows_with_history=int((history > 0).sum()),
                    )
                )
            commit()
        mean = {key: statistics.mean(s[key] for s in steps) for key in steps[0]}
        mean["step"] = mean["verify"] + mean["commit"]
        result[f"{i}:{arm}"] = mean
        print(
            f"{num_requests:3d} requests {arm:>16}: step {mean['step']:8.1f} us "
            f"(verify {mean['verify']:7.1f}, commit {mean['commit']:7.1f}, "
            f"rows with history {mean['rows_with_history']:.1f})",
            flush=True,
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--requests",
        type=int,
        nargs="+",
        default=[2, 3, 5, 7, 12, 16, 24, 32, 40, 52, 64],
    )
    p.add_argument("--arms", nargs="+", choices=ARMS, default=["off", "on", "off"])
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--k-heads", type=int, default=16)
    p.add_argument("--v-heads", type=int, default=48)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--drafts", type=int, default=4)
    p.add_argument("--ring", type=int, default=16)
    p.add_argument("--state-slots", type=int, default=640)
    p.add_argument("--request-rows", type=int, default=128)
    p.add_argument(
        "--accept-rate",
        type=float,
        default=0.64,
        help="probability that a draft token is accepted when the one before it was; "
        "the server logs (mean accept length - 1) / (drafts - 1) as its accept rate "
        "(0.44 at 0.64 with 4 draft tokens: mean accept length 2.31)",
    )
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--out")
    a = p.parse_args()
    flush = torch.empty(128 * 1024 * 1024, device="cuda")
    results = {n: run_point(a, n, flush) for n in a.requests}
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
