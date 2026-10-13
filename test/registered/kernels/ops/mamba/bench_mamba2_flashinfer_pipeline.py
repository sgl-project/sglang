"""Kernel-only 40-layer verify + accepted-state/conv rollback cost.

Uses the production pool, runtime wrapper and scatter operation. Does not load
a model or run requests. Full projections/attention are intentionally excluded.
"""

import gc
import json

import torch
import triton
from sglang.kernels.ops.mamba.mamba2_spec_replay import (
    commit_mamba2_replay,
    verify_mamba2_replay,
)
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.runtime_context import get_context
from test_mamba2_flashinfer_replay import FlashInferCase


@torch.inference_mode()
def benchmark():
    for batch in (16, 32, 64):
        _batch_benchmark(batch)
        gc.collect()
        torch.cuda.empty_cache()


def _batch_benchmark(batch):
    layers, width = 40, 4
    params = Mamba2CacheParams(
        shape=Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8192,
            n_groups=8,
            num_heads=128,
            head_dim=64,
            state_size=128,
            conv_kernel=4,
        ),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float16),
        layers=list(range(layers)),
    )
    pools = [
        MambaPool(
            size=2 * batch + 2,
            spec_state_size=batch,
            cache_params=params,
            mamba_layer_ids=list(range(layers)),
            device="cuda",
            speculative_num_draft_tokens=width,
            speculative_eagle_topk=1,
            enable_mamba2_spec_replay=replay,
            mamba2_replay_dtype=torch.bfloat16,
        )
        for replay in (False, True)
    ]
    cases = []
    for layer in range(layers):
        c = FlashInferCase(batch=batch, rounding=True)
        c.initial = None
        c.state = pools[1].mamba_cache.temporal[layer]
        c.reference = pools[0].mamba_cache.temporal[layer]
        c.snapshots = pools[0].mamba_cache.intermediate_ssm[layer]
        c.state.zero_()
        c.reference.zero_()
        cases.append(c)
    caches = [p.mamba_cache for p in pools]
    for cache in caches:
        cache.conv[0].zero_()
        cache.intermediate_conv_window[0].zero_()
    layer_caches = [pools[1].mamba2_layer_cache(i) for i in range(layers)]
    slots = cases[0].slots
    last = torch.full((batch,), 3, device="cuda", dtype=torch.int32)
    tracks = torch.arange(batch, device="cuda", dtype=torch.int32) * 2
    track_steps = torch.full_like(last, -1)

    def verify(replay):
        for c, cache in zip(cases, layer_caches):
            if replay:
                verify_mamba2_replay(
                    c.state,
                    c.x,
                    c.dt,
                    c.A,
                    c.B,
                    c.C,
                    c.D,
                    layer_cache=cache,
                    dt_bias=c.bias,
                    dt_softplus=True,
                    state_batch_indices=slots,
                    out=c.out,
                    disable_state_update=True,
                    cache_steps=width,
                )
            else:
                c.seed.random_(0, 2**32)
                c.verify(False)

    def commit(replay):
        if replay:
            commit_mamba2_replay(caches[1], slots, last, tracks, track_steps)
        scatter_mamba_states_after_mtp_verify(
            caches[int(replay)],
            slots,
            last,
            tracks,
            track_steps,
            skip_ssm=replay,
        )

    def run(replay):
        verify(replay)
        commit(replay)

    with get_context().override_server_args(
        enable_mamba_cache_stochastic_rounding=True,
        mamba_cache_philox_rounds=5,
    ):
        for tracking in ("none", "all"):
            track_steps.fill_(-1 if tracking == "none" else 1)
            for replay in (False, True):
                run(replay)
            torch.cuda.synchronize()
            timings = {}
            for name, replay in (("baseline_ms", False), ("compact_ms", True)):
                timings[name] = triton.testing.do_bench_cudagraph(
                    lambda: run(replay), rep=200
                )
                for component, fn in (("verify", verify), ("commit_conv", commit)):
                    timings[name.replace("_ms", f"_{component}_ms")] = (
                        triton.testing.do_bench_cudagraph(lambda: fn(replay), rep=200)
                    )
            print(
                json.dumps(
                    {
                        "benchmark": "40_layer_verify_commit_conv",
                        "batch": batch,
                        "layers": layers,
                        "tracking": tracking,
                        "baseline_pool_bytes": round(pools[0].mem_usage * (1 << 30)),
                        "compact_pool_bytes": round(pools[1].mem_usage * (1 << 30)),
                        "speedup": timings["baseline_ms"] / timings["compact_ms"],
                        **timings,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    benchmark()
