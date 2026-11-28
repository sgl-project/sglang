import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.kvcache.batched_store_cache import batched_store_cache
from sglang.srt.mem_cache.memory_pool import (
    _set_kv_buffer_impl,
    _set_kv_buffer_prefix_valid_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)


@marker.parametrize("layers", [2, 4, 5, 8], [5])
@marker.parametrize("tokens", [16, 128, 1024, 8192], [128])
@marker.parametrize("prefix", [False, True], [True])
@marker.benchmark("impl", ["per_layer", "batched"])
def benchmark(layers, tokens, prefix, impl):
    heads, dim, capacity = 8, 128, 32768
    k = torch.randn(
        layers, tokens + 16, heads, dim, device="cuda", dtype=torch.bfloat16
    )[:, :tokens]
    v = torch.randn_like(k)
    kc = tuple(
        torch.empty(capacity, heads, dim, device="cuda", dtype=k.dtype)
        for _ in range(layers)
    )
    vc = tuple(torch.empty_like(c) for c in kc)
    loc = torch.randperm(capacity - 1, device="cuda")[:tokens] + 1
    locs = (loc,) * layers
    width = 16
    lens = (
        torch.full((tokens // width,), 8, device="cuda", dtype=torch.int32)
        if prefix
        else None
    )

    def run():
        if impl == "batched":
            batched_store_cache(k, v, kc, vc, locs, lens, width)
        else:
            for i in range(layers):
                if prefix:
                    _set_kv_buffer_prefix_valid_impl(
                        k[i],
                        v[i],
                        kc[i],
                        vc[i],
                        loc.view(-1, width),
                        lens,
                        heads * dim,
                        k.dtype,
                    )
                else:
                    _set_kv_buffer_impl(
                        k[i],
                        v[i],
                        kc[i],
                        vc[i],
                        loc,
                        heads * dim,
                        k.dtype,
                        torch.cuda,
                        capacity,
                    )

    return marker.do_bench(run, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
