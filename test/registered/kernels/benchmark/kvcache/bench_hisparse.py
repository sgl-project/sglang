import itertools
from functools import cache
from typing import Dict

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE
from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=12, stage="jit-kernel-benchmark", runner_config="amd")

DEVICE = DEFAULT_DEVICE
DTYPE = DEFAULT_DTYPE
TOP_K = 2048
ITEM_SIZE_BYTES = [576, 656]
MISS_RATES = [0.001, 0.2, 1.0]
ROUNDS = 50
WARMUP_ROUNDS = 20
BATCH_SIZES = [1, 2, 8]
HOT_BUFFER_SIZES = [8192]
CONFIGS = [
    (
        batch_size,
        hot_buffer_size,
        item_size_bytes,
        miss_rate,
        batch_size * round(TOP_K * miss_rate),
    )
    for batch_size, hot_buffer_size, item_size_bytes, miss_rate in itertools.product(
        BATCH_SIZES, HOT_BUFFER_SIZES, ITEM_SIZE_BYTES, MISS_RATES
    )
]


def _make_top_k_tokens(
    num_hits: int, num_misses: int, hot_buffer_size: int
) -> torch.Tensor:
    hit_tokens = torch.arange(num_hits, dtype=torch.int32, device=DEVICE)
    miss_tokens = hot_buffer_size + torch.arange(
        num_misses, dtype=torch.int32, device=DEVICE
    )
    return torch.cat([hit_tokens, miss_tokens])


def _miss_tokens_per_req(miss_rate: float) -> int:
    return round(TOP_K * miss_rate)


def _build_inputs(
    batch_size: int, hot_buffer_size: int, item_size_bytes: int, miss_rate: float
) -> Dict[str, torch.Tensor | int]:
    dtype_bytes = torch.empty((), dtype=DTYPE).element_size()
    assert item_size_bytes % dtype_bytes == 0
    kv_dim = item_size_bytes // dtype_bytes
    padded_buffer_size = hot_buffer_size + 1
    seq_len = hot_buffer_size + TOP_K + 1
    num_misses = _miss_tokens_per_req(miss_rate)
    num_hits = TOP_K - num_misses

    top_k_row = _make_top_k_tokens(num_hits, num_misses, hot_buffer_size)
    top_k_tokens = top_k_row.view(1, -1).repeat(batch_size, 1).contiguous()

    host_stride = seq_len
    total_host_tokens = batch_size * host_stride
    host_cache = torch.empty(
        (total_host_tokens, 1, kv_dim), dtype=DTYPE, device="cpu", pin_memory=True
    )
    host_cache.copy_(torch.randn_like(host_cache))

    total_device_tokens = batch_size * padded_buffer_size
    device_buffer = torch.empty(
        (total_device_tokens, 1, kv_dim), dtype=DTYPE, device=DEVICE
    )
    device_buffer.normal_()

    device_buffer_locs = torch.arange(
        total_device_tokens, dtype=torch.int32, device=DEVICE
    ).view(batch_size, padded_buffer_size)
    device_buffer_tokens = torch.full(
        (batch_size, padded_buffer_size), -1, dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens[:, :hot_buffer_size] = torch.arange(
        hot_buffer_size, dtype=torch.int32, device=DEVICE
    )

    lru_slots = (
        torch.arange(hot_buffer_size, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(batch_size, 1)
    )

    return {
        "top_k_tokens": top_k_tokens,
        "device_buffer_tokens": device_buffer_tokens,
        "initial_device_buffer_tokens": device_buffer_tokens.clone(),
        "host_cache_locs": torch.arange(
            total_host_tokens, dtype=torch.int64, device=DEVICE
        ).view(batch_size, host_stride),
        "device_buffer_locs": device_buffer_locs,
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "initial_device_buffer": device_buffer.clone(),
        "top_k_device_locs": torch.empty(
            (batch_size, TOP_K), dtype=torch.int32, device=DEVICE
        ),
        "req_pool_indices": torch.arange(batch_size, dtype=torch.int64, device=DEVICE),
        "seq_lens": torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=DEVICE
        ),
        "lru_slots": lru_slots,
        "initial_lru_slots": lru_slots.clone(),
        "num_real_reqs": torch.tensor([batch_size], dtype=torch.int32, device=DEVICE),
        "miss_top_k_indices": torch.empty(
            (batch_size, TOP_K), dtype=torch.int32, device=DEVICE
        ),
        "miss_counts": torch.empty(batch_size, dtype=torch.int32, device=DEVICE),
    }


@cache
def _time_kernel(
    batch_size: int, hot_buffer_size: int, item_size_bytes: int, miss_rate: float
) -> tuple[float, float]:
    state = _build_inputs(batch_size, hot_buffer_size, item_size_bytes, miss_rate)

    def reset_state():
        state["device_buffer_tokens"].copy_(state["initial_device_buffer_tokens"])
        state["lru_slots"].copy_(state["initial_lru_slots"])
        state["device_buffer"].copy_(state["initial_device_buffer"])

    def run_kernel():
        load_cache_to_device_buffer_mla(
            top_k_tokens=state["top_k_tokens"],
            device_buffer_tokens=state["device_buffer_tokens"],
            host_cache_locs=state["host_cache_locs"],
            device_buffer_locs=state["device_buffer_locs"],
            host_cache=state["host_cache"],
            device_buffer=state["device_buffer"],
            top_k_device_locs=state["top_k_device_locs"],
            req_pool_indices=state["req_pool_indices"],
            seq_lens=state["seq_lens"],
            lru_slots=state["lru_slots"],
            item_size_bytes=item_size_bytes,
            num_top_k=TOP_K,
            hot_buffer_size=hot_buffer_size,
            block_size=960,
            num_real_reqs=state["num_real_reqs"],
            miss_top_k_indices=state["miss_top_k_indices"],
            miss_counts=state["miss_counts"],
        )

    # Compile before capture, then replay the same static buffers that Decode
    # uses in production. Reset cache state outside the timed graph replay so
    # the reported latency contains only miss resolution and Host KV copy.
    reset_state()
    run_kernel()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_kernel()
    torch.cuda.synchronize()

    for _ in range(WARMUP_ROUNDS):
        reset_state()
        graph.replay()
    torch.cuda.synchronize()

    latencies_us = []
    for _ in range(ROUNDS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        reset_state()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0)

    latencies = torch.tensor(latencies_us, dtype=torch.float64)
    return (
        torch.quantile(latencies, 0.5).item(),
        torch.quantile(latencies, 0.95).item(),
    )


@marker.parametrize(
    "batch_size,hot_buffer_size,item_size_bytes,miss_rate,miss_tokens_cnt", CONFIGS
)
@marker.benchmark("metric", ["p50", "p95"])
def benchmark_latency(
    batch_size: int,
    hot_buffer_size: int,
    item_size_bytes: int,
    miss_rate: float,
    miss_tokens_cnt: int,
    metric: str,
) -> marker.BenchResult:
    batch_size = int(batch_size)
    hot_buffer_size = int(hot_buffer_size)
    miss_rate = float(miss_rate)
    assert miss_tokens_cnt == batch_size * _miss_tokens_per_req(miss_rate)
    p50_us, p95_us = _time_kernel(
        batch_size, hot_buffer_size, item_size_bytes, miss_rate
    )
    latency_us = p50_us if metric == "p50" else p95_us
    return marker.BenchResult((0.5,), [latency_us * 1e-6], None)


if __name__ == "__main__":
    benchmark_latency.run()
