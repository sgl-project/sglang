import itertools
from typing import Dict, Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE
from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-kernel-benchmark-1-gpu-large")

DEVICE = DEFAULT_DEVICE
DTYPE = DEFAULT_DTYPE
TOP_K = 2048
ITEM_SIZE_BYTES = 512
MISS_RATES = [0.2, 0.001]
ROUNDS = 5
WARMUP_ROUNDS = 5
BATCH_SIZES = [1, 10, 100]
HOT_BUFFER_SIZES = [4096, 8192]
CONFIGS = [
    (
        batch_size,
        hot_buffer_size,
        miss_rate,
        batch_size * round(TOP_K * miss_rate),
    )
    for batch_size, hot_buffer_size, miss_rate in itertools.product(
        BATCH_SIZES, HOT_BUFFER_SIZES, MISS_RATES
    )
]

LINE_VALS = ["jit"]
LINE_NAMES = ["SGL JIT Kernel"]
STYLES = [("blue", "--")]


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
    batch_size: int, hot_buffer_size: int, miss_rate: float
) -> Dict[str, torch.Tensor | int]:
    dtype_bytes = torch.empty((), dtype=DTYPE).element_size()
    kv_dim = ITEM_SIZE_BYTES // dtype_bytes
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
    }


def _time_kernel(batch_size: int, hot_buffer_size: int, miss_rate: float) -> float:
    state = _build_inputs(batch_size, hot_buffer_size, miss_rate)

    def run_once():
        state["device_buffer_tokens"].copy_(state["initial_device_buffer_tokens"])
        state["lru_slots"].copy_(state["initial_lru_slots"])
        state["top_k_device_locs"].fill_(-1)
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
            item_size_bytes=ITEM_SIZE_BYTES,
            num_top_k=TOP_K,
            hot_buffer_size=hot_buffer_size,
            block_size=1024,
            num_real_reqs=state["num_real_reqs"],
        )

    run_once()
    torch.cuda.synchronize()
    for _ in range(WARMUP_ROUNDS):
        run_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ROUNDS):
        run_once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / ROUNDS


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hot_buffer_size", "miss_rate", "miss_tokens_cnt"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="hisparse-latency",
        args={},
    )
)
def benchmark_latency(
    batch_size: int,
    hot_buffer_size: int,
    miss_rate: float,
    miss_tokens_cnt: int,
    provider: str,
) -> Tuple[float, float, float]:
    assert provider == "jit"
    batch_size = int(batch_size)
    hot_buffer_size = int(hot_buffer_size)
    miss_rate = float(miss_rate)
    assert miss_tokens_cnt == batch_size * _miss_tokens_per_req(miss_rate)
    avg_us = _time_kernel(batch_size, hot_buffer_size, miss_rate)
    return avg_us, avg_us, avg_us


if __name__ == "__main__":
    benchmark_latency.run(print_data=True)
