from typing import Dict

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE
from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = DEFAULT_DEVICE
DTYPE = DEFAULT_DTYPE
TOP_K = 2048
ITEM_SIZE_BYTES = 512

# (hot_buffer_size, miss_rate) correlated axis
HOT_BUFFER_MISS = [
    (4096, 0.2),
    (4096, 0.001),
    (8192, 0.2),
    (8192, 0.001),
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


@marker.parametrize("hot_buffer_size,miss_rate", HOT_BUFFER_MISS, HOT_BUFFER_MISS[:2])
@marker.parametrize("batch_size", [1, 10, 100], [1, 10])
@marker.benchmark("impl", ["jit"])
def benchmark_latency(
    hot_buffer_size: int,
    miss_rate: float,
    batch_size: int,
    impl: str,
):
    hot_buffer_size = int(hot_buffer_size)
    miss_rate = float(miss_rate)
    state = _build_inputs(batch_size, hot_buffer_size, miss_rate)

    def run_once():
        # CPU->GPU transfers can not be CUDA-graph captured; reset mutated state
        # each iteration so the workload matches the original benchmark.
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

    return marker.do_bench(
        run_once,
        use_cuda_graph=False,  # CPU<->GPU memcpy can not be captured in a CUDA graph
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark_latency.run()
