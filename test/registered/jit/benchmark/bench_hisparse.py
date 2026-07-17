import itertools
import math
from functools import cache
from typing import Dict, Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE
from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.jit_kernel.hisparse_sharded import (
    load_cache_to_device_buffer_mla_sharded,
    logical_shards_for_hot_buffer,
)
from sglang.jit_kernel.utils import is_hip_runtime
from sglang.srt.utils.bench_utils import bench_kineto
from sglang.srt.utils.numa_utils import (
    _query_numa_node_for_gpu,
    numa_bind_to_node,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=180, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=90, stage="jit-kernel-benchmark", runner_config="amd")

DEVICE = DEFAULT_DEVICE
DTYPE = DEFAULT_DTYPE
TOP_K = 2048
ITEM_SIZE_BYTES = 512
SHARDED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SHARDED_HOT_BUFFER_SIZES = [8192, 6144]
SHARDED_MISSES_PER_REQ = [2, 410]
SHARDED_CONFIGS = [
    (
        batch_size,
        hot_buffer_size,
        misses_per_req / TOP_K,
        batch_size * misses_per_req,
    )
    for hot_buffer_size, batch_size, misses_per_req in itertools.product(
        SHARDED_HOT_BUFFER_SIZES,
        SHARDED_BATCH_SIZES,
        SHARDED_MISSES_PER_REQ,
    )
]
SHARDED_BLOCK_SIZE = 512
SHARDED_MIN_BLOCKS_PER_SM = 3
SHARDED_N1_BLOCK_SIZE = 1024
SHARDED_N1_MIN_BLOCKS_PER_SM = 1
SHARDED_MAX_CTAS_PER_REQUEST = 64
KINETO_TESTS = 100


@cache
def bind_to_cuda_device_numa_node(device: torch.device | str | int) -> int | None:
    """Keep pinned-host first-touch local to the selected GPU's NUMA node."""
    if isinstance(device, int):
        gpu_id = device
    else:
        gpu_id = torch.device(device).index
        if gpu_id is None:
            gpu_id = torch.cuda.current_device()

    numa_nodes = _query_numa_node_for_gpu(gpu_id)
    if not numa_nodes:
        return None
    numa_node = numa_nodes[0]
    numa_bind_to_node(numa_node)
    return numa_node


if is_hip_runtime():
    SHARDED_LINE_VALS = ["original"]
    SHARDED_LINE_NAMES = ["Original"]
    SHARDED_STYLES = [("blue", "--")]
else:
    SHARDED_LINE_VALS = ["original", "sharded"]
    SHARDED_LINE_NAMES = ["Original", "Sharded"]
    SHARDED_STYLES = [("blue", "--"), ("green", "-")]


def _mix32(value: int) -> int:
    value = (value ^ (value >> 16)) & 0xFFFFFFFF
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value = (value ^ (value >> 15)) & 0xFFFFFFFF
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def _make_sharded_cache_tokens(
    num_hits: int,
    hot_buffer_size: int,
    seq_len: int,
) -> torch.Tensor:
    logical_shards = logical_shards_for_hot_buffer(hot_buffer_size, DEVICE)
    ways = torch.cuda.get_device_properties(DEVICE).warp_size
    tokens_by_shard: list[list[int]] = [[] for _ in range(logical_shards)]

    for token in range(num_hits):
        shard = _mix32(token) % logical_shards
        tokens_by_shard[shard].append(token)
    if any(len(tokens) > ways for tokens in tokens_by_shard):
        raise RuntimeError("top-k hits exceed shard capacity")

    remaining = hot_buffer_size - num_hits
    filler = seq_len
    while remaining:
        shard = _mix32(filler) % logical_shards
        if len(tokens_by_shard[shard]) < ways:
            tokens_by_shard[shard].append(filler)
            remaining -= 1
        filler += 1

    return torch.tensor(
        [token for shard in tokens_by_shard for token in shard], dtype=torch.int32
    )


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


def _sharded_num_ctas(batch_size: int) -> int:
    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    target_ctas_per_request = max(1.0, sm_count / batch_size)
    nearest_power_of_two = 1 << round(math.log2(target_ctas_per_request))
    return min(SHARDED_MAX_CTAS_PER_REQUEST, nearest_power_of_two)


def _build_inputs(
    batch_size: int,
    hot_buffer_size: int,
    miss_rate: float,
    provider: str,
) -> Dict[str, torch.Tensor | int]:
    bind_to_cuda_device_numa_node(DEVICE)
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
    cache_tokens = _make_sharded_cache_tokens(num_hits, hot_buffer_size, seq_len)
    device_buffer_tokens[:, :hot_buffer_size] = cache_tokens.to(DEVICE)

    if provider == "sharded":
        logical_shards = logical_shards_for_hot_buffer(hot_buffer_size, DEVICE)
        ways = torch.cuda.get_device_properties(DEVICE).warp_size
        lru_slots = (
            torch.arange(ways, dtype=torch.uint8, device=DEVICE)
            .view(1, 1, ways)
            .repeat(batch_size, logical_shards, 1)
            .reshape(batch_size, hot_buffer_size)
            .contiguous()
        )
    else:
        lru_slots = (
            torch.arange(hot_buffer_size, dtype=torch.int16, device=DEVICE)
            .view(1, -1)
            .repeat(batch_size, 1)
        )

    resident_mask = cache_tokens < seq_len
    resident_slots = torch.nonzero(resident_mask, as_tuple=False).flatten()
    resident_tokens = cache_tokens[resident_mask].to(torch.int64)
    request_offsets = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1)
    resident_host_locs = request_offsets * seq_len + resident_tokens.unsqueeze(0)
    resident_device_locs = (
        request_offsets * padded_buffer_size + resident_slots.unsqueeze(0)
    )
    device_buffer.index_copy_(
        0,
        resident_device_locs.flatten().to(DEVICE),
        host_cache[resident_host_locs.flatten()].to(DEVICE, non_blocking=True),
    )

    state = {
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
    torch.cuda.synchronize()
    return state


def _reset_state(state: Dict[str, torch.Tensor | int]) -> None:
    state["device_buffer_tokens"].copy_(state["initial_device_buffer_tokens"])
    state["lru_slots"].copy_(state["initial_lru_slots"])
    state["top_k_device_locs"].fill_(-1)


def _launch_kernel(
    state: Dict[str, torch.Tensor | int],
    batch_size: int,
    hot_buffer_size: int,
    provider: str,
) -> None:
    if provider == "sharded":
        num_ctas = _sharded_num_ctas(batch_size)
        sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        block_size = (
            SHARDED_BLOCK_SIZE
            if num_ctas > 1 or batch_size > sm_count
            else SHARDED_N1_BLOCK_SIZE
        )
        min_blocks_per_sm = (
            SHARDED_N1_MIN_BLOCKS_PER_SM if num_ctas == 1 else SHARDED_MIN_BLOCKS_PER_SM
        )
        load_cache_to_device_buffer_mla_sharded(
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
            num_real_reqs=state["num_real_reqs"],
            item_size_bytes=ITEM_SIZE_BYTES,
            num_top_k=TOP_K,
            hot_buffer_size=hot_buffer_size,
            num_ctas=num_ctas,
            block_size=block_size,
            min_blocks_per_sm=min_blocks_per_sm,
        )
    else:
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


def _check_result(state: Dict[str, torch.Tensor | int], provider: str) -> None:
    torch.cuda.synchronize()
    expected_host_locs = state["host_cache_locs"].gather(
        1, state["top_k_tokens"].to(torch.int64)
    )
    expected = state["host_cache"][expected_host_locs.cpu()].to(DEVICE)
    actual = state["device_buffer"][state["top_k_device_locs"].to(torch.int64)]
    if not torch.equal(actual, expected):
        raise AssertionError(f"{provider} returned incorrect KV data")


def _time_sharded_kernel_gpu(
    batch_size: int,
    hot_buffer_size: int,
    miss_rate: float,
    provider: str,
) -> float:
    state = _build_inputs(
        batch_size,
        hot_buffer_size,
        miss_rate,
        provider=provider,
    )

    def reset_and_launch() -> None:
        _reset_state(state)
        _launch_kernel(state, batch_size, hot_buffer_size, provider)

    reset_and_launch()
    _check_result(state, provider)
    kernel_name = (
        "sharded_kernel"
        if provider == "sharded"
        else "load_cache_to_device_buffer_kernel"
    )
    return (
        bench_kineto(
            reset_and_launch,
            kernel_names=kernel_name,
            num_tests=KINETO_TESTS,
        )
        * 1e6
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hot_buffer_size", "miss_rate", "miss_tokens_cnt"],
        x_vals=SHARDED_CONFIGS,
        line_arg="provider",
        line_vals=SHARDED_LINE_VALS,
        line_names=SHARDED_LINE_NAMES,
        styles=SHARDED_STYLES,
        ylabel="us",
        plot_name="hisparse-sharded-kineto-latency",
        args={},
    )
)
def benchmark_sharded_kernel_gpu(
    batch_size: int,
    hot_buffer_size: int,
    miss_rate: float,
    miss_tokens_cnt: int,
    provider: str,
) -> Tuple[float, float, float]:
    batch_size = int(batch_size)
    hot_buffer_size = int(hot_buffer_size)
    miss_rate = float(miss_rate)
    assert miss_tokens_cnt == batch_size * _miss_tokens_per_req(miss_rate)
    avg_us = _time_sharded_kernel_gpu(
        batch_size,
        hot_buffer_size,
        miss_rate,
        provider=provider,
    )
    return avg_us, avg_us, avg_us


if __name__ == "__main__":
    benchmark_sharded_kernel_gpu.run(print_data=True)
