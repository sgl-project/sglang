#!/usr/bin/env python3
"""
Benchmark script for testing sparse KV cache kernels.

python bench_sparse_kernel.py --test_mode jit   --batch_size 20  --top_k 2048  --num_layers 4 --max_seq_len 4000 --pool_size 256 --token_stride_size 512 --num_warmup 3 --num_iterations 10 --test_layer_id 0 --diff_ratio 0.25
"""

import argparse
import logging
from typing import Tuple

import numpy as np
import torch


def create_mock_topk_data(
    batch_size: int,
    top_k: int,
    max_seq_len: int,
    diff_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    prev_top_k_result = []
    curr_top_k_result = []
    seq_lens = []

    for b in range(batch_size):
        num_diff = int(top_k * diff_ratio)
        num_overlap = top_k - num_diff

        min_seq_len = max(top_k, top_k + num_diff)
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        seq_lens.append(seq_len)
        available_tokens = list(range(seq_len))

        prev_indices = np.random.choice(available_tokens, size=top_k, replace=False)
        prev_indices = np.sort(prev_indices)

        overlap_mask = np.random.choice(top_k, size=num_overlap, replace=False)
        overlap_indices = prev_indices[overlap_mask]

        prev_max = prev_indices[-1]
        remaining_tokens = [t for t in available_tokens if t not in prev_indices]

        if prev_max not in overlap_indices:
            candidates_gte_prev_max = [t for t in remaining_tokens if t >= prev_max]
            if len(candidates_gte_prev_max) > 0:
                selected_high = np.random.choice(
                    candidates_gte_prev_max, size=1, replace=False
                )
                remaining_for_rest = [
                    t for t in remaining_tokens if t != selected_high[0]
                ]
                if num_diff > 1 and len(remaining_for_rest) >= num_diff - 1:
                    selected_rest = np.random.choice(
                        remaining_for_rest, size=num_diff - 1, replace=False
                    )
                    new_indices = np.concatenate([selected_high, selected_rest])
                else:
                    new_indices = selected_high
                    if len(remaining_for_rest) > 0:
                        selected_rest = np.random.choice(
                            remaining_for_rest,
                            size=min(num_diff - 1, len(remaining_for_rest)),
                            replace=False,
                        )
                        new_indices = np.concatenate([selected_high, selected_rest])
            else:
                if prev_max not in overlap_indices and num_overlap < top_k:
                    overlap_indices = np.append(overlap_indices, prev_max)
                    num_overlap += 1
                    num_diff -= 1
                new_indices = (
                    np.random.choice(
                        remaining_tokens,
                        size=min(num_diff, len(remaining_tokens)),
                        replace=False,
                    )
                    if num_diff > 0
                    else np.array([])
                )
        else:
            new_indices = np.random.choice(
                remaining_tokens, size=num_diff, replace=False
            )

        curr_indices = np.concatenate([overlap_indices, new_indices])
        curr_indices = np.sort(curr_indices)

        prev_top_k_result.append(prev_indices)
        curr_top_k_result.append(curr_indices)

    prev_top_k_result = torch.tensor(
        np.array(prev_top_k_result), dtype=torch.int64, device="cuda"
    )
    curr_top_k_result = torch.tensor(
        np.array(curr_top_k_result), dtype=torch.int64, device="cuda"
    )
    seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")

    return prev_top_k_result, curr_top_k_result, seq_lens


def create_mock_sparse_manager_states(
    batch_size: int,
    top_k: int,
    num_layers: int,
    max_pool_size: int = 2048,
    max_seq_len: int = 2048,
) -> dict:
    states = {
        "prev_top_k_result_pool": torch.full(
            (max_pool_size, num_layers, top_k), -1, dtype=torch.int64, device="cuda"
        ),
        "prev_device_indices_pool": torch.full(
            (max_pool_size, num_layers, top_k + 1), -1, dtype=torch.int64, device="cuda"
        ),
        "curr_device_indices": torch.full(
            (batch_size, top_k + 1), -1, dtype=torch.int64, device="cuda"
        ),
        "bitmap": torch.full(
            (batch_size, max_seq_len), -1, dtype=torch.int32, device="cuda"
        ),
        "full_host_indices": torch.full(
            (max_pool_size, max_seq_len), -1, dtype=torch.int64, device="cuda"
        ),
        "should_load_device_indices": torch.full(
            (batch_size, top_k + 1), -1, dtype=torch.int64, device="cuda"
        ),
        "should_load_host_indices": torch.full(
            (batch_size, top_k + 1), -1, dtype=torch.int64, device="cuda"
        ),
    }

    return states


def create_mock_memory_pools(
    num_layers: int,
    pool_size: int,
    token_stride_size: int,
    device: str = "cuda",
):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

    device_pool = MLATokenToKVPool(
        size=pool_size,
        page_size=1,
        dtype=torch.float16,
        kv_lora_rank=token_stride_size,
        qk_rope_head_dim=64,
        layer_num=num_layers,
        device=device,
        enable_memory_saver=False,
        use_nsa=True,
    )

    host_pool = MLATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=1.5,
        host_size=0,
        page_size=1,
        layout="layer_first",
        pin_memory=True,
        device="cpu",
    )

    print(f"   Host pool created: {host_pool.size} tokens")
    return device_pool, host_pool


def initialize_host_cache_data(
    host_pool,
    layer_id: int,
    host_cache_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    page_size: int = 1,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    feature_dim = host_pool.kv_buffer[layer_id].shape[-1]
    req_pool_indices_cpu = req_pool_indices.cpu()
    seq_lens_cpu = seq_lens.cpu()
    host_cache_locs_cpu = host_cache_locs.cpu()

    host_locs_to_init = []
    for b in range(batch_size):
        req_idx = req_pool_indices_cpu[b].item()
        seq_len = int(seq_lens_cpu[b].item())

        for token_id in range(seq_len):
            host_loc = host_cache_locs_cpu[req_idx, token_id].item()
            if host_loc >= 0:
                for page_offset in range(page_size):
                    host_locs_to_init.append(host_loc * page_size + page_offset)

    if len(host_locs_to_init) > 0:
        all_random_data = torch.randn(
            len(host_locs_to_init), feature_dim, dtype=torch.float16, device="cpu"
        )
        for i, host_loc in enumerate(host_locs_to_init):
            host_pool.kv_buffer[layer_id][host_loc, 0].copy_(all_random_data[i])


def copy_prev_topk_to_device(
    device_pool,
    host_pool,
    device_buffer_locs: torch.Tensor,
    host_cache_locs: torch.Tensor,
    prev_top_k: torch.Tensor,
    curr_top_k: torch.Tensor,
    req_pool_indices: torch.Tensor,
    layer_id: int,
    page_size: int = 1,
):
    batch_size = prev_top_k.shape[0]
    top_k = prev_top_k.shape[1]

    prev_top_k_cpu = prev_top_k.cpu()
    curr_top_k_cpu = curr_top_k.cpu()
    device_buffer_locs_cpu = device_buffer_locs.cpu()
    req_pool_indices_cpu = req_pool_indices.cpu()
    host_cache_locs_cpu = host_cache_locs.cpu()

    for b in range(batch_size):
        req_idx = req_pool_indices_cpu[b].item()

        prev_max = prev_top_k_cpu[b].max().item()
        curr_max = curr_top_k_cpu[b].max().item()
        cross_page = curr_max != prev_max

        for t in range(top_k):
            token_id = prev_top_k_cpu[b, t].item()

            if cross_page and t == top_k - 1:
                token_id = curr_max

            if token_id < 0:
                continue

            host_loc = host_cache_locs_cpu[req_idx, token_id].item()
            device_loc = device_buffer_locs_cpu[req_idx, layer_id, t].item()

            if host_loc >= 0 and device_loc >= 0:
                for page_offset in range(page_size):
                    data = host_pool.kv_buffer[layer_id][
                        host_loc * page_size + page_offset, 0
                    ]
                    device_pool.kv_buffer[layer_id][
                        device_loc * page_size + page_offset, 0
                    ].copy_(data)


def setup_benchmark_data(
    batch_size: int,
    top_k: int,
    max_seq_len: int,
    diff_ratio: float = 0.20,
):
    """Setup common mock data for benchmarking.

    Returns:
        (prev_top_k_result, curr_top_k_result, seq_lens)
    """
    print("\n[1] Creating mock topk data...")
    prev_top_k_result, curr_top_k_result, seq_lens = create_mock_topk_data(
        batch_size=batch_size,
        top_k=top_k,
        max_seq_len=max_seq_len,
        diff_ratio=diff_ratio,
    )

    overlap_count = 0
    for b in range(batch_size):
        prev_set = set(prev_top_k_result[b].cpu().numpy())
        curr_set = set(curr_top_k_result[b].cpu().numpy())
        overlap_count += len(prev_set & curr_set)
    avg_overlap = overlap_count / (batch_size * top_k)
    avg_diff = 1.0 - avg_overlap
    print(f"   Avg diff ratio: {avg_diff * 100:.2f}%")
    print(f"   Avg overlap ratio: {avg_overlap * 100:.2f}%")

    return prev_top_k_result, curr_top_k_result, seq_lens


def setup_memory_pools(
    num_layers: int,
    pool_size: int,
    token_stride_size: int,
):
    device_pool, host_pool = create_mock_memory_pools(
        num_layers=num_layers,
        pool_size=pool_size,
        token_stride_size=token_stride_size,
        device="cuda",
    )
    print(f"   Host pool created: {host_pool.size} tokens")
    return device_pool, host_pool


def verify_kernel_correctness(
    device_pool,
    host_pool,
    layer_id: int,
    batch_size: int,
    top_k: int,
    top_k_tokens: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    host_cache_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_size: int = 1,
    prev_top_k_tokens: torch.Tensor = None,
) -> tuple[bool, int, int]:
    valid_mask = top_k_tokens >= 0

    if not valid_mask.any():
        return (True, 0, 0)

    if prev_top_k_tokens is not None:
        prev_max = prev_top_k_tokens.max(dim=1)[0]
        curr_max = top_k_tokens.max(dim=1)[0]
        cross_page_mask = (curr_max != prev_max).unsqueeze(1)

        position_mask = torch.zeros(
            batch_size, top_k, dtype=torch.bool, device=top_k_tokens.device
        )
        position_mask[:, top_k - 1] = True

        skip_mask = cross_page_mask & position_mask
        valid_mask = valid_mask & ~skip_mask

    req_indices_expanded = req_pool_indices.unsqueeze(1).expand(-1, top_k)
    token_ids_clamped = top_k_tokens.clamp(min=0, max=host_cache_locs.size(1) - 1)
    host_locs_all = host_cache_locs[req_indices_expanded, token_ids_clamped]
    valid_entries_mask = valid_mask & (host_locs_all >= 0) & (top_k_device_locs >= 0)

    if not valid_entries_mask.any():
        return (True, 0, 0)

    device_locs_valid = top_k_device_locs[valid_entries_mask] * page_size
    host_locs_valid = host_locs_all[valid_entries_mask] * page_size

    total_count = device_locs_valid.numel()

    device_data_all = device_pool.kv_buffer[layer_id][device_locs_valid, 0]
    try:
        host_data_all = host_pool.kv_buffer[layer_id][host_locs_valid.cpu(), 0].cuda()
    except:
        print("   [ERROR] Cannot access host pool buffer from GPU!")
        return (False, total_count, total_count)

    diff = torch.abs(device_data_all - host_data_all)
    threshold = 1e-3 + 1e-3 * torch.abs(host_data_all)
    matches = (diff <= threshold).all(dim=-1)

    mismatch_count = (~matches).sum().item()
    success = mismatch_count == 0

    return (success, mismatch_count, total_count)


def print_benchmark_results(elapsed_times: np.ndarray, kernel_name: str):
    print("\n" + "=" * 50)
    print(f"{kernel_name} Performance:")
    print("=" * 50)
    print(f"Mean:   {elapsed_times.mean():.4f} ms")
    print(f"Median: {np.median(elapsed_times):.4f} ms")
    print(f"Min:    {elapsed_times.min():.4f} ms")
    print(f"Max:    {elapsed_times.max():.4f} ms")
    print(f"Std:    {elapsed_times.std():.4f} ms")
    print(f"P95:    {np.percentile(elapsed_times, 95):.4f} ms")
    print("=" * 50)


def run_unified_benchmark(
    kernel_name: str,
    kernel_executor,
    setup_fn,
    reset_fn,
    verify_fn,
    batch_size: int,
    top_k: int,
    num_layers: int,
    max_seq_len: int,
    pool_size: int,
    token_stride_size: int,
    num_warmup: int,
    num_iterations: int,
    test_layer_id: int,
    diff_ratio: float = 0.20,
):
    print(f"\n========== {kernel_name} Benchmark ==========")
    print(f"Batch Size: {batch_size}")
    print(f"Top K: {top_k}")
    print(f"Num Layers: {num_layers}")
    print(f"Max Seq Len: {max_seq_len}")
    print(f"Pool Size: {pool_size}")
    print(f"Token Stride Size: {token_stride_size}")
    print(f"Test Layer ID: {test_layer_id}")
    print(f"Diff Ratio: {diff_ratio * 100:.1f}%")
    print("=" * 50)

    # Setup common data
    prev_top_k_result, curr_top_k_result, seq_lens = setup_benchmark_data(
        batch_size, top_k, max_seq_len, diff_ratio
    )

    # Create request-level metadata
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    sparse_mask = torch.ones(batch_size, dtype=torch.bool, device="cuda")

    actual_max_len = int(seq_lens.max().item())
    page_table = torch.arange(
        batch_size * actual_max_len, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, actual_max_len)

    page_size = 1
    layer_id = test_layer_id

    # Setup memory pools
    required_pool_size = max(pool_size, batch_size * actual_max_len)
    print(f"\n[2] Creating memory pools (size={required_pool_size})...")
    device_pool, host_pool = setup_memory_pools(
        num_layers, required_pool_size, token_stride_size
    )

    # Kernel-specific setup
    print(f"\n[3] Initializing kernel-specific state...")
    context = setup_fn(
        device_pool,
        host_pool,
        prev_top_k_result,
        curr_top_k_result,
        req_pool_indices,
        seq_lens,
        page_table,
        page_size,
        layer_id,
        batch_size,
        top_k,
        num_layers,
        max_seq_len,
        actual_max_len,
    )

    # Warmup
    print(f"\n[4] Warmup ({num_warmup} iterations)...")
    for i in range(num_warmup):
        reset_fn(context, i)
        kernel_executor(context)

        if i < 3:
            torch.cuda.synchronize()
            success, mismatch, total = verify_fn(context)
            status = "✓" if success else "✗"
            print(f"   Warmup #{i+1}: {status} ({total - mismatch}/{total} correct)")

    torch.cuda.synchronize()
    print("   Warmup complete")

    # CUDA Graph Capture
    print("\n[5] Capturing CUDA Graph...")
    reset_fn(context, 0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel_executor(context)

    torch.cuda.synchronize()
    print("   CUDA Graph captured")

    # Benchmark
    print(f"\n[6] Benchmark CUDA Graph Replay ({num_iterations} iterations)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_times = []

    for i in range(num_iterations):
        reset_fn(context, i)

        # Count misses for first few iterations
        if i < 3:
            actual_misses = count_actual_misses(context)
            expected_data_mb = (
                actual_misses * context.get("item_size_bytes", 1024)
            ) / (1024 * 1024)

        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        elapsed_times.append(elapsed_time)

        success, mismatch, total = verify_fn(context)
        status = "✓" if success else "✗"

        if i < 3:
            print(
                f"   Iter {i+1}/{num_iterations}: {status} ({total - mismatch}/{total} correct) | {elapsed_time:.3f}ms | Misses: {actual_misses} | Transfer: {expected_data_mb:.2f}MB"
            )
        else:
            print(
                f"   Iter {i+1}/{num_iterations}: {status} ({total - mismatch}/{total} correct) | {elapsed_time:.3f}ms"
            )

    # Print results
    elapsed_times = np.array(elapsed_times)
    print_benchmark_results(elapsed_times, kernel_name)

    # Calculate and print bandwidth using expected miss count from setup
    if "expected_miss_count" in context and "item_size_bytes" in context:
        expected_misses = context["expected_miss_count"]
        data_per_iter_mb = (expected_misses * context["item_size_bytes"]) / (
            1024 * 1024
        )
        avg_time_ms = elapsed_times.mean()
        bandwidth_gbps = (data_per_iter_mb / 1024) / (avg_time_ms / 1000)  # GB/s

        print(f"\n[Bandwidth Analysis]")
        print(
            f"  Expected data per iteration: {data_per_iter_mb:.2f} MB ({expected_misses} cache misses)"
        )
        print(f"  Avg kernel time: {avg_time_ms:.3f} ms")
        print(f"  Effective bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(
            f"  Transfer type: {'GPU-to-GPU' if context.get('host_cache_cpu', torch.empty(0)).is_cuda else 'CPU-to-GPU (PCIe)'}"
        )
        print(
            f"  PCIe theoretical max (40 GB/s): {(data_per_iter_mb / 1024) / (40) * 1000:.3f} ms"
        )

    return elapsed_times


logger = logging.getLogger(__name__)

# ========== JIT Kernel ==========


def setup_jit_kernel(
    device_pool,
    host_pool,
    prev_top_k_result,
    curr_top_k_result,
    req_pool_indices,
    seq_lens,
    page_table,
    page_size,
    layer_id,
    batch_size,
    top_k,
    num_layers,
    max_seq_len,
    actual_max_len,
):
    """Setup state for JIT kernel."""
    from sglang.jit_kernel.sparse import load_cache_to_device_buffer_mla

    request_pool_size = max(batch_size * 2, 256)

    device_buffer_tokens = torch.full(
        (request_pool_size, num_layers, top_k), -1, dtype=torch.int32, device="cuda"
    )
    device_buffer_locs = torch.full(
        (request_pool_size, num_layers, top_k), -1, dtype=torch.int32, device="cuda"
    )
    host_cache_locs = torch.full(
        (request_pool_size, max_seq_len), -1, dtype=torch.int64, device="cuda"
    )

    # Initialize mappings
    offset = 0
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        seq_len = int(seq_lens[b].item())
        host_cache_locs[req_idx, :seq_len] = torch.arange(
            offset, offset + seq_len, dtype=torch.int64, device="cuda"
        )
        offset += seq_len

    device_offset = 0
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        device_buffer_tokens[req_idx, layer_id] = prev_top_k_result[b].to(torch.int32)
        device_buffer_locs[req_idx, layer_id] = torch.arange(
            device_offset, device_offset + top_k, dtype=torch.int32, device="cuda"
        )
        device_offset += top_k

    top_k_tokens = curr_top_k_result.to(torch.int32)
    top_k_device_locs = torch.full(
        (batch_size, top_k), -1, dtype=torch.int32, device="cuda"
    )
    bitmap = torch.full((batch_size, max_seq_len), -1, dtype=torch.int16, device="cuda")
    lru_slots = (
        torch.arange(top_k, dtype=torch.int16, device="cuda")
        .unsqueeze(0)
        .expand(request_pool_size, num_layers, -1)
        .contiguous()
    )

    # Initialize test data
    initialize_host_cache_data(
        host_pool,
        layer_id,
        host_cache_locs,
        req_pool_indices,
        seq_lens,
        batch_size,
        page_size,
        seed=42,
    )

    copy_prev_topk_to_device(
        device_pool,
        host_pool,
        device_buffer_locs,
        host_cache_locs,
        prev_top_k_result,
        curr_top_k_result,
        req_pool_indices,
        layer_id,
        page_size,
    )

    # Save initial device buffer state for reset (avoid re-copying from host during graph replay)
    initial_device_buffer = device_pool.kv_buffer[layer_id].clone()

    # Save initial device_buffer_tokens state for reset
    initial_device_buffer_tokens = device_buffer_tokens.clone()

    # Save initial lru_slots state for reset
    initial_lru_slots = lru_slots.clone()

    # Use pinned CPU memory for real PCIe transfer test
    # host_pool.kv_buffer is already pinned (pin_memory=True), GPU can access it directly
    host_cache_cpu = host_pool.kv_buffer[layer_id]  # Keep on CPU

    # For debugging: also create GPU version to compare
    # host_cache_gpu = host_pool.kv_buffer[layer_id].cuda()

    # Calculate item_size_bytes from actual tensor shape and dtype
    actual_feature_dim = host_pool.kv_buffer[layer_id].shape[-1]
    dtype_size = host_pool.dtype.itemsize
    item_size_bytes = actual_feature_dim * dtype_size

    # Calculate expected miss count
    overlap_count = sum(
        len(
            set(prev_top_k_result[b].cpu().tolist())
            & set(curr_top_k_result[b].cpu().tolist())
        )
        for b in range(batch_size)
    )
    total_tokens = batch_size * top_k
    miss_count = total_tokens - overlap_count
    miss_ratio = miss_count / total_tokens
    data_transfer_size_mb = (miss_count * item_size_bytes) / (1024 * 1024)

    kernel_block_size = 512 if top_k >= 1024 else 256
    print(f"   JIT kernel state initialized:")
    print(f"     - block_size={kernel_block_size}, item_size_bytes={item_size_bytes}")
    print(f"     - feature_dim={actual_feature_dim}, dtype_size={dtype_size}")
    print(
        f"     - Expected misses: {miss_count}/{total_tokens} ({miss_ratio*100:.1f}%)"
    )
    print(
        f"     - Expected data transfer per iteration: {data_transfer_size_mb:.2f} MB"
    )
    print(
        f"     - Host cache location: {'GPU' if host_cache_cpu.is_cuda else 'CPU (pinned)'}"
    )
    print(
        f"     - PCIe theoretical time (40 GB/s): {data_transfer_size_mb / (40 * 1024) * 1000:.3f} ms"
    )

    return {
        "kernel_fn": load_cache_to_device_buffer_mla,
        "device_pool": device_pool,
        "host_pool": host_pool,
        "host_cache_cpu": host_cache_cpu,
        "initial_device_buffer": initial_device_buffer,
        "initial_device_buffer_tokens": initial_device_buffer_tokens,
        "initial_lru_slots": initial_lru_slots,
        "device_buffer_tokens": device_buffer_tokens,
        "device_buffer_locs": device_buffer_locs,
        "host_cache_locs": host_cache_locs,
        "top_k_tokens": top_k_tokens,
        "top_k_device_locs": top_k_device_locs,
        "page_table": page_table,
        "bitmap": bitmap,
        "lru_slots": lru_slots,
        "prev_top_k_result": prev_top_k_result,
        "req_pool_indices": req_pool_indices,
        "sparse_mask": torch.ones(batch_size, dtype=torch.bool, device="cuda"),
        "seq_lens": seq_lens,
        "page_size": page_size,
        "layer_id": layer_id,
        "item_size_bytes": item_size_bytes,
        "block_size": kernel_block_size,
        "batch_size": batch_size,
        "top_k": top_k,
        "num_layers": num_layers,
        "expected_miss_count": miss_count,
    }


def reset_jit_kernel(context, iteration):
    """Reset state before each iteration for JIT kernel."""
    layer_id = context["layer_id"]

    # Reset device_buffer_tokens from saved initial state
    context["device_buffer_tokens"].copy_(context["initial_device_buffer_tokens"])

    # Reset lru_slots from saved initial state
    context["lru_slots"].copy_(context["initial_lru_slots"])

    # Reset top_k_device_locs to ensure clean state for each iteration
    context["top_k_device_locs"].fill_(-1)

    # Reset device buffer data from saved initial state
    context["device_pool"].kv_buffer[layer_id].copy_(context["initial_device_buffer"])


def execute_jit_kernel(context):
    """Execute JIT kernel."""
    context["kernel_fn"](
        top_k_tokens=context["top_k_tokens"],
        device_buffer_tokens=context["device_buffer_tokens"],
        host_cache_locs=context["host_cache_locs"],
        device_buffer_locs=context["device_buffer_locs"],
        host_cache=context["host_cache_cpu"],
        device_buffer=context["device_pool"].kv_buffer[context["layer_id"]],
        top_k_device_locs=context["top_k_device_locs"],
        page_table=context["page_table"],
        diff_map=context["bitmap"],
        req_pool_indices=context["req_pool_indices"],
        sparse_mask=context["sparse_mask"],
        seq_lens=context["seq_lens"],
        # lru_slots=context['lru_slots'],
        page_size=context["page_size"],
        layer_id=context["layer_id"],
        item_size_bytes=context["item_size_bytes"],
        block_size=context["block_size"],
        num_top_k=context["top_k"],
        hot_buffer_size=context["top_k"],
    )


def execute_jit_kernel_profiled(context):
    """Execute JIT kernel with detailed profiling."""
    # Profile reset overhead
    reset_start = torch.cuda.Event(enable_timing=True)
    reset_end = torch.cuda.Event(enable_timing=True)

    # Profile kernel execution
    kernel_start = torch.cuda.Event(enable_timing=True)
    kernel_end = torch.cuda.Event(enable_timing=True)

    reset_start.record()
    # Empty - reset is done outside
    reset_end.record()

    kernel_start.record()
    execute_jit_kernel(context)
    kernel_end.record()

    torch.cuda.synchronize()

    return {
        "kernel_time": kernel_start.elapsed_time(kernel_end),
    }


def verify_jit_kernel(context):
    """Verify correctness for JIT kernel."""
    return verify_kernel_correctness(
        context["device_pool"],
        context["host_pool"],
        context["layer_id"],
        context["batch_size"],
        context["top_k"],
        context["top_k_tokens"],
        context["top_k_device_locs"],
        context["host_cache_locs"],
        context["req_pool_indices"],
        context["page_size"],
        None,  # JIT kernel has cross_page optimization, no need to skip
    )


def count_actual_misses(context):
    """Count actual cache misses in the current iteration."""
    batch_size = context["batch_size"]
    top_k = context["top_k"]
    layer_id = context["layer_id"]

    top_k_tokens = context["top_k_tokens"]
    device_buffer_tokens = context["device_buffer_tokens"]
    req_pool_indices = context["req_pool_indices"]

    total_misses = 0
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        curr_tokens = set(top_k_tokens[b].cpu().tolist())
        prev_tokens = set(device_buffer_tokens[req_idx, layer_id].cpu().tolist())

        # Remove -1 (invalid tokens)
        curr_tokens.discard(-1)
        prev_tokens.discard(-1)

        misses = curr_tokens - prev_tokens
        total_misses += len(misses)

    return total_misses


def benchmark_jit_kernel_with_graph(
    batch_size: int,
    top_k: int,
    num_layers: int,
    max_seq_len: int,
    pool_size: int,
    token_stride_size: int,
    num_warmup: int,
    num_iterations: int,
    test_layer_id: int,
    diff_ratio: float = 0.20,
):
    """Benchmark JIT kernel using unified runner."""
    return run_unified_benchmark(
        kernel_name="JIT Kernel",
        kernel_executor=execute_jit_kernel,
        setup_fn=setup_jit_kernel,
        reset_fn=reset_jit_kernel,
        verify_fn=verify_jit_kernel,
        batch_size=batch_size,
        top_k=top_k,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        pool_size=pool_size,
        token_stride_size=token_stride_size,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        test_layer_id=test_layer_id,
        diff_ratio=diff_ratio,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark kernel performance with CUDA Graph"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--top_k", type=int, default=128, help="Top K value")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument(
        "--max_seq_len", type=int, default=500, help="Max sequence length"
    )
    parser.add_argument("--pool_size", type=int, default=256, help="Memory pool size")
    parser.add_argument(
        "--token_stride_size",
        type=int,
        default=512,
        help="Token stride size (kv_lora_rank)",
    )
    parser.add_argument(
        "--num_warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="Number of test iterations"
    )
    parser.add_argument(
        "--test_layer_id",
        type=int,
        default=4,
        help="Layer ID to test (must be < num_layers)",
    )
    parser.add_argument(
        "--diff_ratio",
        type=float,
        default=0.20,
        help="Ratio of different tokens between prev and curr top-k (0.0 to 1.0)",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="jit",
        choices=["jit"],
        help="Test mode:, jit (new jit kernel)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    if args.test_mode in ["jit"]:
        print("\n" + "=" * 60)
        print("Testing JIT Kernel")
        print("=" * 60)
        benchmark_jit_kernel_with_graph(
            batch_size=args.batch_size,
            top_k=args.top_k,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            pool_size=args.pool_size,
            token_stride_size=args.token_stride_size,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            test_layer_id=args.test_layer_id,
            diff_ratio=args.diff_ratio,
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
