"""
Benchmark for KV transfer Triton kernels.

Measures bandwidth achieved when gathering/scattering KV cache data
between GPU and pinned CPU memory.
"""

import argparse

import torch

from sglang.srt.layers.attention.triton_ops.kv_transfer import (
    gather_kv_to_pinned_all_layers,
    scatter_kv_with_staging_all_layers,
)


def create_pointer_tensors(k_buffers, v_buffers):
    """Helper to create pointer tensors and get strides."""
    k_data_ptrs = torch.tensor(
        [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device="cuda"
    )
    v_data_ptrs = torch.tensor(
        [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device="cuda"
    )
    slot_stride = k_buffers[0].stride(0)
    head_stride = k_buffers[0].stride(1)
    return k_data_ptrs, v_data_ptrs, slot_stride, head_stride


def benchmark_cuda_memcpy(
    size_mb: float,
    dtype: torch.dtype,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark raw CUDA memcpy in both directions.
    Returns bandwidth for D2H (GPU -> CPU) and H2D (CPU -> GPU).
    """
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    num_elements = int(size_mb * 1e6 / bytes_per_element)
    total_bytes = num_elements * bytes_per_element

    gpu_tensor = torch.randn(num_elements, dtype=dtype, device="cuda")
    cpu_tensor = torch.empty(num_elements, dtype=dtype, device="cpu", pin_memory=True)

    # D2H benchmark
    for _ in range(warmup):
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        end_events[i].record()
    torch.cuda.synchronize()

    d2h_times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    d2h_avg_time_ms = sum(d2h_times_ms) / len(d2h_times_ms)
    d2h_avg_bw = (total_bytes / 1e9) / (d2h_avg_time_ms / 1000)

    # H2D benchmark
    for _ in range(warmup):
        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
        torch.cuda.synchronize()

    for i in range(rep):
        start_events[i].record()
        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
        end_events[i].record()
    torch.cuda.synchronize()

    h2d_times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    h2d_avg_time_ms = sum(h2d_times_ms) / len(h2d_times_ms)
    h2d_avg_bw = (total_bytes / 1e9) / (h2d_avg_time_ms / 1000)

    del gpu_tensor, cpu_tensor
    torch.cuda.empty_cache()

    return {
        "size_mb": size_mb,
        "d2h_avg_time_ms": d2h_avg_time_ms,
        "d2h_avg_bandwidth_gbs": d2h_avg_bw,
        "h2d_avg_time_ms": h2d_avg_time_ms,
        "h2d_avg_bandwidth_gbs": h2d_avg_bw,
    }


def benchmark_kv_transfer(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 5,
    rep: int = 20,
) -> dict:
    """
    Benchmark gather and scatter kernels with configurable sizes.
    """
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    transfer_bytes = num_layers * 2 * num_tokens * num_heads * head_dim * bytes_per_element

    print(f"\n  Configuration:")
    print(f"    num_layers={num_layers}, num_tokens={num_tokens}")
    print(f"    num_heads={num_heads}, head_dim={head_dim}")
    print(f"    total_slots={total_slots:,}")
    print(f"    transfer size = {transfer_bytes / 1e9:.2f} GB ({transfer_bytes / 1e6:.1f} MB)")

    # Create KV buffers
    print(f"\n  Allocating KV pool...")
    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    print(f"  Pool allocated. Free GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Create slot indices - random access pattern
    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Allocate pinned buffer
    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_buffer = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    # Contiguous GPU buffer for baselines
    contiguous_gpu = torch.randn(output_size, dtype=dtype, device="cuda")

    results = {}

    # =========================================================================
    # D2H memcpy baseline
    # =========================================================================
    print("\n  Benchmarking D2H memcpy (GPU -> pinned CPU)...")
    for _ in range(warmup):
        pinned_buffer.copy_(contiguous_gpu, non_blocking=False)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        pinned_buffer.copy_(contiguous_gpu, non_blocking=False)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    d2h_time_ms = sum(times_ms) / len(times_ms)
    d2h_bw = (transfer_bytes / 1e9) / (d2h_time_ms / 1000)
    results["d2h_memcpy"] = {"time_ms": d2h_time_ms, "bandwidth_gbs": d2h_bw}
    print(f"    D2H memcpy: {d2h_time_ms:.1f} ms, {d2h_bw:.2f} GB/s")

    # =========================================================================
    # H2D memcpy baseline
    # =========================================================================
    print("\n  Benchmarking H2D memcpy (pinned CPU -> GPU)...")
    for _ in range(warmup):
        contiguous_gpu.copy_(pinned_buffer, non_blocking=False)
        torch.cuda.synchronize()

    for i in range(rep):
        start_events[i].record()
        contiguous_gpu.copy_(pinned_buffer, non_blocking=False)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    h2d_time_ms = sum(times_ms) / len(times_ms)
    h2d_bw = (transfer_bytes / 1e9) / (h2d_time_ms / 1000)
    results["h2d_memcpy"] = {"time_ms": h2d_time_ms, "bandwidth_gbs": h2d_bw}
    print(f"    H2D memcpy: {h2d_time_ms:.1f} ms, {h2d_bw:.2f} GB/s")

    del contiguous_gpu

    # =========================================================================
    # Gather benchmark
    # =========================================================================
    print("\n  Benchmarking gather (scattered GPU -> pinned CPU)...")

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    for _ in range(warmup):
        gather_kv_to_pinned_all_layers(
            k_data_ptrs=k_data_ptrs,
            v_data_ptrs=v_data_ptrs,
            slot_indices=slot_indices,
            pinned_output=pinned_buffer,
            head_start=0,
            num_heads_to_gather=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
        )

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        gather_kv_to_pinned_all_layers(
            k_data_ptrs=k_data_ptrs,
            v_data_ptrs=v_data_ptrs,
            slot_indices=slot_indices,
            pinned_output=pinned_buffer,
            head_start=0,
            num_heads_to_gather=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
        )
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    gather_time_ms = sum(times_ms) / len(times_ms)
    gather_bw = (transfer_bytes / 1e9) / (gather_time_ms / 1000)
    results["gather"] = {"time_ms": gather_time_ms, "bandwidth_gbs": gather_bw}
    print(f"    Gather: {gather_time_ms:.1f} ms, {gather_bw:.2f} GB/s ({gather_bw/d2h_bw*100:.0f}% of D2H)")

    # =========================================================================
    # Scatter benchmark
    # =========================================================================
    print("\n  Benchmarking scatter (pinned CPU -> scattered GPU)...")

    pinned_input = torch.randn(output_size, dtype=dtype, device="cpu", pin_memory=True)

    k_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    k_data_ptrs_dst, v_data_ptrs_dst, dst_slot_stride, dst_head_stride = create_pointer_tensors(k_buffers_dst, v_buffers_dst)

    for _ in range(warmup):
        scatter_kv_with_staging_all_layers(
            pinned_input=pinned_input,
            k_data_ptrs=k_data_ptrs_dst,
            v_data_ptrs=v_data_ptrs_dst,
            slot_indices=slot_indices,
            head_start=0,
            num_heads_to_scatter=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            dst_slot_stride=dst_slot_stride,
            dst_head_stride=dst_head_stride,
        )

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        scatter_kv_with_staging_all_layers(
            pinned_input=pinned_input,
            k_data_ptrs=k_data_ptrs_dst,
            v_data_ptrs=v_data_ptrs_dst,
            slot_indices=slot_indices,
            head_start=0,
            num_heads_to_scatter=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            dst_slot_stride=dst_slot_stride,
            dst_head_stride=dst_head_stride,
        )
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    scatter_time_ms = sum(times_ms) / len(times_ms)
    scatter_bw = (transfer_bytes / 1e9) / (scatter_time_ms / 1000)
    results["scatter"] = {"time_ms": scatter_time_ms, "bandwidth_gbs": scatter_bw}
    print(f"    Scatter: {scatter_time_ms:.1f} ms, {scatter_bw:.2f} GB/s ({scatter_bw/h2d_bw*100:.0f}% of H2D)")

    # Clean up
    del k_buffers, v_buffers, k_buffers_dst, v_buffers_dst
    del pinned_buffer, pinned_input, slot_indices
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV transfer kernels")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=20, help="Benchmark repetitions")
    parser.add_argument("--num-layers", type=int, default=92, help="Number of layers")
    parser.add_argument("--num-tokens", type=int, default=32768, help="Number of tokens")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--total-slots", type=int, default=None, help="Total slots (default: 4x num_tokens)")
    args = parser.parse_args()

    print("=" * 80)
    print(" KV Transfer Kernel Benchmark")
    print(" Single-kernel gather/scatter (O(1) GPU memory overhead)")
    print("=" * 80)

    # Get GPU info
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {props.name}")

    total_slots = args.total_slots or (args.num_tokens * 4)
    transfer_size_gb = args.num_layers * 2 * args.num_tokens * args.num_heads * args.head_dim * 2 / 1e9

    print("\n" + "=" * 80)
    print(" KV Transfer Benchmark")
    print(f" {args.num_layers} layers, {args.num_tokens} tokens, {args.num_heads} heads, {args.head_dim} head_dim")
    print(f" Transfer size: {transfer_size_gb:.2f} GB")
    print("=" * 80)

    results = benchmark_kv_transfer(
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        total_slots=total_slots,
        warmup=args.warmup,
        rep=args.rep,
    )

    # Summary
    print("\n" + "=" * 80)
    print(f" Summary ({transfer_size_gb:.2f} GB transfers)")
    print("=" * 80)

    d2h_bw = results["d2h_memcpy"]["bandwidth_gbs"]
    h2d_bw = results["h2d_memcpy"]["bandwidth_gbs"]

    print(f"\n PCIe Baselines (contiguous transfers):")
    print(f" {'Method':<25} {'Time (ms)':<12} {'BW (GB/s)':<12}")
    print("-" * 50)
    print(f" {'D2H memcpy (GPU->CPU)':<25} {results['d2h_memcpy']['time_ms']:<12.1f} {d2h_bw:<12.2f}")
    print(f" {'H2D memcpy (CPU->GPU)':<25} {results['h2d_memcpy']['time_ms']:<12.1f} {h2d_bw:<12.2f}")

    print(f"\n Kernel Performance:")
    print(f" {'Method':<25} {'Time (ms)':<12} {'BW (GB/s)':<12} {'Efficiency':<12}")
    print("-" * 65)
    gather_eff = results['gather']['bandwidth_gbs'] / d2h_bw * 100
    scatter_eff = results['scatter']['bandwidth_gbs'] / h2d_bw * 100
    print(f" {'Gather (GPU->CPU)':<25} {results['gather']['time_ms']:<12.1f} {results['gather']['bandwidth_gbs']:<12.2f} {gather_eff:.0f}% of D2H")
    print(f" {'Scatter (CPU->GPU)':<25} {results['scatter']['time_ms']:<12.1f} {results['scatter']['bandwidth_gbs']:<12.2f} {scatter_eff:.0f}% of H2D")

    print(f"\n Key features:")
    print(f"   - Single kernel launch for all {args.num_layers} layers")
    print(f"   - O(1) extra GPU memory (just {args.num_layers * 16 / 1024:.1f} KB for pointer tensors)")
    print(f"   - No staging buffers needed")


if __name__ == "__main__":
    main()
