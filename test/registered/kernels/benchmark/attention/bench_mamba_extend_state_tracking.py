import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE, get_benchmark_range
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    copy_mamba_state_extend_rows,
    derive_track_ssm_copy_flags,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")

LOCAL_HV_VALUES = [2, 4, 8, 16, 24, 48]
V_DIM = 128
K_DIM = 128
TRACKED_ROWS_LIST = get_benchmark_range(
    full_range=[1, 8, 16, 32, 64, 128, 256],
    ci_range=[32, 128],
)
DTYPE = torch.bfloat16
NUM_LAYERS = 64


def _make_h_to_state_case(local_hv: int, tracked_rows: int):
    src_rows = max(512, tracked_rows * 2 + 32)
    dst_rows = max(512, tracked_rows * 2 + 32)
    shape = (local_hv, V_DIM, K_DIM)
    h = torch.randn((src_rows, *shape), device=DEFAULT_DEVICE, dtype=torch.float32)
    ssm_states = torch.randn((dst_rows, *shape), device=DEFAULT_DEVICE, dtype=DTYPE)
    src_indices = torch.randperm(src_rows, device=DEFAULT_DEVICE, dtype=torch.int32)[
        :tracked_rows
    ].contiguous()
    dst_indices = (
        torch.randperm(dst_rows - NUM_LAYERS, device=DEFAULT_DEVICE, dtype=torch.int32)[
            :tracked_rows
        ].contiguous()
        + NUM_LAYERS
    )
    empty = torch.empty((0,), device=DEFAULT_DEVICE, dtype=torch.int32)
    return h, ssm_states, src_indices, dst_indices, empty, empty, src_rows, dst_rows


def _make_state_to_state_case(local_hv: int, tracked_rows: int):
    rows = max(512, tracked_rows * 3 + 64)
    shape = (local_hv, V_DIM, K_DIM)
    ssm_states = torch.randn((rows, *shape), device=DEFAULT_DEVICE, dtype=DTYPE)
    span = rows - NUM_LAYERS
    final_src = (
        torch.randperm(span // 2, device=DEFAULT_DEVICE, dtype=torch.int32)[
            :tracked_rows
        ].contiguous()
        + NUM_LAYERS
    )
    final_dst = (
        torch.randperm(span // 2, device=DEFAULT_DEVICE, dtype=torch.int32)[
            :tracked_rows
        ].contiguous()
        + NUM_LAYERS
        + span // 2
    )
    empty = torch.empty((0,), device=DEFAULT_DEVICE, dtype=torch.int32)
    h = torch.randn(
        (tracked_rows + 16, *shape), device=DEFAULT_DEVICE, dtype=torch.float32
    )
    return h, ssm_states, empty, empty, final_src, final_dst, tracked_rows + 16, rows


def _measure_alloc_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(0, peak - before)


def _run_integrated(
    h,
    ssm_states,
    track_ssm_h_src,
    track_ssm_h_dst,
    track_ssm_final_src,
    track_ssm_final_dst,
    *,
    h_indices_trusted: bool,
    final_indices_trusted: bool,
    final_state_disjoint: bool,
):
    copy_mamba_state_extend_rows(
        h,
        ssm_states,
        track_ssm_h_src,
        track_ssm_h_dst,
        track_ssm_final_src,
        track_ssm_final_dst,
        h_indices_trusted=h_indices_trusted,
        final_indices_trusted=final_indices_trusted,
        final_state_disjoint=final_state_disjoint,
    )


def _run_advanced(
    h,
    ssm_states,
    track_ssm_h_src,
    track_ssm_h_dst,
    track_ssm_final_src,
    track_ssm_final_dst,
):
    if track_ssm_h_src.numel() > 0:
        ssm_states[track_ssm_h_dst] = h[track_ssm_h_src].to(
            ssm_states.dtype, copy=False
        )
    if track_ssm_final_src.numel() > 0:
        ssm_states[track_ssm_final_dst] = ssm_states[track_ssm_final_src]


def _bench_case(case_name: str, local_hv: int, tracked_rows: int, make_case):
    h, ssm_states, h_src, h_dst, final_src, final_dst, h_rows, pool_rows = make_case(
        local_hv, tracked_rows
    )
    h_trusted, final_trusted, final_disjoint = derive_track_ssm_copy_flags(
        h_src,
        h_dst,
        final_src,
        final_dst,
        h_state_rows=h_rows,
        pool_rows=pool_rows,
    )

    for provider in ("integrated_direct", "advanced_indexing"):

        def fn():
            if provider == "integrated_direct":
                _run_integrated(
                    h,
                    ssm_states,
                    h_src,
                    h_dst,
                    final_src,
                    final_dst,
                    h_indices_trusted=h_trusted,
                    final_indices_trusted=final_trusted,
                    final_state_disjoint=final_disjoint,
                )
            else:
                _run_advanced(h, ssm_states, h_src, h_dst, final_src, final_dst)

        median_ms = triton.testing.do_bench(
            fn, warmup=25, rep=200, return_mode="median"
        )
        median_us = median_ms * 1000.0
        alloc_bytes = _measure_alloc_bytes(fn)
        print(
            f"{case_name},{provider},{local_hv},{tracked_rows},{median_us:.3f},{alloc_bytes}"
        )


def benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required for this benchmark.")
    dev = torch.device(DEFAULT_DEVICE)
    props = torch.cuda.get_device_properties(dev)
    print(f"# device={props.name}")
    print(f"# torch={torch.__version__}")
    print(f"# triton={triton.__version__}")
    print("case,provider,local_hv,tracked_rows,median_us,alloc_bytes")

    for local_hv in LOCAL_HV_VALUES:
        for tracked_rows in TRACKED_ROWS_LIST:
            _bench_case(
                "h_to_state",
                local_hv,
                tracked_rows,
                _make_h_to_state_case,
            )
            _bench_case(
                "state_to_state_disjoint",
                local_hv,
                tracked_rows,
                _make_state_to_state_case,
            )


if __name__ == "__main__":
    benchmark()
