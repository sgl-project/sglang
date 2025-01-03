# usage: torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_id=none --master_addr=127.0.0.1 --master_port=23456 test/test_gemm_rs.py 2048 10240 40960
import argparse
import datetime
import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from typing import Union

import flux
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

# from flux.gemm_rs_sm80 import get_intra_node_pg_group
import torch.nn as nn
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
# WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
WORLD_SIZE = 2
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE
torch.cuda.set_device(LOCAL_RANK)

os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
np.random.seed(3 + RANK)

torch.distributed.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=RANK,
    timeout=datetime.timedelta(seconds=1800),
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
print = partial(print, flush=True)


class PerfResult:
    def __init__(
        self, name: str, output: torch.Tensor, gemm_time_ms: float, comm_time_ms: float
    ) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: gemm {self.gemm_time_ms:.3f} ms, comm {self.comm_time_ms:.3f} ms"
            f", total {self.gemm_time_ms + self.comm_time_ms:.3f} ms"
        )


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    warmup: int,
    iters: int,
    transpose_weight: bool = False,
    input_scale: Union[None, torch.Tensor] = None,
    weight_scale: Union[None, torch.Tensor] = None,
):
    torch.distributed.barrier()

    warmup_iters = warmup
    output_dtype = input.dtype if not is_fp8 else torch.bfloat16
    m = input.size(0)
    n = weight.size(0)
    w = weight

    full_output = torch.zeros(
        [m, n],
        dtype=output_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    output = torch.zeros(
        [m // WORLD_SIZE, n],
        dtype=output_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    op = (
        flux.GemmOnly(
            input_dtype=input.dtype,
            output_dtype=output_dtype,
            transpose_weight=transpose_weight,
            use_fp8_gemm=is_fp8,
        )
        # None
        if is_fp8
        else None
    )

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.distributed.barrier()

    for i in range(total_iters):
        start_events[i].record()
        # if is_fp8:
        #     full_output = op.forward(
        #         input,
        #         w,
        #         bias=bias,
        #         output_buf=full_output,
        #         input_scale=input_scale,
        #         weight_scale=weight_scale,
        #         output_scale=None,
        #         fast_accum=False,
        #     )
        # else:
        full_output = torch.matmul(input, weight.t())
        if bias is not None:
            full_output += bias
        gemm_end_events[i].record()
        # print(f'torch: output dimension: {output.shape}, full_output: {full_output.shape}')
        dist.reduce_scatter_tensor(output, full_output, group=TP_GROUP)
        end_events[i].record()

    gemm_times = []
    comm_times = []
    for i in range(total_iters):
        gemm_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(gemm_end_events[i]) / 1000)
            comm_times.append(gemm_end_events[i].elapsed_time(end_events[i]) / 1000)
    # print(gemm_times)
    # print(comm_times)
    gemm_time = sum(gemm_times) / iters * 1000
    comm_time = sum(comm_times) / iters * 1000
    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        gemm_time_ms=gemm_time,
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_te(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    transpose_weight: bool,
    fuse_reduction: bool,
    warmup: int,
    iters: int,
    runs_per_node: bool = False,
    input_scale: Union[None, torch.Tensor] = None,
    weight_scale: Union[None, torch.Tensor] = None,
):
    dist.barrier()
    m = input.size(0)
    w = weight
    n = w.size(0)

    output_dtype = input.dtype if not is_fp8 else torch.bfloat16

    full_output = torch.zeros(
        [m, n],
        dtype=output_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    output = torch.zeros(
        [m // WORLD_SIZE, n],
        dtype=output_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    dist.barrier()

    for i in range(total_iters):
        start_events[i].record()
        workspace_size = input.size(0) * weight.size(
            0
        )  # pre-allocate memory for tex.gemm()
        workspace = torch.empty((workspace_size,), dtype=torch.float16, device="cuda")
        output_gemm = tex.gemm(
            weight,
            input,
            # input, weight,
            dtype=input.dtype,
            workspace=workspace,
            gelu=False,
            grad=False,
            accumulate=False,
        )
        full_output = output_gemm[0]
        expected_shape = (output.size(0) * WORLD_SIZE, output.size(1))
        if full_output.size() != expected_shape:
            raise ValueError(
                f"full_output shape {full_output.size()} does not match the expected shape {expected_shape}, input size {input.size()} and weight.t() size {weight.t().size()} and weight size {weight.size()} "
            )
        if bias is not None:
            full_output += bias
        gemm_end_events[i].record()
        # print(f'te: output dimension: {output.shape}, full_output: {full_output.shape}')
        dist.reduce_scatter_tensor(output, full_output, group=TP_GROUP)
        end_events[i].record()
    # torch.cuda.current_stream().synchronize()

    gemm_times = []
    comm_times = []
    for i in range(total_iters):
        gemm_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(gemm_end_events[i]) / 1000)
            comm_times.append(gemm_end_events[i].elapsed_time(end_events[i]) / 1000)
    # print(gemm_times)
    # print(comm_times)
    gemm_time = sum(gemm_times) / iters * 1000
    comm_time = sum(comm_times) / iters * 1000

    return PerfResult(
        name=f"te  #{TP_GROUP.rank()}",
        output=output,
        gemm_time_ms=gemm_time,
        comm_time_ms=comm_time,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="dump torch.profiler.profile",
    )
    parser.add_argument(
        "--transpose_weight",
        default=False,
        action="store_true",
        help="whether to transpose weight",
    )
    parser.add_argument(
        "--fuse_reduction",
        default=False,
        action="store_true",
        help="fuse reduction to gemm",
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether have bias"
    )
    parser.add_argument(
        "--runs_per_node", default=False, action="store_true", help="multi-stage gemm"
    )
    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _rand(shape, dtype):
    if flux.util.is_fp8_dtype(dtype):
        tensor: torch.Tensor = (
            (-2 * torch.rand(shape, dtype=torch.float16, device="cuda"))
            / 100
            * (TP_GROUP.rank() + 1)
        )
        with flux.util.with_torch_deterministic(False):
            return tensor.to(dtype)
    return (
        (-2 * torch.rand(shape, dtype=dtype).cuda() + 1) / 100 * (TP_GROUP.rank() + 1)
    )


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()

    # print("before flux shm initialization")
    # if NNODES > 1 and args.runs_per_node:
    #     flux.init_flux_shm(get_intra_node_pg_group(TP_GROUP, NNODES))
    # else:
    #     flux.init_flux_shm(TP_GROUP)
    # torch.cuda.synchronize()
    # print("after flux shm initialization")

    dtype = DTYPE_MAP[args.dtype]
    is_fp8 = False  # tobe supported in the future
    if args.transpose_weight and is_fp8:
        raise ValueError("FP8 GEMM does not support RRR layout")

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    # input: [M, K], weight: [N, K]
    input = _rand((args.M, local_K), dtype=dtype)
    weight = _rand((args.N, local_K), dtype=dtype)

    input_scale, weight_scale = None, None

    bias = None
    bias_dtype = dtype if not is_fp8 else torch.bfloat16  # always BF16 for FP8 matmul
    if args.has_bias:
        bias = (
            torch.rand((args.M, args.N), dtype=dtype).cuda()
            / 10
            * (TP_GROUP.rank() + 1)
        )

    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        if args.profile
        else nullcontext()
    )
    with ctx:
        perf_res_te = perf_te(
            input,
            weight,
            bias,
            args.transpose_weight,
            args.fuse_reduction,
            args.warmup,
            args.iters,
            args.runs_per_node,
            input_scale,
            weight_scale,
        )
        perf_res_torch = perf_torch(
            input,
            weight,
            bias,
            args.warmup,
            args.iters,
            args.transpose_weight,
            input_scale,
            weight_scale,
        )

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    if TP_GROUP.rank() == 0:
        print(
            f"SOL time for GEMM(M={args.M},N={args.N},K={args.K},TP={TP_GROUP.size()}):"
        )
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_torch)
        torch.distributed.barrier()
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_te)
        torch.distributed.barrier()
    torch.distributed.barrier()

    te_output = perf_res_te.output
    torch_output = perf_res_torch.output
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    # flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
