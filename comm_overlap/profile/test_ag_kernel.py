import argparse
import datetime
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
# WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
WORLD_SIZE = 2
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

torch.cuda.set_device(LOCAL_RANK)
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=2)
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


class PerfResult:
    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.output = output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.time3 = time3
        self.gemm_only_time_ms = gemm_only_time_ms

    def __repr__(self) -> str:
        if self.gemm_only_time_ms == 0.0:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms"
            )
        else:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms, {self.time3} {self.gemm_only_time_ms:.3f} ms"
            )


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_fp8: bool,
    warmup: int,
    iters: int,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    # All gather input tensors from all gpus
    full_input = torch.zeros(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        full_input = full_input.to(torch.bfloat16)

    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)

    torch.distributed.barrier()
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(total_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()

        output = alpha_scale * torch.matmul(full_input, weight.t())
        if is_fp8:
            output = output.to(torch.bfloat16)
        if bias is not None:
            output += bias
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(total_iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(
                start_events[i].elapsed_time(allgather_end_events[i]) / 1000
            )
            gemm_times.append(
                allgather_end_events[i].elapsed_time(end_events[i]) / 1000
            )

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_te(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_fp8: bool,
    warmup: int,
    iters: int,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    dist.barrier()
    # Initialize output buffer for AllGather
    full_input = torch.zeros(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        full_input = full_input.to(torch.bfloat16)

    dist.all_gather_into_tensor(full_input, input, group=TP_GROUP)

    dist.barrier()
    # Warmup and timing setup
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(warmup + iters)]
    allgather_end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(warmup + iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(warmup + iters)]

    dist.barrier()

    for i in range(total_iters):
        start_events[i].record()
        # AllGather operation
        dist.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()

        workspace_size = M * weight.size(0)  # pre-allocate memory for tex.gemm()
        workspace = torch.empty((workspace_size,), dtype=torch.float16, device="cuda")
        # GEMM operation using TransformerEngine
        output_gemm = tex.gemm(
            full_input,
            weight.t(),
            dtype=input.dtype,
            workspace=workspace,
            gelu=False,
            grad=False,
            accumulate=False,
        )
        output = output_gemm[0]
        if is_fp8:
            output = output.to(torch.bfloat16)
        if bias is not None:
            output += bias
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(warmup, warmup + iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(
                start_events[i].elapsed_time(allgather_end_events[i]) / 1000
            )
            gemm_times.append(
                allgather_end_events[i].elapsed_time(end_events[i]) / 1000
            )

    comm_time = sum(comm_times) / len(comm_times) * 1000
    gemm_time = sum(gemm_times) / len(gemm_times) * 1000

    return PerfResult(
        name=f"te #{TP_GROUP.rank()}",
        output=output,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="dump torch.profiler.profile",
    )
    parser.add_argument(
        "--local_copy",
        dest="local_copy",
        action=argparse.BooleanOptionalAction,
        help="perform local copy",
        default=True,
    )
    parser.add_argument(
        "--gather_output",
        default=False,
        action="store_true",
        help="output gather results",
    )
    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether have bias"
    )
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--ring_mode",
        default=-1,
        type=int,
        help="ring mode. -1 for auto detect. 0 for all-to-all, 1 for 1d ring. 2 for 2d ring. 3 for custom ring.",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}

if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()

    torch.cuda.synchronize()

    dtype = DTYPE_MAP[args.dtype]
    is_fp8 = False  # to be supported in the future
    if args.transpose_weight and is_fp8:
        raise ValueError("FP8 GEMM does not support RRR layout")

    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_M = args.M // TP_GROUP.size()
    local_N = args.N // TP_GROUP.size()

    # input: [M, K], weight: [N, K]
    input = None
    weight = None
    input = (
        (-2 * torch.rand((local_M, args.K), dtype=dtype).cuda() + 1)
        / 100
        * (TP_GROUP.rank() + 1)
    )
    weight = (
        (-2 * torch.rand((local_N, args.K), dtype=dtype).cuda() + 1)
        / 100
        * (TP_GROUP.rank() + 1)
    )
    input_scale = None
    weight_scale = None
    bias = None
    bias = (
        torch.rand((args.M, local_N), dtype=dtype).cuda() / 10 * (TP_GROUP.rank() + 1)
        if args.has_bias
        else None
    )

    torch.distributed.barrier()

    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        if args.profile
        else nullcontext()
    )

    with ctx:
        perf_res_torch = perf_torch(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            is_fp8,
            args.warmup,
            args.iters,
        )
        perf_res_te = perf_te(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            is_fp8,
            args.warmup,
            args.iters,
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
    torch.distributed.barrier()

    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_te)
        torch.distributed.barrier()
    torch.distributed.barrier()

    torch_output = perf_res_torch.output
    # flux_output = perf_res_flux.output
    torch.distributed.barrier()
    torch.cuda.synchronize()
