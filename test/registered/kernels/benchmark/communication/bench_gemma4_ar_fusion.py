"""One Gemma-4 layer site, fused (NVLS all-reduce + norm -> residual add -> norm ->
per-128 FP8 quant in one kernel) against the stock chain (the TP all-reduce
the model runs today: CustomAllReduceV2 within its tuned size bands, NCCL
above them; then gemma_rmsnorm_residual_scalar + RMSNorm +
per_token_group_quant_fp8), per token count. ``fused-nocopy`` leaves out the
staging copy into the symmetric buffer, the cost a producer writing straight
into it would save. Multi-GPU only; relaunches under torchrun.

The per-TP-size defaults of DEFAULT_MIN_TOKENS
(srt/layers/communication/gemma4_ar_fusion.py) come from this sweep.
"""

from __future__ import annotations

import atexit
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import get_benchmark_range, multigpu_bench_main
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import gemma4_ar_fusion
from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_rmsnorm_residual_scalar
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)

HIDDEN = 5376  # Gemma-4 31B
EPS = 1e-6
GROUP = gemma4_ar_fusion.GROUP
NUM_TOKENS = get_benchmark_range([128, 256, 512, 1024, 2048, 4096, 8192], [512, 4096])
MAX_TOKENS = max(NUM_TOKENS)
PROVIDERS = ["stock", "fused", "fused-nocopy"]


@cache_once
def _init():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    # The production configuration: tuned size bands and workspace sizes. The
    # fused kernel borrows this plane's barriers, as it does in the model.
    comm = CustomAllReduceV2(coord.cpu_group, torch.device(f"cuda:{local_rank}"))
    assert not comm.disabled and comm.has_multicast, "needs NVLink multicast"
    gemma4_ar_fusion.register_comm(lambda: comm.obj, world_size=comm.world_size)
    register_comm_cleanup(comm)
    return coord, comm


def _stock_all_reduce(comm: CustomAllReduceV2, group, x: torch.Tensor) -> torch.Tensor:
    """What GroupCoordinator.all_reduce runs for this size."""
    if comm.should_custom_ar(x):
        return comm.custom_all_reduce(x)
    dist.all_reduce(x, group=group)
    return x


def _symm(shape, dtype, group_name):
    t = _SymmetricMemory.empty_strided_p2p(
        shape, [shape[1], 1], dtype, torch.device("cuda"), group_name
    )
    handle = _SymmetricMemory.rendezvous(t)
    assert handle is not None and handle.multicast_ptr != 0
    t.zero_()
    return t, int(handle.multicast_ptr)


@cache_once
def _buffers():
    name = _init()[0].cpu_group.group_name
    return (
        _symm((MAX_TOKENS, HIDDEN), torch.bfloat16, name),
        _symm((MAX_TOKENS, HIDDEN), torch.float8_e4m3fn, name),
        _symm((MAX_TOKENS, HIDDEN // GROUP), torch.float32, name),
    )


@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, provider: str):
    coord, comm = _init()
    world_size, rank = coord.world_size, coord.rank
    partial = torch.randn(num_tokens, HIDDEN, device="cuda").bfloat16()
    residual = torch.randn(num_tokens, HIDDEN, device="cuda").bfloat16()
    w_a = torch.ones(HIDDEN, device="cuda").bfloat16()
    w_b = torch.ones(HIDDEN, device="cuda").bfloat16()
    scalar = torch.ones(1, device="cuda")

    if provider == "stock":

        def fn(x):
            x = _stock_all_reduce(comm, coord.device_group, x)
            r = gemma_rmsnorm_residual_scalar(x, w_a, residual, scalar, EPS)
            y = torch.nn.functional.rms_norm(r, (HIDDEN,), w_b, EPS)
            return per_token_group_quant_fp8(y, GROUP)

    else:
        (p_t, p_mc), (o_t, o_mc), (s_t, s_mc) = _buffers()
        per_rank = (num_tokens + world_size - 1) // world_size
        row0 = min(rank * per_rank, num_tokens)
        rows = max(0, min(num_tokens, row0 + per_rank) - row0)
        mine = residual[row0 : row0 + rows].clone()
        copy = provider == "fused"

        def fn(x):
            if copy:
                p_t[:num_tokens].copy_(x)  # the staging copy the model path pays
            gemma4_ar_fusion.site(
                p_t[:num_tokens],
                p_mc,
                o_t[:num_tokens],
                o_mc,
                s_t[:num_tokens],
                s_mc,
                mine,
                w_a,
                w_b,
                scalar,
                EPS,
                row0,
                rows,
                False,
                world_size=world_size,
            )
            return o_t[:num_tokens], s_t[:num_tokens]

    return marker.do_bench(
        fn,
        input_args=(partial,),
        # the custom all-reduce registers its graph inputs during capture
        graph_context_fn=comm.capture,
        sync_multigpu_fn=lambda: dist.barrier(coord.device_group),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=int(partial.nbytes * 2 * (world_size - 1) / world_size),
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__, file=__file__, num_gpus=(2, 4, 8), main_fn=benchmark.run
    )
