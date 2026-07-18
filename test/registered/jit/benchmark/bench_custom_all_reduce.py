from __future__ import annotations

import atexit
import contextlib
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import get_benchmark_range, multigpu_bench_main
from sglang.jit_kernel.mp import register_comm_cleanup
from sglang.jit_kernel.utils import cache_once, is_arch_support_pdl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
DTYPE_ITEMSIZE = DTYPE.itemsize
MESSAGE_SIZES_KB = [2**x for x in range(2, 17)]
MESSAGE_SIZES_KB += [192, 384, 640, 768, 896, 1536, 3072]
MESSAGE_SIZES_KB.sort()
WORLD_SIZES = list(range(2, 9))
MAX_BYTES = max(MESSAGE_SIZES_KB) * 1024
# trtllm allreduce_fusion only supports these world sizes.
FI_SUPPORTED_WORLD_SIZES = (2, 4, 8)
# AOT custom_all_reduce (v1) only supports these world sizes.
AOT_SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
# jit-eager times the naive-loop dispatch (eager heuristics); jit-graph
# captures the calls in a CUDA graph (graph heuristics + pointer table).
PROVIDERS = ["nccl", "aot", "jit-eager", "jit-graph", "fi"]
WORLD_SIZES = get_benchmark_range(WORLD_SIZES, [2, 4, 8])

# ---------------------------------------------------------------------------
# Per-rank distributed init (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group() -> dist.ProcessGroup:
    _init_cpu_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_group = torch.distributed.new_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    assert isinstance(device_group, dist.ProcessGroup)
    return device_group


# ---------------------------------------------------------------------------
# Backend wrappers - each exposes:
#   .all_reduce(tensor) -> Tensor
#   .graph_context()    -> context manager wrapping cuda-graph capture
#                          (nullcontext when capture is not required)
# ---------------------------------------------------------------------------


class NCCLAllReduceBackend:
    def __init__(self) -> None:
        self.group = _init_nccl_group()

    def graph_context(self):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self.group)
        return tensor


class JITAllReduceBackend:
    def __init__(self) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        # tuned workspace sizes, capped at the sweep maximum
        self.comm = CustomAllReduceV2(_init_cpu_group(), device, max_size=MAX_BYTES)
        if self.comm.disabled:
            raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
        # keep the whole sweep on the custom-AR path: the tuned config would
        # otherwise send the largest sizes back to NCCL
        self.comm.uncap_pull_thresholds()
        register_comm_cleanup(self.comm)

    def graph_context(self):
        return self.comm.capture()

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class AOTAllReduceBackend:
    def __init__(self) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.comm = CustomAllreduce(_init_cpu_group(), device, max_size=MAX_BYTES)
        if self.comm.disabled:
            raise RuntimeError("AOT CustomAllreduce is disabled on this system")
        register_comm_cleanup(self.comm)

    def graph_context(self):
        return self.comm.capture()

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class FlashInferAllReduceBackend:
    def __init__(self) -> None:
        import flashinfer.comm as comm

        group = _init_cpu_group()
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        # Use the smallest message size as the inner hidden dim, so any
        # message in the sweep is an integer multiple of it.
        hidden_dim = 1024 * min(MESSAGE_SIZES_KB) // DTYPE_ITEMSIZE
        num_tokens = MAX_BYTES // (hidden_dim * DTYPE_ITEMSIZE)
        self._comm = comm
        self._hidden_dim = hidden_dim
        self._workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=num_tokens,
            hidden_dim=hidden_dim,
            dtype=DTYPE,
        )

    def graph_context(self):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._comm.allreduce_fusion(
            input=tensor.view(-1, self._hidden_dim),
            workspace=self._workspace,
            pattern=self._comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=is_arch_support_pdl(),
            fp32_acc=True,
        )


@cache_once
def _init_nccl_backend() -> NCCLAllReduceBackend:
    return NCCLAllReduceBackend()


@cache_once
def _init_jit_backend() -> JITAllReduceBackend:
    return JITAllReduceBackend()


@cache_once
def _init_aot_backend() -> AOTAllReduceBackend:
    return AOTAllReduceBackend()


@cache_once
def _init_fi_backend() -> FlashInferAllReduceBackend:
    return FlashInferAllReduceBackend()


BACKEND_FACTORY = {
    "nccl": _init_nccl_backend,
    "jit-eager": _init_jit_backend,
    "jit-graph": _init_jit_backend,
    "aot": _init_aot_backend,
    "fi": _init_fi_backend,
}


@cache_once
def _init_all_backends() -> None:
    """Pre-build every supported backend before any timed iteration so JIT
    compilation / IPC setup don't bleed into the first measured size.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:  # NOTE: log some verbose info on initialization
        logging.basicConfig(level=logging.INFO)

    world_size = dist.get_world_size(_init_cpu_group())
    factories = dict(BACKEND_FACTORY)
    if world_size not in AOT_SUPPORTED_WORLD_SIZES:
        factories.pop("aot")
    if world_size not in FI_SUPPORTED_WORLD_SIZES:
        factories.pop("fi")
    for fn in factories.values():
        fn()

    # reset level to warning
    logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@marker.parametrize("message_KB", MESSAGE_SIZES_KB)
@marker.benchmark("provider", PROVIDERS)
def benchmark(message_KB: int, provider: str):
    cpu_group = _init_cpu_group()
    gpu_group = _init_nccl_group()
    world_size = dist.get_world_size(cpu_group)
    if provider == "fi" and world_size not in FI_SUPPORTED_WORLD_SIZES:
        marker.skip(
            f"flashinfer trtllm allreduce_fusion needs world_size in "
            f"{FI_SUPPORTED_WORLD_SIZES}"
        )
    if provider == "aot" and world_size not in AOT_SUPPORTED_WORLD_SIZES:
        marker.skip(
            f"AOT custom_all_reduce needs world_size in " f"{AOT_SUPPORTED_WORLD_SIZES}"
        )
    _init_all_backends()
    backend = BACKEND_FACTORY[provider]()
    message_bytes = message_KB * 1024
    numel = message_bytes // DTYPE_ITEMSIZE
    device_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{device_id}")
    x = torch.randn(numel, dtype=DTYPE, device=device)
    ctx_fn = backend.graph_context if not provider.endswith("eager") else None
    # Bandwidth-equivalent bytes moved by a ring all-reduce per rank.
    effective_bytes = int(x.nbytes * 2 * (world_size - 1) / world_size)
    return marker.do_bench(
        backend.all_reduce,
        input_args=(x,),
        graph_context_fn=ctx_fn,
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        # all-reduce is in-place w.r.t. its argument; explicit footprint
        # captures the cross-GPU traffic instead.
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=effective_bytes,
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=WORLD_SIZES,
        main_fn=benchmark.run,
    )
