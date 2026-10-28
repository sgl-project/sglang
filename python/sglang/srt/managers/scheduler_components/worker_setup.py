"""Bringing up the device and the model workers for the scheduler: the steps
that run once, before the KV cache and the scheduler's components are built."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.environ import envs
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.managers.scheduler_components.bundles import Workers
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.model_executor.runner_utils.pool import prewarm_graph_pool_borrow
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_is_npu = is_npu()


def _prewarm_hccl_group(device, group, device_module):
    warmup_tensor = torch.zeros(1, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(warmup_tensor, group=group)
    device_module.synchronize()


def maybe_prewarm_hccl_dp(hf_config) -> None:
    if not (_is_npu and is_deepseek_v4(hf_config)):
        return

    parallel = get_parallel()
    tp_group = parallel.tp_group
    rank = parallel.dp_rank if parallel.dp_rank is not None else tp_group.rank_in_group
    logger.info("HCCL DP prewarm start: rank=%s", rank)
    _prewarm_hccl_group(
        device=tp_group.device,
        group=tp_group.device_group,
        device_module=tp_group.device_module,
    )
    logger.info("HCCL DP prewarm done: rank=%s", rank)


def init_zbal_on_npu() -> None:
    if _is_npu:
        from sglang.srt.hardware_backend.npu.utils import init_zbal

        if get_parallel().pp_size > 1:
            logger.error("only zbal mix mode support pp_size > 1!")
        init_zbal(
            get_parallel().tp_size, get_device().gpu_id, get_parallel().tp_rank
        )  # only switch allocator if is mix mode


def init_moe_gemm_config(hf_text_config) -> None:
    # Use the language config already normalized by ModelConfig. Multimodal
    # wrappers expose it under different attributes (for example,
    # ``text_config`` or ``llm_config``).
    config_to_check = hf_text_config

    # Different MoE architectures expose the per-token expert count under
    # different attribute names (e.g. Gemma4 uses ``top_k_experts``,
    # LongCat-2.0 uses ``moe_topk``).
    moe_topk_attrs = (
        "num_experts_per_tok",
        "num_experts_per_token",
        "top_k_experts",
        "moe_top_k",
        "moe_topk",
    )
    if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
        initialize_moe_config()

    # Initialize GEMM-related configuration for FP8 and FP4 backends.
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    initialize_bf16_gemm_config()


def runs_draft_worker(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether this rank runs a draft worker for speculative decoding."""
    if spec_algorithm.is_none():
        return False

    if (
        envs.SGLANG_ENABLE_PP_SPEC.get()
        and get_parallel().pp_size > 1
        and get_parallel().pp_rank != get_parallel().pp_size - 1
    ):
        # PP+spec: the draft model (MTP layer) needs final hidden states and
        # the lm_head, both of which live on the last PP stage only.
        return False
    return True


def alloc_target_memory_pool(*, tp_worker, draft_worker) -> None:
    """Allocate target KV cache pools if they have not been allocated yet."""
    if (
        tp_worker.model_runner.memory_pool_config is not None
        and tp_worker.model_runner.req_to_token_pool is not None
        and tp_worker.model_runner.token_to_kv_pool_allocator is not None
    ):
        return
    preloaded_weights_bytes = tp_worker.preloaded_weights_bytes
    if draft_worker is not None:
        preloaded_weights_bytes += draft_worker.preloaded_weights_bytes
    tp_worker.model_runner.account_preloaded_weights(preloaded_weights_bytes)
    tp_worker.alloc_memory_pool(
        draft_parallel_layout=(
            draft_worker.draft_parallel_layout if draft_worker is not None else None
        )
    )


def alloc_draft_memory_pool(*, tp_worker, draft_worker) -> None:
    # Lands the retraction backend on the disagg bag before the draft
    # worker's HiCache plan reads it.
    kv_cache_builder.resolve_decode_retraction_backup(tp_worker=tp_worker)
    if draft_worker is not None:
        pool, allocator = tp_worker.get_memory_pool()
        draft_worker.alloc_memory_pool(
            memory_pool_config=tp_worker.model_runner.memory_pool_config,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
        )
        draft_worker.init_hicache_draft_plan()


@dataclass(frozen=True, slots=True)
class WorkerStartup:
    """What finishing the workers' startup hands back to the scheduler."""

    schedule_stream: Any
    # Time spent resizing the KV pool after graph capture, in seconds.
    kv_resize_time: float


def finish_worker_startup(*, tp_worker, draft_worker) -> WorkerStartup:
    """Prewarm the workers on the stream the loop forwards on, resize the KV
    pool to the memory graph capture left, and finish the weight load."""
    kv_resize_time = 0.0
    model_runner = tp_worker.model_runner
    if model_runner.token_to_kv_pool.post_capture_active:
        kv_cache_builder.prepare_hicache_staging(
            tp_worker=tp_worker,
            draft_plan=(
                draft_worker.hicache_draft_plan if draft_worker is not None else None
            ),
        )
    device_module = torch.get_device_module(model_runner.device)
    schedule_stream = device_module.Stream(priority=0)
    # Match run_batch / PPRing.run_slot_batch so warmup allocations stay reusable.
    forward_stream = (
        model_runner.forward_stream
        if (not get_schedule().disable_overlap_schedule) or get_parallel().pp_size > 1
        else schedule_stream
    )
    with device_module.stream(forward_stream):
        prewarm_graph_pool_borrow()
        if draft_worker is None:
            model_runner.prewarm_sampling()
        else:
            draft_worker.prewarm_sampling()
    if model_runner.token_to_kv_pool.post_capture_active:
        tic = time.perf_counter()
        model_runner.post_capture_resize_kv_pool(
            draft_runners=(
                draft_worker._draft_model_runners() if draft_worker is not None else ()
            )
        )
        kv_resize_time += time.perf_counter() - tic

    if get_model().is_startup_weight_load_overlap:
        tp_worker.finalize_startup_weight_load()

    # Adaptive/speculative graphs and post-capture KV sizing can consume
    # the headroom seen by the initial DeepGEMM layout budget. Refresh it
    # after these allocations, before elastic EP rejoins healthy ranks
    # that do not participate in this startup collective.
    from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
        refresh_deep_gemm_layout_memory_budget,
    )

    refresh_deep_gemm_layout_memory_budget(model_runner, only_if_initialized=True)

    if (
        get_exec().moe.elastic_ep_backend is not None
        and get_exec().moe.ep_join_mode == "recover"
    ):
        model_runner.post_capture_elastic_ep_recover()
    return WorkerStartup(schedule_stream=schedule_stream, kv_resize_time=kv_resize_time)


def select_model_worker(
    *, tp_worker, draft_worker, dllm_worker, spec_algorithm: SpeculativeAlgorithm
):
    """The worker the scheduler runs its forwards through."""
    if dllm_worker is not None:
        return dllm_worker
    elif spec_algorithm.is_none() or draft_worker is None:
        # PP+spec: non-last stages have no draft worker; they run the
        # verify-shaped target forward through the plain tp_worker.
        return tp_worker
    else:
        return draft_worker


def init_tp_model_worker(*, server_args: ServerArgs, nccl_port: int):
    worker_kwargs = dict(
        server_args=server_args,
        gpu_id=get_device().gpu_id,
        nccl_port=nccl_port,
    )

    # FIXME: move tp worker's init logic outside of the scheduler.
    from sglang.srt.managers.tp_worker import TpModelWorker

    return TpModelWorker(**worker_kwargs)


def maybe_init_draft_worker(
    *,
    server_args: ServerArgs,
    nccl_port: int,
    spec_algorithm: SpeculativeAlgorithm,
    tp_worker,
):
    if not runs_draft_worker(spec_algorithm):
        return None

    # Launch a draft worker for speculative decoding. It builds its draft
    # from this process's own config: what differs for the draft — the
    # target's context length, the draft load format, its attention backend
    # — is resolved per runner, not on a config copy.
    draft_worker_kwargs = dict(
        server_args=server_args,
        gpu_id=get_device().gpu_id,
        nccl_port=nccl_port,
        target_worker=tp_worker,
    )

    DraftWorkerClass = spec_algorithm.create_worker(server_args)
    return DraftWorkerClass(**draft_worker_kwargs)


def maybe_init_dllm_worker(*, server_args: ServerArgs, tp_worker):
    if get_exec().dllm.dllm_algorithm is None:
        return None
    from sglang.srt.dllm.worker import DllmWorker

    return DllmWorker(tp_worker, server_args)


def init_memory_pools(*, tp_worker, draft_worker) -> None:
    """Allocate KV cache pools for target and draft workers."""
    alloc_target_memory_pool(tp_worker=tp_worker, draft_worker=draft_worker)
    alloc_draft_memory_pool(tp_worker=tp_worker, draft_worker=draft_worker)


def init_all_attention_backends(*, tp_worker, draft_worker) -> None:
    """Initialize attention backends for all workers."""
    tp_worker.init_attention_backends()
    if draft_worker is not None:
        draft_worker.init_attention_backends()


# Not DynamicGradMode: tensors created under inference mode here could not
# be written in place by the no_grad disagg event loops.
@torch.no_grad()
def init_all_cuda_graphs(*, tp_worker, draft_worker) -> None:
    """Capture cuda graphs for all workers."""
    tp_worker.init_cuda_graphs()
    if draft_worker is not None:
        draft_worker.init_cuda_graphs()


@dataclass(frozen=True, slots=True)
class StartedWorkers:
    """The workers once they are up, and what starting them measured."""

    workers: Workers
    # The dLLM worker, which reports the worker info in dLLM mode.
    dllm_worker: Any
    schedule_stream: Any
    # Time spent allocating the KV pools, resize included, in seconds.
    kv_cache_allocation_time: float


def start_workers(
    *, server_args: ServerArgs, nccl_port: int, spec_algorithm: SpeculativeAlgorithm
) -> StartedWorkers:
    """Load the workers, allocate their KV pools, capture their graphs and
    finish their startup."""
    # Load model weights.
    tp_worker = init_tp_model_worker(server_args=server_args, nccl_port=nccl_port)
    if get_model().is_startup_weight_load_overlap:
        tp_worker.start_startup_weight_load()
    draft_worker = maybe_init_draft_worker(
        server_args=server_args,
        nccl_port=nccl_port,
        spec_algorithm=spec_algorithm,
        tp_worker=tp_worker,
    )
    dllm_worker = maybe_init_dllm_worker(server_args=server_args, tp_worker=tp_worker)

    # Prepare KV cache pools for all workers
    tic = time.perf_counter()
    init_memory_pools(tp_worker=tp_worker, draft_worker=draft_worker)
    kv_cache_allocation_time = time.perf_counter() - tic
    if get_memory().enable_unified_memory:
        # The first compaction move must not happen while serving with a
        # full pool (see MultiEndedAllocator.prewarm_compaction); prewarm it
        # before graph capture and KV sizing account for its memory.
        tp_worker.model_runner.token_to_kv_pool_allocator.prewarm_compaction()

    init_all_attention_backends(tp_worker=tp_worker, draft_worker=draft_worker)
    init_all_cuda_graphs(tp_worker=tp_worker, draft_worker=draft_worker)

    startup = finish_worker_startup(tp_worker=tp_worker, draft_worker=draft_worker)
    kv_cache_allocation_time += startup.kv_resize_time

    model_worker = select_model_worker(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        dllm_worker=dllm_worker,
        spec_algorithm=spec_algorithm,
    )
    return StartedWorkers(
        workers=Workers(
            tp_worker=tp_worker, draft_worker=draft_worker, model_worker=model_worker
        ),
        dllm_worker=dllm_worker,
        schedule_stream=startup.schedule_stream,
        kv_cache_allocation_time=kv_cache_allocation_time,
    )


def set_default_pp_max_micro_batch_size(max_running_requests: int) -> None:
    if not get_parallel().pp_max_micro_batch_size:
        get_context().override(
            "scheduler.pp_max_micro_batch_size_default",
            pp_max_micro_batch_size=max(
                max_running_requests // get_parallel().pp_size, 1
            ),
        )


def measure_startup_memory(
    *,
    device: str,
    max_total_num_tokens: int,
    max_prefill_tokens: int,
    max_running_requests: int,
    context_len: int,
) -> float:
    """The free device memory once the workers are up, logged with the limits."""
    startup_available_gpu_memory_gb = get_available_gpu_memory(
        device, get_device().gpu_id, empty_cache=False
    )
    if get_parallel().tp_rank == 0:
        logger.info(
            f"max_total_num_tokens={max_total_num_tokens}, "
            f"chunked_prefill_size={get_schedule().chunked_prefill_size}, "
            f"max_prefill_tokens={max_prefill_tokens}, "
            f"max_running_requests={max_running_requests}, "
            f"context_len={context_len}, "
            f"{'available_cpu_mem' if device == 'cpu' else 'available_gpu_mem'}="
            f"{startup_available_gpu_memory_gb:.2f} GB"
        )
    return startup_available_gpu_memory_gb
