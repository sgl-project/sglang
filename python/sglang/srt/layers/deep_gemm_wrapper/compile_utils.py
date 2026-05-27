import logging
import multiprocessing as mp
import os
import queue
import traceback
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_div, get_available_gpu_memory, is_musa

logger = logging.getLogger(__name__)

_is_musa = is_musa()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()
_COMPILE_WORKERS = envs.SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS.get()
# Keep a small safety margin for CUDA contexts, allocator metadata, and transient
# DeepGEMM compiler buffers that are not covered by the executor tensor estimate.
_WARMUP_MEMORY_HEADROOM_GB = envs.SGLANG_JIT_DEEPGEMM_WARMUP_MEMORY_HEADROOM_GB.get()
_WARMUP_WORKER_MEMORY_OVERHEAD_GB = (
    envs.SGLANG_JIT_DEEPGEMM_WARMUP_WORKER_MEMORY_OVERHEAD_GB.get()
)
_WARMUP_PROGRESS_BATCH_SIZE = envs.SGLANG_JIT_DEEPGEMM_WARMUP_PROGRESS_BATCH_SIZE.get()
_WARMUP_PROGRESS_ERROR = "error"


# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    _BUILTIN_M_LIST = []

    if _FAST_WARMUP:
        # In fast warmup mode, only compile a small set of typical Ms

        # First cover all the small bs to ensure decode performance
        _BUILTIN_M_LIST += list(range(1, 1025))

        # Then cover larger batch sizes with gradually increasing steps
        # For example, when chunekd prefill size is 16384
        # The sampled Ms would be:
        #   1024, 1026, ... 2046 (step 2)
        #   2048, 2052, ... 4092 (step 4)
        #   4096, 5004, ... 8184 (step 8)
        #   8192, 9008, ... 16384 (step 16)
        # Totally 1024 + 1024 / 2 + 2048 / 4 + 4096 / 8 + 8192 / 16 = 3072 kernels
        next_m, sample_step = 1024, 2
        max_prefill_bs = (
            min(server_args.chunked_prefill_size, 32 * 1024)
            if server_args.chunked_prefill_size >= 1
            else 16 * 1024
        )
        while next_m < max_prefill_bs:
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))
            next_m = next_m * 2
            sample_step = sample_step * 2
        _BUILTIN_M_LIST.append(max_prefill_bs)
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))
    else:
        # When fast warmup isn't enabled, generate m_max and compile all the covered Ms.
        m_max = 1024 * 16
        if server_args.chunked_prefill_size < 1:
            m_max = 1024 * 64
        elif server_args.chunked_prefill_size > 8192:
            m_max = server_args.chunked_prefill_size * 2
        m_max = min(1024 * 128, m_max)
        _BUILTIN_M_LIST += list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()
    GEMM_NT_BF16BF16F32 = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


@contextmanager
def _deep_gemm_compile_mode():
    # compile_mode is available only on newer DeepGEMM versions.
    has_compile_mode_api = hasattr(deep_gemm, "get_compile_mode") and hasattr(
        deep_gemm, "set_compile_mode"
    )
    if not has_compile_mode_api:
        yield
        return

    old_compile_mode = deep_gemm.get_compile_mode()
    deep_gemm.set_compile_mode(1)
    try:
        yield
    finally:
        deep_gemm.set_compile_mode(old_compile_mode)


def _in_deep_gemm_precompile_stage() -> bool:
    # The precompile CLI updates this env after module import, so read it lazily.
    return envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()


def _split_m_list(m_list: List[int], num_workers: int) -> List[List[int]]:
    # Use static contiguous chunks so each worker can allocate one executor with
    # max_m equal to its chunk maximum.
    chunk_size = ceil_div(len(m_list), num_workers)
    return [
        m_list[start : start + chunk_size]
        for start in range(0, len(m_list), chunk_size)
    ]


def _estimate_warmup_chunk_memory(
    kernel_type: DeepGemmKernelType,
    m_list_chunk: List[int],
    n: int,
    k: int,
    num_groups: int,
) -> float:
    # A warmup executor allocates buffers for the largest M in its chunk.
    return _BaseWarmupExecutor.get_memory_requirement(
        kernel_type,
        max_m=max(m_list_chunk),
        n=n,
        k=k,
        num_groups=num_groups,
    )


def _estimate_warmup_executor_memory(
    kernel_type: DeepGemmKernelType,
    m_list_chunks: List[List[int]],
    n: int,
    k: int,
    num_groups: int,
) -> float:
    return sum(
        _estimate_warmup_chunk_memory(
            kernel_type, m_list_chunk, n=n, k=k, num_groups=num_groups
        )
        for m_list_chunk in m_list_chunks
    )


def _get_warmup_compile_device_memory_budgets() -> Dict[int, float]:
    current_device = torch.cuda.current_device()
    current_capability = torch.cuda.get_device_capability(current_device)
    memory_budgets = {}
    try:
        for gpu_id in range(torch.cuda.device_count()):
            device_capability = torch.cuda.get_device_capability(gpu_id)
            if device_capability != current_capability:
                # A JIT cache produced for one architecture is not necessarily
                # reusable on another, so do not mix devices with different SMs.
                logger.warning(
                    "Skipping cuda:%s for DeepGEMM warmup compilation because "
                    "its compute capability %s differs from current device "
                    "cuda:%s compute capability %s.",
                    gpu_id,
                    device_capability,
                    current_device,
                    current_capability,
                )
                continue
            # get_available_gpu_memory reads the target CUDA device state.
            torch.cuda.set_device(gpu_id)
            memory_budgets[gpu_id] = get_available_gpu_memory(
                device="cuda", gpu_id=gpu_id
            )
    finally:
        torch.cuda.set_device(current_device)

    return memory_budgets


def _assign_warmup_chunks_to_devices(
    kernel_type: DeepGemmKernelType,
    m_list_chunks: List[List[int]],
    n: int,
    k: int,
    num_groups: int,
    device_memory_budgets: Dict[int, float],
) -> List[Tuple[int, List[int]]]:
    if not device_memory_budgets:
        return []

    device_ids = sorted(device_memory_budgets)
    executor_memory_by_device = {gpu_id: 0.0 for gpu_id in device_ids}
    worker_count_by_device = {gpu_id: 0 for gpu_id in device_ids}
    assignments: Dict[int, Tuple[int, List[int]]] = {}

    # Place the largest chunks first. This is a conservative bin-packing pass
    # that reduces the chance of accepting a plan that later OOMs.
    chunks_with_memory = [
        (
            index,
            m_list_chunk,
            _estimate_warmup_chunk_memory(
                kernel_type, m_list_chunk, n=n, k=k, num_groups=num_groups
            ),
        )
        for index, m_list_chunk in enumerate(m_list_chunks)
    ]

    for index, m_list_chunk, chunk_memory in sorted(
        chunks_with_memory, key=lambda item: item[2], reverse=True
    ):
        best_gpu_id = None
        best_remaining_memory = None
        for gpu_id in device_ids:
            # Account for both persistent executor tensors and per-process CUDA
            # overhead, then leave headroom for transient compiler allocations.
            next_executor_memory = executor_memory_by_device[gpu_id] + chunk_memory
            next_worker_count = worker_count_by_device[gpu_id] + 1
            next_total_memory = (
                next_executor_memory
                + next_worker_count * _WARMUP_WORKER_MEMORY_OVERHEAD_GB
                + _WARMUP_MEMORY_HEADROOM_GB
            )
            remaining_memory = device_memory_budgets[gpu_id] - next_total_memory
            if remaining_memory < 0:
                continue
            if (
                best_remaining_memory is None
                or remaining_memory > best_remaining_memory
            ):
                best_gpu_id = gpu_id
                best_remaining_memory = remaining_memory

        if best_gpu_id is None:
            # Signal the caller to retry with fewer workers.
            return []

        executor_memory_by_device[best_gpu_id] += chunk_memory
        worker_count_by_device[best_gpu_id] += 1
        assignments[index] = (best_gpu_id, m_list_chunk)

    # Return the original chunk order so progress remains deterministic.
    return [assignments[index] for index in range(len(m_list_chunks))]


def _log_parallel_warmup_plan(
    kernel_type: DeepGemmKernelType,
    warmup_plan: List[Tuple[int, List[int]]],
    n: int,
    k: int,
    num_groups: int,
    device_memory_budgets: Dict[int, float],
) -> None:
    plan_by_device: Dict[int, List[List[int]]] = {}
    for gpu_id, m_list_chunk in warmup_plan:
        plan_by_device.setdefault(gpu_id, []).append(m_list_chunk)

    plan_parts = []
    for gpu_id, chunks in sorted(plan_by_device.items()):
        executor_memory = _estimate_warmup_executor_memory(
            kernel_type, chunks, n=n, k=k, num_groups=num_groups
        )
        worker_overhead = len(chunks) * _WARMUP_WORKER_MEMORY_OVERHEAD_GB
        estimated_memory = (
            executor_memory + worker_overhead + _WARMUP_MEMORY_HEADROOM_GB
        )
        plan_parts.append(
            f"cuda:{gpu_id} workers={len(chunks)} "
            f"estimated={estimated_memory:.2f}GB/"
            f"available={device_memory_budgets[gpu_id]:.2f}GB"
        )

    logger.info("DeepGEMM warmup compile device plan: %s.", "; ".join(plan_parts))


def _get_parallel_warmup_plan(
    kernel_type: DeepGemmKernelType,
    m_list: List[int],
    n: int,
    k: int,
    num_groups: int,
    device_memory_budgets: Dict[int, float],
) -> List[Tuple[int, List[int]]]:
    requested_workers = max(_COMPILE_WORKERS, 1)
    max_workers_by_shapes = min(requested_workers, len(m_list))
    num_workers = max_workers_by_shapes

    # Try the requested parallelism first, then shrink until the estimated
    # memory footprint fits the visible GPU set.
    while num_workers > 1:
        m_list_chunks = _split_m_list(m_list, num_workers)
        warmup_plan = _assign_warmup_chunks_to_devices(
            kernel_type,
            m_list_chunks,
            n=n,
            k=k,
            num_groups=num_groups,
            device_memory_budgets=device_memory_budgets,
        )
        if warmup_plan:
            if num_workers < max_workers_by_shapes:
                logger.warning(
                    "Reducing DeepGEMM warmup compile workers from %s to %s "
                    "because the requested worker plan exceeds available GPU memory.",
                    max_workers_by_shapes,
                    num_workers,
                )
            _log_parallel_warmup_plan(
                kernel_type,
                warmup_plan,
                n=n,
                k=k,
                num_groups=num_groups,
                device_memory_budgets=device_memory_budgets,
            )
            return warmup_plan
        num_workers -= 1

    if requested_workers > 1 and max_workers_by_shapes > 1:
        logger.warning(
            "Falling back to single-process DeepGEMM warmup because parallel "
            "workers would exceed the available GPU memory budget."
        )
    return [(torch.cuda.current_device(), m_list)]


def _run_deep_gemm_warmup_serial(
    kernel_type: DeepGemmKernelType,
    max_m: int,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
    desc: str,
) -> None:
    # Keep the old single-process path for fallback and for worker_count == 1.
    executor = _BaseWarmupExecutor.create(
        kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
    )
    try:
        with _deep_gemm_compile_mode():
            for m in tqdm(m_list, desc=desc):
                executor.execute(m=m)
        torch.cuda.current_stream().synchronize()
    finally:
        del executor
        torch.cuda.empty_cache()


def _run_deep_gemm_warmup_worker(
    kernel_type_value: int,
    gpu_id: int,
    max_m: int,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
    progress_queue,
    error_queue,
) -> None:
    pending_progress = 0
    saved_context = None
    try:
        # Each subprocess owns one CUDA context and must not participate in the
        # rank-wide symmetric-memory collective path.
        saved_context = disable_symmetric_memory_context()
        torch.cuda.set_device(gpu_id)
        kernel_type = DeepGemmKernelType(kernel_type_value)
        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        try:
            with _deep_gemm_compile_mode():
                for m in m_list:
                    executor.execute(m=m)
                    pending_progress += 1
                    if pending_progress >= _WARMUP_PROGRESS_BATCH_SIZE:
                        # Batch progress updates to keep multiprocessing queue
                        # traffic from becoming visible in the compile path.
                        progress_queue.put(pending_progress)
                        pending_progress = 0
            torch.cuda.current_stream().synchronize()
        finally:
            del executor
            torch.cuda.empty_cache()

        if pending_progress:
            progress_queue.put(pending_progress)
    except BaseException as e:
        if pending_progress:
            progress_queue.put(pending_progress)
        error_payload = (os.getpid(), gpu_id, repr(e), traceback.format_exc())
        error_queue.put(error_payload)
        # Also write the error to the progress queue so the parent can unblock
        # even when it is waiting for progress instead of polling errors.
        progress_queue.put((_WARMUP_PROGRESS_ERROR, *error_payload))
        raise SystemExit(1)
    finally:
        if saved_context is not None:
            restore_symmetric_memory_context(saved_context)


def _drain_warmup_errors(error_queue) -> List[Tuple[int, int, str, str]]:
    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            return errors


def _raise_warmup_worker_error(error_queue, processes) -> None:
    # Prefer the child traceback when available; otherwise report the failed
    # process exit code, which covers early crashes before the queues are ready.
    errors = _drain_warmup_errors(error_queue)
    if errors:
        pid, gpu_id, error_repr, tb = errors[0]
        raise RuntimeError(
            f"DeepGEMM warmup worker {pid} on cuda:{gpu_id} failed: "
            f"{error_repr}\n{tb}"
        )

    failed_processes = [
        process for process in processes if process.exitcode not in (None, 0)
    ]
    if failed_processes:
        failed = failed_processes[0]
        raise RuntimeError(
            f"DeepGEMM warmup worker {failed.name} pid={failed.pid} exited "
            f"with code {failed.exitcode}."
        )


def _raise_warmup_progress_error(progress) -> None:
    if (
        isinstance(progress, tuple)
        and len(progress) == 5
        and progress[0] == _WARMUP_PROGRESS_ERROR
    ):
        _, pid, gpu_id, error_repr, tb = progress
        raise RuntimeError(
            f"DeepGEMM warmup worker {pid} on cuda:{gpu_id} failed: "
            f"{error_repr}\n{tb}"
        )


def _get_warmup_progress_count(progress) -> int:
    _raise_warmup_progress_error(progress)
    if isinstance(progress, int):
        return progress
    raise RuntimeError(f"Unexpected DeepGEMM warmup progress message: {progress!r}.")


def _stop_warmup_processes(processes) -> None:
    # Terminate first to allow CUDA teardown, then kill any process that is stuck
    # inside compiler or driver cleanup.
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()


def _run_deep_gemm_warmup_parallel(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
    device_memory_budgets: Dict[int, float],
) -> None:
    warmup_plan = _get_parallel_warmup_plan(
        kernel_type,
        m_list,
        n=n,
        k=k,
        num_groups=num_groups,
        device_memory_budgets=device_memory_budgets,
    )
    if len(warmup_plan) == 1:
        # Avoid multiprocessing overhead when planning collapsed to one chunk.
        _run_deep_gemm_warmup_serial(
            kernel_type,
            max_m=max(m_list),
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=m_list,
            desc="DeepGEMM warmup",
        )
        return

    device_ids = sorted(set(gpu_id for gpu_id, _ in warmup_plan))
    logger.info(
        "Running DeepGEMM warmup compilation with %s worker processes across "
        "%s CUDA devices.",
        len(warmup_plan),
        len(device_ids),
    )
    # Use spawn for CUDA safety: fork can inherit an initialized CUDA runtime
    # from the parent and lead to undefined behavior.
    mp_context = mp.get_context("spawn")
    progress_queue = mp_context.Queue()
    error_queue = mp_context.Queue()
    processes = [
        mp_context.Process(
            name=f"deep-gemm-warmup-cuda-{gpu_id}",
            target=_run_deep_gemm_warmup_worker,
            args=(
                int(kernel_type),
                gpu_id,
                max(m_list_chunk),
                n,
                k,
                num_groups,
                m_list_chunk,
                progress_queue,
                error_queue,
            ),
        )
        for gpu_id, m_list_chunk in warmup_plan
    ]

    started_processes = []
    try:
        # Track only successfully started processes so cleanup does not touch
        # Process objects that never reached start().
        for process in processes:
            process.start()
            started_processes.append(process)

        completed = 0
        total = len(m_list)
        with tqdm(total=total, desc="DeepGEMM warmup compile") as pbar:
            while completed < total:
                _raise_warmup_worker_error(error_queue, processes)
                try:
                    progress = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    _raise_warmup_worker_error(error_queue, processes)
                    if all(process.exitcode is not None for process in processes):
                        break
                    continue
                progress_count = _get_warmup_progress_count(progress)
                completed += progress_count
                pbar.update(progress_count)

            for process in processes:
                process.join()
            # Workers may send their final progress right before exit. Drain
            # those messages after join before validating the total.
            while True:
                try:
                    progress = progress_queue.get_nowait()
                except queue.Empty:
                    break
                progress_count = _get_warmup_progress_count(progress)
                completed += progress_count
                pbar.update(progress_count)

        _raise_warmup_worker_error(error_queue, processes)
        if completed != total:
            raise RuntimeError(
                f"DeepGEMM warmup compiled {completed} of {total} requested shapes."
            )
    except BaseException:
        _stop_warmup_processes(started_processes)
        raise
    finally:
        progress_queue.close()
        error_queue.close()


# TODO improve code
def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        # TODO maybe improve logs
        if not _in_deep_gemm_precompile_stage() and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering DeepGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not run `sglang.compile_deep_gemm`. "
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
                " for pre-compilation to reduce the overhead if you have not run it before. "
                "For example: "
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _in_deep_gemm_precompile_stage() else ''}"
        )

        _compile_deep_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )


# NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
def _compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))
        if not m_list:
            logger.info("Skip DeepGEMM warmup because no valid M values are selected.")
            return

        # Precompilation runs on the first rank on this node. Use all CUDA
        # devices visible to this process for worker placement, and keep the
        # current device as the serial fallback/load device.
        current_gpu_id = torch.cuda.current_device()
        device_memory_budgets = _get_warmup_compile_device_memory_budgets()
        memory_budget = device_memory_budgets[current_gpu_id]
        logger.info(
            "DeepGEMM warmup visible GPU memory budgets: %s.",
            ", ".join(
                f"cuda:{gpu_id}={budget:.2f}GB"
                for gpu_id, budget in sorted(device_memory_budgets.items())
            ),
        )

        # If the memory budget is less memory requirement, we need to reduce max_m to avoid out of memory, which might further cause hanging during warmup
        max_m = max(m_list)
        required_memory = _BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"
        )
        if memory_budget < required_memory:
            # TODO: Maybe compute the max_m based on the memory budget
            while (
                _BaseWarmupExecutor.get_memory_requirement(
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget
                and max_m > 4096
            ):
                max_m = max_m // 2
            logger.warning(
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"
            )
            m_list = [m for m in m_list if m <= max_m]
            if not m_list:
                logger.warning(
                    "Skip DeepGEMM warmup because memory reduction removed all M values."
                )
                return

        _run_deep_gemm_warmup_parallel(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=m_list,
            device_memory_budgets=device_memory_budgets,
        )
    finally:
        # Restore symmetric memory context
        restore_symmetric_memory_context(saved_context)


class _BaseWarmupExecutor:
    @staticmethod
    def create(kernel_type: DeepGemmKernelType, **kwargs):
        return {
            DeepGemmKernelType.GEMM_NT_F8F8BF16: _NormalWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,
            DeepGemmKernelType.GEMM_NT_BF16BF16F32: _BF16F32WarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG: _BF16GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED: _BF16GroupedMaskedWarmupExecutor,
        }[kernel_type](**kwargs)

    @staticmethod
    def get_memory_requirement(
        kernel_type: DeepGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:
        # Return the required memory space in GB for warmup executor
        _GB = 1 << 30
        if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
            return (max_m * k + n * k + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:
            return (
                max_m * k * 2 + num_groups * n * k * 2 + max_m * 4 + max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:
            # bf16 lhs + bf16 rhs + fp32 out
            return (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED:
            return (
                num_groups * max_m * k * 2
                + num_groups * n * k * 2
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def execute(self, m):
        raise NotImplementedError


def _empty_token_fp8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128


class _NormalWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            self.m_indices[:m],
        )


class _BF16GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.a = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            self.a[:m],
            self.b,
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expect_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _BF16F32WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.float32)

    def execute(self, m):
        deep_gemm.bf16_gemm_nt(self.lhs[:m], self.rhs, self.out[:m])


class _BF16GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.a = torch.empty(
            (num_groups, max_m, k), device="cuda", dtype=torch.bfloat16
        )
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_bf16_gemm_nt_masked(
            self.a,
            self.b,
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expect_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if _is_musa:
        return nullcontext()

    return _deep_gemm_execution_hook(m, n, k, num_groups, kernel_type)


@contextmanager
def _deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield
