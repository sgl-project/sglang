import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_align, ceil_div, get_available_gpu_memory, is_musa

logger = logging.getLogger(__name__)

_is_musa = is_musa()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()
_USE_SHARED_PRECOMPILE_MARKER = (
    envs.SGLANG_JIT_DEEPGEMM_USE_SHARED_PRECOMPILE_MARKER.get()
)
_SHARD_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_SHARD_PRECOMPILE.get()
_SHOW_PROGRESS = envs.SGLANG_JIT_DEEPGEMM_SHOW_PROGRESS.get()
_PRECOMPILE_SHARD_RANK = 0
_PRECOMPILE_SHARD_WORLD_SIZE = 1

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
    global _PRECOMPILE_SHARD_RANK
    global _PRECOMPILE_SHARD_WORLD_SIZE

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
    _PRECOMPILE_SHARD_RANK, _PRECOMPILE_SHARD_WORLD_SIZE = (
        _get_local_precompile_shard_info(gpu_id, server_args)
    )

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _SHARD_PRECOMPILE or _IS_FIRST_RANK_ON_NODE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()
    GEMM_NT_BF16BF16F32 = auto()
    TF32_HC_PRENORM_GEMM = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


def _get_local_precompile_shard_info(
    gpu_id: int, server_args: ServerArgs
) -> Tuple[int, int]:
    tp_size_per_node = max(
        1,
        (server_args.tp_size + max(1, server_args.nnodes) - 1)
        // max(1, server_args.nnodes),
    )
    shard_size_override = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE_SHARD_SIZE.get()
    shard_rank_override = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE_SHARD_RANK.get()
    world_size = int(
        shard_size_override
        or os.getenv("LOCAL_WORLD_SIZE")
        or os.getenv("LOCAL_SIZE")
        or tp_size_per_node
    )
    world_size = max(1, world_size)

    rank = -1 if shard_rank_override is None else int(shard_rank_override)
    if rank < 0:
        rank = (
            (gpu_id - server_args.base_gpu_id) // server_args.gpu_id_step
        ) % world_size
    return rank, world_size


def _get_deep_gemm_shape_owner_rank(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> int:
    if _PRECOMPILE_SHARD_WORLD_SIZE <= 1:
        return 0
    encoded = f"{kernel_type.value}:{n}:{k}:{num_groups}".encode()
    return int(hashlib.sha256(encoded).hexdigest(), 16) % (_PRECOMPILE_SHARD_WORLD_SIZE)


def _is_deep_gemm_shape_owner(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> bool:
    return _PRECOMPILE_SHARD_RANK == _get_deep_gemm_shape_owner_rank(
        kernel_type, n, k, num_groups
    )


def _should_show_deep_gemm_warmup_progress() -> bool:
    return _SHOW_PROGRESS


def _get_deep_gemm_warmup_progress_position() -> int:
    return _PRECOMPILE_SHARD_RANK if _SHARD_PRECOMPILE else 0


def _safe_get_deep_gemm_version() -> str:
    if not ENABLE_JIT_DEEPGEMM:
        return "disabled"

    version = getattr(deep_gemm, "__version__", None)
    if version is not None:
        return str(version)

    try:
        from importlib import metadata

        for package_name in ("sgl-deep-gemm", "deep_gemm", "deep-gemm"):
            try:
                return metadata.version(package_name)
            except metadata.PackageNotFoundError:
                pass
    except Exception:
        pass

    return "unknown"


def _safe_get_triton_version() -> str:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return "unknown"


def _safe_get_device_key() -> str:
    try:
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            capability = ".".join(
                str(v) for v in torch.cuda.get_device_capability(gpu_id)
            )
            return f"{torch.cuda.get_device_name(gpu_id)}|sm{capability}"
    except Exception:
        pass
    return "unknown"


def _m_list_fingerprint(m_list: List[int]) -> Dict[str, object]:
    encoded = json.dumps(m_list, separators=(",", ":")).encode()
    return {
        "count": len(m_list),
        "first": m_list[0] if m_list else None,
        "last": m_list[-1] if m_list else None,
        "sha256": hashlib.sha256(encoded).hexdigest()[:16],
    }


def _deep_gemm_precompile_marker_payload(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> Dict[str, object]:
    return {
        "schema": 1,
        "kernel_type": kernel_type.name,
        "n": n,
        "k": k,
        "num_groups": num_groups,
        "m_list": _m_list_fingerprint(m_list),
        "fast_warmup": bool(_FAST_WARMUP),
        "deep_gemm_version": _safe_get_deep_gemm_version(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "triton_version": _safe_get_triton_version(),
        "device": _safe_get_device_key(),
        "dg_jit_use_nvrtc": os.getenv("DG_JIT_USE_NVRTC", ""),
    }


def _deep_gemm_precompile_marker_path(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> Path:
    payload = _deep_gemm_precompile_marker_payload(
        kernel_type, n, k, num_groups, m_list
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    marker_dir = Path(os.environ["DG_JIT_CACHE_DIR"]) / ".sglang_precompile_markers"
    return marker_dir / f"{digest}.done"


def _deep_gemm_precompile_marker_dir() -> Path:
    return Path(os.environ["DG_JIT_CACHE_DIR"]) / ".sglang_precompile_markers"


def is_deep_gemm_precompile_marked(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: Optional[List[int]] = None,
) -> bool:
    if m_list is None:
        m_list = _BUILTIN_M_LIST
    return _deep_gemm_precompile_marker_path(
        kernel_type, n, k, num_groups, m_list
    ).exists()


def mark_deep_gemm_precompiled(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: Optional[List[int]] = None,
) -> None:
    if m_list is None:
        m_list = _BUILTIN_M_LIST
    marker_path = _deep_gemm_precompile_marker_path(
        kernel_type, n, k, num_groups, m_list
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _deep_gemm_precompile_marker_payload(
        kernel_type, n, k, num_groups, m_list
    )
    payload["created_at"] = time.time()
    marker_path.write_text(json.dumps(payload, sort_keys=True))


def sync_deep_gemm_precompile_markers() -> List[str]:
    """All-gather visible marker paths across distributed ranks.

    The compiled artifacts themselves live in DeepGEMM's cache directory; this
    function is a lightweight synchronization point for the runtime metadata.
    """
    marker_dir = _deep_gemm_precompile_marker_dir()
    local_markers = sorted(str(path) for path in marker_dir.glob("*.done"))

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_markers

    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_markers)
    return sorted({marker for markers in gathered for marker in (markers or [])})


@contextmanager
def _deep_gemm_precompile_file_lock(marker_path: Path):
    lock_path = marker_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except Exception:
            logger.warning(
                "Failed to acquire DeepGEMM precompile lock %s; continuing without "
                "cross-process serialization.",
                lock_path,
                exc_info=True,
            )
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_file.close()


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

        marker_path = _deep_gemm_precompile_marker_path(
            kernel_type, n, k, num_groups, _BUILTIN_M_LIST
        )
        use_marker = _USE_SHARED_PRECOMPILE_MARKER or _SHARD_PRECOMPILE
        if use_marker and marker_path.exists():
            logger.info(
                "Skip DeepGEMM all-M warmup for <%s> N=%s, K=%s, num_groups=%s "
                "because shared precompile marker exists: %s",
                kernel_type.name,
                n,
                k,
                num_groups,
                marker_path,
            )
            return
        if _SHARD_PRECOMPILE and not _is_deep_gemm_shape_owner(
            kernel_type, n, k, num_groups
        ):
            owner_rank = _get_deep_gemm_shape_owner_rank(kernel_type, n, k, num_groups)
            logger.info(
                "Skip DeepGEMM all-M warmup for <%s> N=%s, K=%s, num_groups=%s "
                "on shard rank %s/%s; owner shard rank is %s.",
                kernel_type.name,
                n,
                k,
                num_groups,
                _PRECOMPILE_SHARD_RANK,
                _PRECOMPILE_SHARD_WORLD_SIZE,
                owner_rank,
            )
            return

        maybe_lock = (
            _deep_gemm_precompile_file_lock(marker_path)
            if use_marker
            else nullcontext()
        )

        with maybe_lock:
            if use_marker and marker_path.exists():
                logger.info(
                    "Skip DeepGEMM all-M warmup for <%s> N=%s, K=%s, "
                    "num_groups=%s because another process created marker: %s",
                    kernel_type.name,
                    n,
                    k,
                    num_groups,
                    marker_path,
                )
                return

            # TODO maybe improve logs
            if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
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
                f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
            )

            _compile_deep_gemm_one_type_all(
                kernel_type=kernel_type,
                n=n,
                k=k,
                num_groups=num_groups,
                m_list=_BUILTIN_M_LIST,
            )
            if use_marker:
                mark_deep_gemm_precompiled(
                    kernel_type, n, k, num_groups, _BUILTIN_M_LIST
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

        # Precompilation normally runs on the first local rank, but sharded
        # precompile can assign a shape to any local rank.
        gpu_id = torch.cuda.current_device()
        memory_budget = get_available_gpu_memory(device="cuda", gpu_id=gpu_id)

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

        # Need some methods to estimate needed memory for warmup
        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        has_compile_mode_api = hasattr(deep_gemm, "get_compile_mode") and hasattr(
            deep_gemm, "set_compile_mode"
        )
        if has_compile_mode_api:
            old_compile_mode = deep_gemm.get_compile_mode()
            deep_gemm.set_compile_mode(1)

        # TODO can use multi thread
        for m in tqdm(
            m_list,
            desc=(
                "DeepGEMM warmup"
                f" shard={_PRECOMPILE_SHARD_RANK}/{_PRECOMPILE_SHARD_WORLD_SIZE}"
                f" {kernel_type.name} N={n} K={k} G={num_groups}"
            ),
            disable=not _should_show_deep_gemm_warmup_progress(),
            position=_get_deep_gemm_warmup_progress_position(),
            dynamic_ncols=True,
            leave=False,
        ):
            executor.execute(m=m)
        if has_compile_mode_api:
            deep_gemm.set_compile_mode(old_compile_mode)

        # clean up input buffers
        torch.cuda.current_stream().synchronize()
        del executor
        torch.cuda.empty_cache()
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
            DeepGemmKernelType.TF32_HC_PRENORM_GEMM: _TF32HcPrenormWarmupExecutor,
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
        elif kernel_type == DeepGemmKernelType.TF32_HC_PRENORM_GEMM:
            # The generic hook's fourth dimension is num_splits for MHC.
            # A value of 0 represents DeepGEMM's unsplit num_splits=None path.
            num_splits = num_groups if num_groups > 0 else 1
            return (max_m * k * 2 + n * k * 4 + num_splits * max_m * (n + 1) * 4) / _GB
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
            self.m_indices[:m],
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


class _TF32HcPrenormWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.x = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.fn = torch.empty((n, k), device="cuda", dtype=torch.float32)
        self.n = n
        # The generic warmup executor's num_groups argument is num_splits here.
        # A value of 0 represents DeepGEMM's unsplit num_splits=None path.
        self.num_splits = num_groups if num_groups > 0 else None

    def execute(self, m):
        if self.num_splits is None:
            out = torch.empty((m, self.n), device="cuda", dtype=torch.float32)
            sqrsum = torch.empty((m,), device="cuda", dtype=torch.float32)
        else:
            # Slicing the middle dimension of a preallocated
            # (num_splits, max_m, n) output would create a strided view.
            out = torch.empty(
                (self.num_splits, m, self.n), device="cuda", dtype=torch.float32
            )
            sqrsum = torch.empty(
                (self.num_splits, m), device="cuda", dtype=torch.float32
            )
        deep_gemm.tf32_hc_prenorm_gemm(
            self.x[:m],
            self.fn,
            out,
            sqrsum,
            num_splits=self.num_splits,
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


def pp_parallel_deep_gemm_warmup(model_runner) -> None:
    """Run per-PP-rank dummy DECODE+EXTEND forwards so each rank's
    DeepGEMM JIT compiles in parallel instead of serially via the warmup
    /generate flowing through the pipeline. Opt-in via
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.
    """
    # n_splits ~= n_sms / ceil(bs/block_m) with block_m=64; sweep 5 bs to
    # cover the brackets real /generate hits (smallest decode shape,
    # mid-low, two mid, and n_splits=1 for ~5K+ token prefill). Ceil-align
    # bs to the CP padding alignment (cp_size, or 2*cp_size for DSA
    # in-seq-split). _dummy_run does not pad q/hidden like the real flow, so
    # an unaligned bs makes DSA's padded num_splits longer than the q tokens
    # and trips FlashMLA's "num_splits must have shape (b+1)" check.
    from sglang.srt.layers.utils.cp_utils import get_cp_padding_align_size

    n_sms = torch.cuda.get_device_properties(model_runner.device).multi_processor_count
    block_m = 64
    cp = max(get_cp_padding_align_size(), 1)
    batch_sizes = sorted(
        {
            ceil_align(bs, cp)
            for bs in (
                1,
                2 * block_m,
                max(n_sms // 8, 2) * block_m,
                max(n_sms // 4, 4) * block_m,
                n_sms * block_m,
            )
        }
    )

    # In PD, prefill-only nodes never decode (indexer would OOM at large
    # bs) and decode-only nodes never extend.
    disagg_mode = model_runner.server_args.disaggregation_mode
    run_decode = model_runner.is_generation and disagg_mode != "prefill"
    run_extend = disagg_mode != "decode"

    logger.info(
        "PP-parallel DeepGEMM warmup start "
        "(pp_rank=%d, tp_rank=%d, batch_sizes=%s, disagg=%s).",
        model_runner.pp_rank,
        model_runner.tp_rank,
        batch_sizes,
        disagg_mode,
    )

    t0 = time.perf_counter()
    with torch.inference_mode():
        for bs in batch_sizes:
            if run_decode:
                model_runner._dummy_run(
                    batch_size=bs, forward_mode_override=ForwardMode.DECODE
                )
            if run_extend:
                model_runner._dummy_run(
                    batch_size=bs, forward_mode_override=ForwardMode.EXTEND
                )

    logger.info(
        "PP-parallel DeepGEMM warmup done in %.2fs (pp_rank=%d).",
        time.perf_counter() - t0,
        model_runner.pp_rank,
    )
