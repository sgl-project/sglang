"""Runtime entrypoints for TileLang FP8 blockwise GEMM."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.tilelang_gemm_wrapper.availability import assert_available
from sglang.srt.layers.tilelang_gemm_wrapper.configs import (
    SPLIT_K_KERNEL_TYPES,
    SWAP_AB_KERNEL_TYPES,
    SelectedConfigStore,
    config_compatibility_error,
    default_config,
    generate_candidate_configs,
    validate_search_policy,
    write_selected_config_file,
)
from sglang.srt.layers.tilelang_gemm_wrapper.tuning import make_autotune_metadata

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_DO_COMPILE = True
_SELECTED_CONFIGS: Dict[Tuple[int, int, int], dict] = {}
_PARTIAL_BUFFER_CACHE: Dict[Tuple[str, int, int, str, str], torch.Tensor] = {}
_CONFIG_STORE = SelectedConfigStore()
_CONFIG_PATH_LOADED: Optional[str] = None
_AUTOTUNE_BACKENDS = ("event", "cupti", "cudagraph")
_AUTOTUNE_BACKEND = Literal["event", "cupti", "cudagraph"]


def update_tilelang_config(gpu_id: int, server_args: "ServerArgs") -> None:
    """Update TileLang runtime config.

    The first-rank policy is wired here so warmup/autotune can run before CUDA
    graph capture.
    """

    global _DO_COMPILE

    assert_available()

    _DO_COMPILE = server_args.base_gpu_id == gpu_id

    logger.info(
        "TileLang FP8 GEMM config updated: do_compile=%s",
        _DO_COMPILE,
    )


def load_selected_configs(path: str) -> None:
    """Load exported TileLang selected configs for reproducible runs."""

    global _CONFIG_STORE, _CONFIG_PATH_LOADED

    _CONFIG_STORE = SelectedConfigStore.from_path(path)
    _CONFIG_PATH_LOADED = path
    _SELECTED_CONFIGS.clear()
    for config in _CONFIG_STORE.as_list():
        _record_selected_config(config)
    logger.info(
        "Loaded %s TileLang FP8 GEMM selected configs from %s",
        len(_CONFIG_STORE.as_list()),
        path,
    )


def merge_selected_configs(path: str) -> None:
    """Merge selected configs into the current store without clearing it."""

    store = SelectedConfigStore.from_path(path)
    _CONFIG_STORE.update(store)
    for config in store.as_list():
        _record_selected_config(config)
    logger.info(
        "Merged %s TileLang FP8 GEMM selected configs from %s",
        len(store.as_list()),
        path,
    )


def _ensure_selected_configs_loaded() -> None:
    config_path = envs.SGLANG_TILELANG_GEMM_CONFIG_PATH.get()
    if not config_path or _CONFIG_PATH_LOADED == config_path:
        return
    load_selected_configs(config_path)


def _select_config(M: int, N: int, K: int) -> dict:
    """Return the selected kernel config for a concrete shape.

    Exported configs can select any supported kernel family. Without an exported
    config, the fallback is intentionally conservative until SM89/SM90 validation
    selects tuned defaults.
    """

    _ensure_selected_configs_loaded()
    if _CONFIG_STORE.configs_by_nk:
        return _CONFIG_STORE.select(M, N, K)
    return default_config(M, N, K)


def _validate_config(config: dict) -> None:
    error = config_compatibility_error(
        config, int(config["M"]), int(config["N"]), int(config["K"])
    )
    if error is not None:
        raise RuntimeError(
            "TileLang FP8 GEMM selected config is incompatible with the "
            f"requested shape: {error}."
        )


@lru_cache(maxsize=256)
def _get_kernel(
    kernel_type: str,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    split_k: int,
    out_dtype: str,
    accum_dtype: str,
    c_scale_local: bool,
    a_scale_shm: bool,
    b_scale_shm: bool,
    swizzle_panel: int,
    swizzle_order: str,
):
    assert_available()

    from sglang.srt.layers.tilelang_gemm_wrapper.kernels import (
        fp8_blockwise_gemm_base_kernel,
        fp8_blockwise_gemm_split_k_kernel,
        fp8_blockwise_gemm_split_k_swap_ab_kernel,
        fp8_blockwise_gemm_swap_ab_kernel,
    )

    common = {
        "N": N,
        "K": K,
        "block_M": block_M,
        "block_N": block_N,
        "block_K": block_K,
        "num_stages": num_stages,
        "threads": threads,
        "out_dtype": out_dtype,
        "accum_dtype": accum_dtype,
        "c_scale_local": c_scale_local,
    }

    if kernel_type == "base":
        return fp8_blockwise_gemm_base_kernel(
            **common,
            a_scale_shm=a_scale_shm,
            swizzle_panel=swizzle_panel,
            swizzle_order=swizzle_order,
        )
    if kernel_type == "swapAB":
        return fp8_blockwise_gemm_swap_ab_kernel(**common, b_scale_shm=b_scale_shm)
    if kernel_type == "splitK":
        return fp8_blockwise_gemm_split_k_kernel(
            **common, split_k=split_k, a_scale_shm=a_scale_shm
        )
    if kernel_type == "splitK_swapAB":
        return fp8_blockwise_gemm_split_k_swap_ab_kernel(
            **common, split_k=split_k, b_scale_shm=b_scale_shm
        )

    raise RuntimeError(f"Unknown TileLang FP8 GEMM kernel type: {kernel_type}.")


def _compile_from_config(config: dict):
    return _get_kernel(
        config["kernel_type"],
        config["N"],
        config["K"],
        config["block_M"],
        config["block_N"],
        config["block_K"],
        config["num_stages"],
        config["threads"],
        config["split_k"],
        config["out_dtype"],
        config["accum_dtype"],
        config["c_scale_local"],
        config["a_scale_shm"],
        config["b_scale_shm"],
        config.get("swizzle_panel", 0),
        config.get("swizzle_order", "row"),
    )


def _record_selected_config(config: dict) -> None:
    key = (config["M"], config["N"], config["K"])
    _SELECTED_CONFIGS[key] = dict(config)


def _get_partial_buffer(
    kernel_type: str,
    split_k: int,
    M: int,
    N: int,
    device: torch.device,
    dtype: str,
) -> torch.Tensor:
    device_key = str(device)
    key = (kernel_type, split_k, N, device_key, dtype)
    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    if key not in _PARTIAL_BUFFER_CACHE or _PARTIAL_BUFFER_CACHE[key].shape[1] < M:
        _PARTIAL_BUFFER_CACHE[key] = torch.zeros(
            (split_k, M, N), device=device, dtype=torch_dtype
        )
    buffer = _PARTIAL_BUFFER_CACHE[key]
    # Return a compact contiguous view for the requested M even when the cached
    # backing buffer has grown larger than this call needs.
    return buffer.as_strided((split_k, M, N), (M * N, N, 1))


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_autotune_backend(backend: str) -> _AUTOTUNE_BACKEND:
    if backend not in _AUTOTUNE_BACKENDS:
        raise ValueError(
            "TileLang FP8 GEMM autotune backend must be one of "
            f"{_AUTOTUNE_BACKENDS}, got {backend}."
        )
    return backend  # type: ignore[return-value]


def _make_autotune_inputs(M: int, N: int, K: int) -> tuple[torch.Tensor, ...]:
    A_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cuda")
    A_scale = torch.ones(
        (M, _ceildiv(K, 128)),
        dtype=torch.float32,
        device="cuda",
    )
    B_fp8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device="cuda")
    B_scale = torch.ones(
        (_ceildiv(N, 128), _ceildiv(K, 128)),
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    return A_fp8, A_scale, B_fp8, B_scale, out


def _kernel_args_from_inputs(
    config: dict,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    A_fp8, A_scale, B_fp8, B_scale, out = inputs
    kernel_type = config["kernel_type"]

    if kernel_type in SPLIT_K_KERNEL_TYPES:
        partial = _get_partial_buffer(
            kernel_type,
            config["split_k"],
            config["M"],
            config["N"],
            out.device,
            config["accum_dtype"],
        )
        if kernel_type in SWAP_AB_KERNEL_TYPES:
            return B_fp8, B_scale, A_fp8, A_scale, partial, out
        return A_fp8, A_scale, B_fp8, B_scale, partial, out

    if kernel_type in SWAP_AB_KERNEL_TYPES:
        return B_fp8, B_scale, A_fp8, A_scale, out

    return A_fp8, A_scale, B_fp8, B_scale, out


def _benchmark_config_with_tilelang_profiler(
    config: dict,
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    rep: int,
    backend: _AUTOTUNE_BACKEND,
) -> float:
    kernel = _compile_from_config(config)
    kernel_args = _kernel_args_from_inputs(config, inputs)
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(
        warmup=warmup,
        rep=rep,
        input_tensors=list(kernel_args),
        backend=backend,
        return_mode="mean",
    )
    return float(latency)


def autotune_shape(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int = 25,
    rep: int = 100,
    backend: str = "cudagraph",
    kernel_types: Optional[Iterable[str]] = None,
    max_configs: Optional[int] = None,
    search_policy: str = "family_pruned",
) -> dict:
    """Tune one concrete shape with TileLang's profiler.

    TileLang 0.1.9 exposes a CUDA graph profiler backend. Use it by default so
    small-M decode shapes are measured with the same semantics that motivated
    the historical custom tuner.
    """

    assert_available()

    profile_backend = _validate_autotune_backend(backend)
    policy = validate_search_policy(search_policy)
    candidates = generate_candidate_configs(M, N, K, kernel_types, policy)
    if max_configs is not None and max_configs > 0:
        candidates = candidates[:max_configs]
    if not candidates:
        raise RuntimeError(
            f"No TileLang FP8 GEMM autotune candidates for M={M}, N={N}, K={K}."
        )

    logger.info(
        "Autotuning TileLang FP8 GEMM shape M=%s, N=%s, K=%s with %s configs "
        "using %s profiler backend and %s search policy",
        M,
        N,
        K,
        len(candidates),
        profile_backend,
        policy,
    )

    inputs = _make_autotune_inputs(M, N, K)
    best_config: Optional[dict] = None
    best_latency_ms = float("inf")

    for idx, config in enumerate(candidates):
        _validate_config(config)
        try:
            latency_ms = _benchmark_config_with_tilelang_profiler(
                config,
                inputs,
                warmup=warmup,
                rep=rep,
                backend=profile_backend,
            )
        except Exception:
            logger.debug(
                "TileLang FP8 GEMM autotune config failed at index %s: %s",
                idx,
                config,
                exc_info=True,
            )
            continue

        if latency_ms < best_latency_ms:
            best_latency_ms = latency_ms
            best_config = dict(config)
            logger.debug(
                "TileLang FP8 GEMM best latency so far: %.6f ms with %s",
                best_latency_ms,
                best_config,
            )

    if best_config is None:
        raise RuntimeError(
            f"TileLang FP8 GEMM autotune failed for M={M}, N={N}, K={K}; "
            "no candidate compiled and profiled successfully."
        )

    best_config.update(
        {
            "tuned_latency_ms": best_latency_ms,
            "tuned_profiler_backend": profile_backend,
            "tuned_search_policy": policy,
            "tuned_warmup": warmup,
            "tuned_rep": rep,
        }
    )
    _CONFIG_STORE.add(best_config)
    _record_selected_config(best_config)

    logger.info(
        "Selected TileLang FP8 GEMM config for M=%s, N=%s, K=%s: %.6f ms, %s",
        M,
        N,
        K,
        best_latency_ms,
        best_config,
    )
    return dict(best_config)


def autotune_shapes(
    shapes: Iterable[Tuple[int, int, int]],
    *,
    warmup: int = 25,
    rep: int = 100,
    backend: str = "cudagraph",
    kernel_types: Optional[Iterable[str]] = None,
    max_configs: Optional[int] = None,
    search_policy: str = "family_pruned",
    checkpoint_config_path: Optional[str] = None,
    resume_config_path: Optional[str] = None,
    export_metadata: Optional[dict] = None,
) -> list[dict]:
    """Tune concrete (M, N, K) shapes and return selected configs."""

    if not _DO_COMPILE:
        return []

    if resume_config_path:
        merge_selected_configs(resume_config_path)
    if checkpoint_config_path:
        try:
            merge_selected_configs(checkpoint_config_path)
        except FileNotFoundError:
            pass

    selected = []
    for M, N, K in shapes:
        existing_config = _CONFIG_STORE.get_exact_compatible(M, N, K)
        if existing_config is not None:
            logger.info(
                "Skipping TileLang FP8 GEMM autotune for M=%s, N=%s, K=%s; "
                "selected config already exists.",
                M,
                N,
                K,
            )
            _record_selected_config(existing_config)
            selected.append(existing_config)
            continue

        selected.append(
            autotune_shape(
                M,
                N,
                K,
                warmup=warmup,
                rep=rep,
                backend=backend,
                kernel_types=kernel_types,
                max_configs=max_configs,
                search_policy=search_policy,
            )
        )
        if checkpoint_config_path:
            export_selected_configs(checkpoint_config_path, metadata=export_metadata)
    return selected


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> None:
    """Compute out = A @ B^T for FP8 inputs with FP32 block scales."""

    A_fp8, A_scale = lhs
    B_fp8, B_scale = rhs
    M, K = A_fp8.shape
    N, weight_k = B_fp8.shape
    if K != weight_k:
        raise RuntimeError(
            f"TileLang FP8 GEMM got mismatched K dimensions: A K={K}, B K={weight_k}."
        )
    config = _select_config(M, N, K)
    _validate_config(config)
    kernel = _compile_from_config(config)
    _record_selected_config(config)

    kernel_type = config["kernel_type"]
    if kernel_type in SPLIT_K_KERNEL_TYPES:
        partial = _get_partial_buffer(
            kernel_type,
            config["split_k"],
            M,
            N,
            out.device,
            config["accum_dtype"],
        )
        if kernel_type in SWAP_AB_KERNEL_TYPES:
            kernel(B_fp8, B_scale, A_fp8, A_scale, partial, out)
        else:
            kernel(A_fp8, A_scale, B_fp8, B_scale, partial, out)
    elif kernel_type in SWAP_AB_KERNEL_TYPES:
        kernel(B_fp8, B_scale, A_fp8, A_scale, out)
    else:
        kernel(A_fp8, A_scale, B_fp8, B_scale, out)


def warmup_or_autotune_shapes(
    shapes: Iterable[Tuple[int, int, int]],
    *,
    autotune: Optional[bool] = None,
    warmup: int = 25,
    rep: int = 100,
    backend: Optional[str] = None,
    kernel_types: Optional[Iterable[str]] = None,
    max_configs: Optional[int] = None,
    search_policy: Optional[str] = None,
    checkpoint_config_path: Optional[str] = None,
    resume_config_path: Optional[str] = None,
) -> None:
    """Compile kernels for concrete (M, N, K) shapes on the compile rank.

    The default path compiles selected configs. If autotune is enabled, the
    shape is tuned first with TileLang's profiler and the selected config is
    retained for later export.
    """

    if not _DO_COMPILE:
        return

    assert_available()
    if autotune is None:
        autotune = envs.SGLANG_TILELANG_GEMM_AUTOTUNE.get()
    if backend is None:
        backend = envs.SGLANG_TILELANG_GEMM_AUTOTUNE_BACKEND.get()
    if search_policy is None:
        search_policy = envs.SGLANG_TILELANG_GEMM_AUTOTUNE_POLICY.get()
    if max_configs is None:
        env_max_configs = envs.SGLANG_TILELANG_GEMM_AUTOTUNE_MAX_CONFIGS.get()
        max_configs = env_max_configs if env_max_configs > 0 else None

    if autotune:
        autotune_shapes(
            shapes,
            warmup=warmup,
            rep=rep,
            backend=backend,
            kernel_types=kernel_types,
            max_configs=max_configs,
            search_policy=search_policy,
            checkpoint_config_path=checkpoint_config_path,
            resume_config_path=resume_config_path,
            export_metadata=make_autotune_metadata(
                backend,
                search_policy,
                warmup,
                rep,
                max_configs,
                kernel_types,
            ),
        )
        return

    for M, N, K in shapes:
        config = _select_config(M, N, K)
        _validate_config(config)
        _compile_from_config(config)
        _record_selected_config(config)


def get_candidate_configs(
    M: int,
    N: int,
    K: int,
    kernel_types: Optional[Iterable[str]] = None,
    search_policy: str = "full",
) -> list[dict]:
    """Return legal configs for benchmark/autotune tooling."""

    return generate_candidate_configs(M, N, K, kernel_types, search_policy)


def has_selected_config(M: int, N: int, K: int) -> bool:
    """Return whether a compatible exact selected config is loaded."""

    _ensure_selected_configs_loaded()
    return _CONFIG_STORE.get_exact_compatible(M, N, K) is not None


def get_kernel_info(M: int, N: int, K: int) -> dict:
    """Return the config that would be selected for a shape."""

    config = _select_config(M, N, K)
    _validate_config(config)
    return dict(config)


def list_available_configs() -> list[Tuple[int, int]]:
    """List (N, K) pairs available from loaded selected configs."""

    _ensure_selected_configs_loaded()
    return sorted(_CONFIG_STORE.configs_by_nk)


def export_selected_configs(path: str, metadata: Optional[dict] = None) -> None:
    """Export selected configs for reproducible benchmark runs."""

    write_selected_config_file(
        path,
        [v for _, v in sorted(_SELECTED_CONFIGS.items())],
        metadata=metadata,
    )


def clear_runtime_cache() -> None:
    """Clear compiled kernels and temporary buffers, preserving selected configs."""

    _get_kernel.cache_clear()
    _PARTIAL_BUFFER_CACHE.clear()


def clear_cache() -> None:
    """Reset TileLang wrapper cache and loaded selected-config state."""

    global _CONFIG_STORE, _CONFIG_PATH_LOADED

    clear_runtime_cache()
    _SELECTED_CONFIGS.clear()
    _CONFIG_STORE = SelectedConfigStore()
    _CONFIG_PATH_LOADED = None
