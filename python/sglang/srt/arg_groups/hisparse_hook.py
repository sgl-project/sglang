from __future__ import annotations

import json
import logging
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")
RUNTIME_SPARSE_BACKENDS_BY_ALGORITHM = {
    "quest": {"fa3", "flashattention"},
}
RUNTIME_SPARSE_ALGORITHMS = set(RUNTIME_SPARSE_BACKENDS_BY_ALGORITHM)
RUNTIME_SPARSE_ATTENTION_BACKEND_ALIASES = {"flashattention": "fa3"}
RUNTIME_SPARSE_CUDA_GRAPH_PROVIDER = "sglang.srt.mem_cache.sparsity.cuda_graph_support"


def _load_hisparse_config(server_args: ServerArgs) -> dict:
    if not server_args.hisparse_config:
        return {}
    try:
        config = json.loads(server_args.hisparse_config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    if not isinstance(config, dict):
        raise ValueError(
            f"hisparse_config must be a JSON object, got {type(config).__name__}."
        )
    return config


def get_hisparse_algorithm(server_args: ServerArgs) -> str | None:
    algorithm = _load_hisparse_config(server_args).get("algorithm")
    return algorithm.strip().lower() if isinstance(algorithm, str) else None


def get_hisparse_backend(server_args: ServerArgs) -> str | None:
    backend = _load_hisparse_config(server_args).get("backend")
    return backend.strip().lower() if isinstance(backend, str) else None


def get_hisparse_page_size(server_args: ServerArgs):
    return _load_hisparse_config(server_args).get("page_size")


def get_hisparse_attention_backend(server_args: ServerArgs) -> str | None:
    backend = get_hisparse_backend(server_args)
    if backend is None:
        return None
    return RUNTIME_SPARSE_ATTENTION_BACKEND_ALIASES.get(backend, backend)


def get_runtime_sparse_backends(server_args: ServerArgs) -> set[str]:
    algorithm = get_hisparse_algorithm(server_args)
    return RUNTIME_SPARSE_BACKENDS_BY_ALGORITHM.get(algorithm, set())


def use_runtime_sparse_attention(server_args: ServerArgs) -> bool:
    return (
        server_args.enable_hisparse
        and get_hisparse_algorithm(server_args) in RUNTIME_SPARSE_ALGORITHMS
    )


def use_runtime_sparse_cuda_graph(server_args: ServerArgs) -> bool:
    if not use_runtime_sparse_attention(server_args):
        return False
    try:
        provider_spec = find_spec(RUNTIME_SPARSE_CUDA_GRAPH_PROVIDER)
    except ModuleNotFoundError as exc:
        if not RUNTIME_SPARSE_CUDA_GRAPH_PROVIDER.startswith(f"{exc.name}."):
            raise
        return False
    if provider_spec is None:
        return False

    import torch

    from sglang.srt.mem_cache.sparsity.factory import (
        _create_sparse_algorithm,
        parse_runtime_sparse_config,
    )

    config = parse_runtime_sparse_config(server_args)
    device = torch.device("cuda")
    algorithm = _create_sparse_algorithm(config, device)
    provider_module = import_module(RUNTIME_SPARSE_CUDA_GRAPH_PROVIDER)
    preflight = getattr(provider_module, "is_runtime_sparse_cuda_graph_available", None)
    if not callable(preflight):
        raise RuntimeError(
            f"{RUNTIME_SPARSE_CUDA_GRAPH_PROVIDER} must define "
            "is_runtime_sparse_cuda_graph_available(coordinator)"
        )
    return bool(
        preflight(
            SimpleNamespace(
                config=config,
                algorithm=algorithm,
                device=device,
            )
        )
    )


def _is_hip() -> bool:
    from sglang.srt.server_args import is_hip

    return is_hip()


def _is_cuda() -> bool:
    from sglang.srt.server_args import is_cuda

    return is_cuda()


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if _is_hip():
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if _is_hip():
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use --dsa-{label}-backend="
            f"{_hisparse_default_backend(kv_cache_dtype)} "
            "or omit it."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""
    if not server_args.enable_hisparse:
        return

    if use_runtime_sparse_attention(server_args):
        from sglang.srt.arg_groups.overrides import attention_backends_of
        from sglang.srt.configs.hybrid_arch import mambaish_config
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.sparsity import parse_runtime_sparse_config
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

        config = parse_runtime_sparse_config(server_args)
        subject = f"Sparse attention algorithm {config.algorithm!r}"
        supported_backends = get_runtime_sparse_backends(server_args)
        if _is_hip():
            raise ValueError(
                f"{subject} does not yet support ROCm; Quest requires CUDA FA3."
            )
        if not _is_cuda():
            raise ValueError(f"{subject} currently requires the NVIDIA CUDA FA3 path.")
        if envs.SGLANG_USE_HND_KVCACHE.get():
            raise ValueError(
                f"{subject} does not support HND KV-cache layout; use NHD layout."
            )
        if config.backend not in supported_backends:
            raise ValueError(
                f"{subject} supports backend values {sorted(supported_backends)}, "
                f"but got {config.backend!r}."
            )
        if not server_args.disable_radix_cache:
            raise ValueError(f"{subject} currently requires --disable-radix-cache.")
        prefill_backend, decode_backend = attention_backends_of(server_args)
        if any(
            backend not in supported_backends
            for backend in (prefill_backend, decode_backend)
        ):
            raise ValueError(
                f"{subject} requires FA3 for prefill and decode, but got "
                f"prefill={prefill_backend}, decode={decode_backend}."
            )
        unsupported_options = (
            (server_args.speculative_algorithm is not None, "speculative decoding"),
            (server_args.dllm_algorithm is not None, "diffusion LLM inference"),
            (server_args.enable_pdmux, "PD multiplexing"),
            (server_args.disaggregation_mode != "null", "PD disaggregation"),
            (server_args.enable_two_batch_overlap, "two-batch overlap"),
            (server_args.enable_mixed_chunk, "mixed chunked prefill"),
            (server_args.enable_torch_compile, "torch.compile"),
            (server_args.enable_dp_attention, "DP attention"),
            (server_args.enable_prefill_cp, "prefill context parallelism"),
            (server_args.attn_cp_size > 1, "attention context parallelism"),
            (server_args.dcp_size > 1, "decode context parallelism"),
        )
        for enabled, label in unsupported_options:
            if enabled:
                raise ValueError(f"{subject} does not yet support {label}.")

        model_config = server_args.get_model_config()
        if model_config.attention_arch != AttentionArch.MHA:
            raise ValueError(
                f"{subject} currently supports standard MHA/GQA models only."
            )
        if model_config.is_encoder_decoder:
            raise ValueError(f"{subject} does not support encoder-decoder models.")
        if model_config.is_multimodal:
            raise ValueError(f"{subject} does not support multimodal models.")
        if not model_config.is_generation:
            raise ValueError(f"{subject} is only supported for generation models.")
        if model_config.num_attention_layers != model_config.num_hidden_layers:
            raise ValueError(
                f"{subject} requires one attention layer per hidden layer."
            )
        sliding_window_size = model_config.sliding_window_size
        has_sliding_window = isinstance(sliding_window_size, (int, float)) and (
            sliding_window_size > -1
        )
        if (
            model_config.is_hybrid_swa
            or has_sliding_window
            or model_config.attention_chunk_size is not None
        ):
            raise ValueError(f"{subject} does not support sliding/local attention.")
        if mambaish_config(model_config) is not None:
            raise ValueError(f"{subject} does not support hybrid linear attention.")
        num_kv_shared_layers = getattr(
            model_config.hf_text_config, "num_kv_shared_layers", 0
        )
        if (
            isinstance(num_kv_shared_layers, int)
            and not isinstance(num_kv_shared_layers, bool)
            and num_kv_shared_layers > 0
        ):
            raise ValueError(f"{subject} does not support cross-layer KV sharing.")

        graph_available = use_runtime_sparse_cuda_graph(server_args)
        locked = getattr(server_args, "_cuda_graph_config_locked", set())
        for phase, phase_config in (
            (Phase.DECODE, server_args.cuda_graph_config.decode),
            (Phase.PREFILL, server_args.cuda_graph_config.prefill),
        ):
            graph_supported = phase == Phase.DECODE and graph_available
            if (
                not graph_supported
                and (phase, "backend") in locked
                and phase_config.backend != Backend.DISABLED
            ):
                raise ValueError(
                    f"{subject} requires {phase.value} CUDA graph to be disabled."
                )
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_hip = _is_hip()
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    from sglang.srt.arg_groups.overrides import resolved_view

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)


def apply_runtime_sparse_eager_defaults(server_args: ServerArgs) -> None:
    """Keep unsupported Quest phases eager without constraining merge order."""
    if not use_runtime_sparse_attention(server_args):
        return

    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    locked = getattr(server_args, "_cuda_graph_config_locked", set())
    graph_available = use_runtime_sparse_cuda_graph(server_args)
    phases = [(Phase.PREFILL, server_args.cuda_graph_config.prefill)]
    if not graph_available:
        phases.append((Phase.DECODE, server_args.cuda_graph_config.decode))
    for phase, phase_config in phases:
        if (phase, "backend") not in locked:
            phase_config.backend = Backend.DISABLED
