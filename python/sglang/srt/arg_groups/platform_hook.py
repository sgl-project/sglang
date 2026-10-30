# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the per-platform backend defaults."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    declare_resolution,
    model_config_of,
    record_foreign_defaults,
    register_post_process,
    resolved_view,
    resolving_view,
    run_post_process_pass,
    use_mla_backend,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import is_host_cpu_arm64

logger = logging.getLogger(__name__)


def handle_hardware_runtime_validation(server_args: Any):
    # `server_args` is accepted, not read: every resolution-hook step takes
    # it, uniformly, so `run_hook` never has to special-case an arity. The
    # check below is intentionally independent of `server_args.device`:
    # setting SGLANG_USE_MLX opts into the MLX backend and must fail
    # immediately if the environment cannot honor that request. With the
    # flag unset, use_mlx() remains lazy and does not import MLX.
    use_mlx()


def handle_npu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "npu":
        from sglang.srt.hardware_backend.npu.utils import set_default_server_args

        set_default_server_args(server_args)


def handle_mps_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "mps":
        if not use_mlx():
            declare_resolution(
                server_args,
                "_handle_mps_backends",
                disable_overlap_schedule=True,
            )


def handle_amd_specifics(server_args: Any):
    if get_platform().is_hip:
        declare_resolution(
            server_args, "_handle_amd_specifics", triton_attention_num_kv_splits=16
        )
        # Above this the HIP runtime registers a pageable H2D source with the
        # GPU rather than staging it, and the MMU notifier on that registration
        # evicts our KFD queues once per tensor while weights load. In KB.
        os.environ.setdefault("GPU_PINNED_MIN_XFER_SIZE", str(4 * 1024 * 1024))


def handle_nccl_pre_warm(server_args: Any):
    # pre_warm_nccl is only used with CUDA or HIP hardware or NPU hardware
    cfg = resolving_view(server_args)
    if cfg.pre_warm_nccl and not (
        get_platform().is_cuda or get_platform().is_hip or get_platform().is_npu
    ):
        logger.warning(
            "pre_warm_nccl is only applicable for CUDA or HIP hardware or NPU hardware. "
            "Ignoring pre_warm_nccl setting on current hardware."
        )
        declare_resolution(server_args, "_handle_nccl_pre_warm", pre_warm_nccl=False)


def handle_platform_defaults(server_args: Any):
    """An out-of-tree platform's defaults, declared like every rule beside it.

    `Platform.apply_server_args_defaults` is a plugin interface: the platform is
    handed a configuration and assigns the fields it wants defaulted. In-tree
    platforms do not implement it -- the base is a no-op and nothing overrides
    it -- so this captures nothing here and exists for the platforms that live
    outside this tree.

    Ordering: it must precede `handle_gpu_memory_settings`, whose symm-mem
    prealloc default keys off `enable_symm_mem`.
    """
    # `current_platform` is the plugin object; `get_platform()` is the facts
    # view over it and carries neither the name nor the hook.
    from sglang.srt.platforms import current_platform

    record_foreign_defaults(
        server_args,
        f"platform:{current_platform.device_name}",
        current_platform.apply_server_args_defaults,
    )


def handle_symm_mem_device_support(server_args: Any):
    cfg = resolving_view(server_args)
    # The symm-mem allocator compiles a CUDA plugin and links -lnccl, so off
    # CUDA/HIP (e.g. Ascend NPU) it fails deep in a build step rather than here.
    if cfg.enable_symm_mem and not (get_platform().is_cuda or get_platform().is_hip):
        logger.warning(
            "--enable-symm-mem is not supported on non CUDA/HIP devices "
            "(NCCL symmetric memory is unavailable). Disabling symmetric memory."
        )
        declare_resolution(
            server_args, "_handle_symm_mem_device_support", enable_symm_mem=False
        )


def handle_xpu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "xpu":
        if cfg.sampling_backend is None:
            declare_resolution(
                server_args,
                "_handle_xpu_backends",
                sampling_backend="intel_xpu",
            )
        # Enable the fused qknorm + rope by default as the fused q/k-norm + RoPE kernel is a pure win here.
        if cfg.enable_fused_qk_norm_rope is None:
            declare_resolution(
                server_args,
                "_handle_xpu_backends",
                enable_fused_qk_norm_rope=True,
            )
        # Decode graph is opt-in on XPU: unless the user explicitly set
        # --cuda-graph-backend-decode (or --cuda-graph-config), keep it
        # disabled so the default startup doesn't require graph capture.
        if (Phase.DECODE, "backend") not in server_args._cuda_graph_config_locked:
            declare_resolution(
                server_args,
                "_handle_xpu_backends",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                ),
            )
        elif cfg.cuda_graph_config.decode.backend not in (
            Backend.DISABLED,
            Backend.FULL,
        ):
            logger.warning(
                "XPU platform only supports decode backend 'full'; "
                "disabling unsupported decode backend '%s'.",
                cfg.cuda_graph_config.decode.backend,
            )
            declare_resolution(
                server_args,
                "_handle_xpu_backends",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                ),
            )


def handle_sail_backends(server_args: Any):
    if not get_platform().is_sail:
        return
    from sglang.srt.configs.model_config import is_deepseek_dsa

    cfg = resolving_view(server_args)
    # A captured prefill graph pins MLA to the absorbed path, which hands the
    # backend a non-None q_v; the SAIL FA3 interface rejects that argument, so
    # MLA prefill must stay eager on SAIL. DSA models keep their own policy.
    # Decode has no eager escape hatch and is routed off fa3 afterwards, in
    # handle_sail_attention_backend_compatibility.
    if use_mla_backend(server_args) and not is_deepseek_dsa(
        model_config_of(server_args).hf_config
    ):
        declare_resolution(
            server_args,
            "_handle_sail_backends",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )


@register_post_process
def _sail_fa3_qv_fallbacks(view: Any) -> dict:
    """Keep MLA off the SAIL FA3 paths that hand the kernel a q_v.

    The vendor FA3 build only accepts q_v on Hopper. Absorbed MLA decode always
    passes qv=q_nope and, unlike prefill, has no eager route to dodge it, so
    decode can never be fa3; prefill keeps fa3 only while it stays eager and
    non-chunked.
    """
    if not get_platform().is_sail:
        return {}
    from sglang.srt.configs.model_config import is_deepseek_dsa

    overrides = {}
    prefill_backend, decode_backend = attention_backends_of(view)

    # Decode: absorbed MLA decode always passes q_v. DSA models keep their own
    # backend policy (see handle_sail_backends).
    if (
        decode_backend == "fa3"
        and use_mla_backend(view)
        and not is_deepseek_dsa(model_config_of(view).hf_config)
    ):
        logger.warning(
            "On SAIL, fa3 cannot serve MLA decode: absorbed MLA decode passes "
            "q_v, which the SAIL FA3 kernel only supports for Hopper. Falling "
            "back to --decode-attention-backend flashmla; fa3 stays the prefill "
            "backend."
        )
        overrides["decode_attention_backend"] = "flashmla"
        decode_backend = "flashmla"

    # Prefill: a captured prefill graph or a disabled chunked prefix cache pins
    # MLA to the absorbed path, so fa3 prefill survives only while it stays
    # eager.
    if (prefill_backend, decode_backend) != ("fa3", "flashmla"):
        return overrides

    mtp_wo_decode_sam = (
        view.speculative_algorithm is not None
        and view.speculative_attention_mode != "decode"
    )
    prefill_cg_enabled = (
        view.cuda_graph_config is not None
        and view.cuda_graph_config.prefill.backend != Backend.DISABLED
    )
    chunked_prefix_disabled = view.disable_chunked_prefix_cache
    if not (mtp_wo_decode_sam or prefill_cg_enabled or chunked_prefix_disabled):
        return overrides

    reasons = []
    if mtp_wo_decode_sam:
        reasons.append("MTP enabled without --speculative-attention-mode decode")
    if prefill_cg_enabled:
        reasons.append("prefill-phase CUDA graph is not disabled")
    if chunked_prefix_disabled:
        reasons.append("--disable-chunked-prefix-cache is set")
    logger.warning(
        "Unsafe SAIL attention backend combo fa3 prefill + flashmla decode: %s. "
        "Falling back to a unified flashmla backend to avoid 'q_v not support'. "
        "To keep fa3 prefill, use --speculative-attention-mode decode, disable "
        "the prefill CUDA graph, and do not set --disable-chunked-prefix-cache.",
        "; ".join(reasons),
    )
    overrides["prefill_attention_backend"] = "flashmla"
    return overrides


@register_post_process
def _sail_flashmla_fp8_kv_cache_fallback(view: Any) -> dict:
    if not get_platform().is_sail or view.kv_cache_dtype not in (
        "fp8_e4m3",
        "fp8_e5m2",
    ):
        return {}

    prefill_backend, decode_backend = attention_backends_of(view)
    incompatible = []
    if "flashmla" in (prefill_backend, decode_backend):
        incompatible.append("flashmla")
    if view.speculative_draft_attention_backend == "flashmla":
        incompatible.append("draft flashmla")
    if view.dsa_prefill_backend == "flashmla_sparse":
        incompatible.append("flashmla_sparse")
    if not incompatible:
        return {}

    logger.warning(
        "On SAIL, %s does not support --kv-cache-dtype %s. "
        "Falling back to --kv-cache-dtype auto.",
        " and ".join(incompatible),
        view.kv_cache_dtype,
    )
    return {"kv_cache_dtype": "auto"}


def handle_sail_attention_backend_compatibility(server_args: Any):
    if not get_platform().is_sail:
        return

    run_post_process_pass(server_args, _sail_fa3_qv_fallbacks)
    run_post_process_pass(server_args, _sail_flashmla_fp8_kv_cache_fallback)
    cfg = resolved_view(server_args)
    if "fa3" in (
        cfg.attention_backend,
        *attention_backends_of(cfg),
        cfg.speculative_draft_attention_backend,
    ):
        envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.set(0)


def handle_cpu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "cpu":
        if cfg.attention_backend is None:
            declare_resolution(
                server_args,
                "_handle_cpu_backends",
                attention_backend=(
                    "torch_native" if is_host_cpu_arm64() else "intel_amx"
                ),
            )
        declare_resolution(
            server_args,
            "_handle_cpu_backends",
            sampling_backend="pytorch",
        )


def handle_hpu_backends(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.device == "hpu":
        declare_resolution(
            server_args,
            "_handle_hpu_backends",
            attention_backend="torch_native",
        )
        declare_resolution(
            server_args,
            "_handle_hpu_backends",
            sampling_backend="pytorch",
        )
