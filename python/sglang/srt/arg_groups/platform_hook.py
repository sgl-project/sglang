# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the per-platform backend defaults."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    record_foreign_defaults,
    resolving_view,
)
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import is_host_cpu_arm64, is_ppu

logger = logging.getLogger(__name__)


def handle_ppu_backends(server_args: Any):
    # PPU INT8 MoE / linear layers run on the acext backend. Initialize the
    # acext CUDA path (defaulting it on) and validate the acext version, then
    # size its token limit from the resolved chunked-prefill size.
    if not is_ppu():
        return
    from sglang.srt.environ import envs
    from sglang.srt.utils.common import (
        check_acext_version_compatibility,
        set_acext_token_limit,
    )

    cfg = resolving_view(server_args)
    if not envs.SGLANG_SAIL_USE_ACEXT_CUDA.is_set():
        envs.SGLANG_SAIL_USE_ACEXT_CUDA.set(True)
    if envs.SGLANG_SAIL_USE_ACEXT_CUDA.get():
        check_acext_version_compatibility()
    # deep gemm init
    if not envs.SGLANG_SAIL_DEEPGEMM_DENSE.is_set():
        envs.SGLANG_SAIL_DEEPGEMM_DENSE.set(True)
    if not envs.SGLANG_SAIL_DEEPGEMM_MOE.is_set():
        envs.SGLANG_SAIL_DEEPGEMM_MOE.set(True)
    # DSA FlashMLA decode compute stays non-FP8 on PPU by default.
    if not envs.SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8.is_set():
        envs.SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8.set(False)
    # [q_v-not-support fallback] The PPU vendor fa3 library rejects the q_v
    # argument ("q_v is only supported for Hopper GPUs") and the DSA fa3
    # decode path always passes q_v, so the sub-SM90 CUDA default ("fa3")
    # crashes at CUDA-graph capture. Default the DSA decode backend to
    # flashmla_kv, the PPU-validated DSA decode impl; the DSA
    # split-backend resolution treats a non-None value as user-set and
    # keeps it. An explicit "--dsa-decode-backend fa3" fails fast with an
    # actionable message instead of the vendor crash at graph capture.
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if is_deepseek_dsa(model_config_of(server_args).hf_config):
        if cfg.dsa_decode_backend is None:
            declare_resolution(
                server_args,
                "_handle_ppu_backends",
                dsa_decode_backend="flashmla_kv",
            )
            logger.info(
                "PPU fa3 does not support q_v; defaulting the DSA decode "
                "backend to flashmla_kv."
            )
        elif cfg.dsa_decode_backend == "fa3":
            raise ValueError(
                "--dsa-decode-backend fa3 is not supported on PPU: the fa3 "
                'vendor library rejects q_v ("q_v is only supported for '
                'Hopper GPUs") and the DSA fa3 decode path requires it. Use '
                "--dsa-decode-backend flashmla_kv instead (the PPU default)."
            )
    # Use CUDA PLA fast path by default on PPU.
    if not envs.SGLANG_SAIL_PLA_CUDA.is_set():
        envs.SGLANG_SAIL_PLA_CUDA.set(True)
    # Use CUDA FLA (Flash Linear Attention) fast path by default on PPU.
    if not envs.SGLANG_SAIL_FLA_CUDA.is_set():
        envs.SGLANG_SAIL_FLA_CUDA.set(True)
    # Disable DeepSeek-V4 topk_v2 on PPU by default.
    if not envs.SGLANG_OPT_USE_TOPK_V2.is_set():
        envs.SGLANG_OPT_USE_TOPK_V2.set(False)
    # K3's Blackwell-specific fused paths cannot run on PPU. Keep the
    # flags false so an explicit environment setting cannot re-enable them.
    # SGLANG_K3_FUSED_FRONT stays at its default (True): the PPU W4A16
    # MMA/fused-MoE path relies on the fused MoE front.
    for field in (
        envs.SGLANG_K3_AR_FUSION,
        envs.SGLANG_K3_SP_COLLECTIVE,
        envs.SGLANG_K3_SP_ATTN_RES,
        envs.SGLANG_K3_GEMM_AR,
    ):
        field.set(False)
    # Disable shared experts fusion for mxfp4 and mixed_precision_w4: those
    # models use a different quant method for routed experts and shared
    # experts, so the fused shared-expert path cannot run.
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.utils.common import get_quantization_config

    quant_method = get_quantization_config(model_config_of(server_args).hf_config)
    if quant_method in ("mxfp4", "mixed_precision_w4"):
        declare_resolution(
            server_args,
            "_handle_ppu_backends",
            disable_shared_experts_fusion=True,
        )
        logger.info(
            f"{quant_method} model uses different quant method for routed experts and shared experts. --disable-shared-experts-fusion is automatically set."
        )
    # Disable custom allreduce by default on PPU (use pccl allreduce for perf).
    if not cfg.disable_custom_all_reduce:
        declare_resolution(
            server_args,
            "_handle_ppu_backends",
            disable_custom_all_reduce=True,
        )
        logger.info(
            "Disable custom allreduce and use pccl allreduce on ppu for better perf."
        )
    if cfg.chunked_prefill_size is not None:
        set_acext_token_limit(
            acext_num_tokens=int(cfg.chunked_prefill_size * 0.5),
        )


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

        current = cfg.cuda_graph_config.prefill.tc_compiler
        if current is not None and current != "eager":
            logger.warning(
                "At this moment Ascend platform only support prefill graph compilation with "
                "cuda_graph_config[prefill].tc_compiler='eager'."
            )
            declare_resolution(
                server_args,
                "_handle_npu_backends",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, tc_compiler="eager"
                ),
            )


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
