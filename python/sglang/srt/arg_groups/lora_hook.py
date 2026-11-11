# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the LoRA adapters."""

from __future__ import annotations

import logging
from typing import Any, List

from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    use_mla_backend,
)
from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.utils import get_normalized_target_modules
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def _check_v2_lora_modes(cfg: Any, option: str):
    for unsupported, mode in (
        (cfg.attn_dp_size > 1, "DP-attention with --attn-dp-size > 1"),
        (cfg.enable_pdmux, "PD-multiplexing"),
        (cfg.enable_two_batch_overlap, "two-batch overlap"),
        # Both reach the legacy kv_b correction kernels with V2 batch metadata.
        (envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get(), "SGLANG_EXPERIMENTAL_LORA_OPTI"),
        (get_platform().is_hip, "ROCm"),
    ):
        if unsupported:
            raise ValueError(f"{option} does not yet support {mode}")


def parse_lora_paths(lora_paths: Any) -> List[LoRARef]:
    """Normalize configured adapter paths to LoRARef records."""
    if lora_paths is None:
        return []
    if isinstance(lora_paths, dict):
        return [
            LoRARef(
                lora_id=LoRARef.deterministic_id(k, v),
                lora_name=k,
                lora_path=v,
                pinned=False,
            )
            for k, v in lora_paths.items()
        ]
    if not isinstance(lora_paths, list):
        raise ValueError(
            f"Invalid type for --lora-paths: {type(lora_paths)}. "
            "Expected a list or a dictionary."
        )
    parsed_lora_paths = []
    for lora_path in lora_paths:
        if isinstance(lora_path, str):
            if "=" in lora_path:
                name, path = lora_path.split("=", 1)
                lora_ref = LoRARef(
                    lora_id=LoRARef.deterministic_id(name, path),
                    lora_name=name,
                    lora_path=path,
                    pinned=False,
                )
            else:
                lora_ref = LoRARef(
                    lora_id=LoRARef.deterministic_id(lora_path, lora_path),
                    lora_name=lora_path,
                    lora_path=lora_path,
                    pinned=False,
                )
        elif isinstance(lora_path, dict):
            assert "lora_name" in lora_path and "lora_path" in lora_path, (
                f"When providing LoRA paths as a list of dict, each dict should contain 'lora_name' and 'lora_path' keys. Got: {lora_path}"
            )
            lora_ref = LoRARef(
                lora_id=LoRARef.deterministic_id(
                    lora_path["lora_name"], lora_path["lora_path"]
                ),
                lora_name=lora_path["lora_name"],
                lora_path=lora_path["lora_path"],
                pinned=lora_path.get("pinned", False),
            )
        else:
            raise ValueError(
                f"Invalid type for item in --lora-paths list: {type(lora_path)}. "
                "Expected a string or a dictionary."
            )
        parsed_lora_paths.append(lora_ref)
    return parsed_lora_paths


def check_lora_server_args(server_args: Any):
    cfg = resolving_view(server_args)

    assert cfg.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

    # Enable LoRA if any LoRA paths are provided for backward compatibility.
    if cfg.lora_paths:
        if cfg.enable_lora is None:
            declare_resolution(server_args, "check_lora_server_args", enable_lora=True)
            logger.warning(
                "--enable-lora is set to True because --lora-paths is provided."
            )
        elif cfg.enable_lora is False:
            logger.warning(
                "--enable-lora is set to False, any provided lora_paths will be ignored."
            )

    if cfg.enable_lora:
        if cfg.lora_backend == "triton_v2" and not cfg.moe_runner_backend.startswith(
            "lora"
        ):
            _check_v2_lora_modes(cfg, "--lora-backend triton_v2")

        if cfg.enable_lora_overlap_loading is None:
            declare_resolution(
                server_args, "check_lora_server_args", enable_lora_overlap_loading=False
            )

        if cfg.enable_lora_overlap_loading:
            # TODO (glenliu21): use some sort of buffer with eviction instead of enforcing a limit
            max_loaded_loras_limit = cfg.max_loras_per_batch * 2
            assert (
                cfg.max_loaded_loras is not None
                and cfg.max_loaded_loras <= max_loaded_loras_limit
            ), (
                "Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, "
                f"so --max-loaded-loras must be less than or equal to double --max-loras-per-batch: {max_loaded_loras_limit}"
            )

        # Validate compatibility with speculative decoding
        check_lora_speculative_compatibility(server_args)

        declare_resolution(
            server_args,
            "check_lora_server_args",
            lora_paths=parse_lora_paths(cfg.lora_paths),
        )

        # Normalize target modules to a set; keep {"all"} as a sentinel
        # that gets resolved model-awarely in lora_manager.init_lora_shapes().
        if cfg.lora_target_modules:
            declare_resolution(
                server_args,
                "check_lora_server_args",
                lora_target_modules=set(cfg.lora_target_modules),
            )
            if "all" in cfg.lora_target_modules:
                assert len(cfg.lora_target_modules) == 1, (
                    "If 'all' is specified in --lora-target-modules, it should be the only module specified."
                )

        # Ensure sufficient information is provided for LoRA initialization.
        assert cfg.lora_paths or (cfg.max_lora_rank and cfg.lora_target_modules), (
            "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."
        )

        # Validate max_loaded_loras
        if cfg.max_loaded_loras is not None:
            assert cfg.max_loaded_loras >= cfg.max_loras_per_batch, (
                "max_loaded_loras should be greater than or equal to max_loras_per_batch. "
                f"max_loaded_loras={cfg.max_loaded_loras}, max_loras_per_batch={cfg.max_loras_per_batch}"
            )
            assert len(cfg.lora_paths) <= cfg.max_loaded_loras, (
                "The number of LoRA paths should not exceed max_loaded_loras. "
                f"max_loaded_loras={cfg.max_loaded_loras}, lora_paths={len(cfg.lora_paths)}"
            )

        if cfg.max_lora_chunk_size is not None:
            assert (
                16 <= cfg.max_lora_chunk_size <= 128
                and (cfg.max_lora_chunk_size & (cfg.max_lora_chunk_size - 1)) == 0
            ), "--max-lora-chunk-size must be a power of 2 between 16 and 128."

        if cfg.lora_use_virtual_experts:
            logger.info("Virtual expert computation enabled.")

        assert cfg.lora_drain_wait_threshold >= 0.0, (
            "--lora-drain-wait-threshold must be non-negative."
        )

        check_kv_b_lora_prefix_cache(server_args)

    check_lora_moe_runner_args(server_args)


def _lora_targets_kv_b_proj(server_args: Any) -> bool:
    """Whether the CLI LoRA targets name ``kv_b_proj``.

    Targets that only come with --lora-paths adapters resolve at load time and
    are not covered here.
    """
    cfg = resolving_view(server_args)
    targets = cfg.lora_target_modules
    if not targets:
        return False
    if set(targets) == {"all"}:
        return use_mla_backend(server_args)
    return "kv_b_proj" in get_normalized_target_modules(targets)


def check_kv_b_lora_prefix_cache(server_args: Any):
    """Keep trtllm_mla cached prefixes on the absorbed MLA path for kv_b_proj adapters.

    The MHA cached-prefix paths re-project the cached latent through kv_b_proj
    on prefix rows while the wrapped layer runs the adapter with the batch's
    extend-token segments; on trtllm_mla every extend with a cached prefix takes
    that path and faults. With the chunked prefix cache disabled, trtllm_mla
    sends those extends to the absorbed MLA path, which applies the kv_b_proj
    delta per token. Other backends are out of this rule's scope and keep their
    previous behaviour; KV sharding needs the chunked prefix cache.
    """
    if not _lora_targets_kv_b_proj(server_args):
        return
    cfg = resolving_view(server_args)
    if cfg.disable_chunked_prefix_cache or cfg.enable_kv_cache_sharding:
        return
    prefill_backend, _ = attention_backends_of(cfg)
    if prefill_backend != "trtllm_mla":
        return
    declare_resolution(
        server_args,
        "check_kv_b_lora_prefix_cache",
        disable_chunked_prefix_cache=True,
    )
    logger.info(
        "LoRA targets kv_b_proj: chunked prefix cache disabled so cached "
        "prefixes take the absorbed MLA path, which applies the adapter."
    )


def check_lora_moe_runner_args(server_args: Any):
    """Reject unsupported LoRA MoE configurations before layer attachment."""
    cfg = resolving_view(server_args)
    if (
        cfg.speculative_algorithm is not None
        and cfg.speculative_moe_runner_backend is not None
        and cfg.speculative_moe_runner_backend.startswith("lora")
    ):
        # Draft models have no LoRA adapters.
        raise ValueError(
            f"--speculative-moe-runner-backend {cfg.speculative_moe_runner_backend} "
            "is not supported: the LoRA MoE runner serves adapter traffic on the "
            "target model"
        )
    backend = cfg.moe_runner_backend
    if not backend.startswith("lora"):
        return
    if not cfg.enable_lora:
        raise ValueError(
            f"--moe-runner-backend {backend} requires --enable-lora (or --lora-paths)"
        )
    if cfg.lora_backend != "triton_v2":
        raise ValueError(
            f"--moe-runner-backend {backend} requires --lora-backend triton_v2"
        )
    _check_v2_lora_modes(cfg, f"--moe-runner-backend {backend}")
    # Layer attachment checks the specific weight layout within each scheme.
    if cfg.quantization not in (None, "fp8", "modelopt_fp4"):
        raise ValueError(
            f"--moe-runner-backend {backend} supports unquantized BF16, 128-block "
            f"FP8 and ModelOpt NVFP4 MoE, got --quantization {cfg.quantization}"
        )
    if cfg.quantization == "modelopt_fp4" and backend != "lora_marlin":
        raise ValueError(
            f"--moe-runner-backend {backend} cannot run ModelOpt NVFP4 MoE; "
            "use --moe-runner-backend lora_marlin"
        )
    if cfg.moe_a2a_backend != "none":
        raise ValueError(
            f"--moe-runner-backend {backend} requires Standard dispatch, "
            f"got --moe-a2a-backend {cfg.moe_a2a_backend}"
        )
    if cfg.ep_join_mode is not None:
        raise ValueError(
            f"--moe-runner-backend {backend} does not yet support elastic EP"
        )
    if (
        cfg.enable_eplb
        or cfg.init_expert_location != "trivial"
        or cfg.ep_num_redundant_experts > 0
    ):
        raise ValueError(
            f"--moe-runner-backend {backend} currently requires trivial expert placement "
            "without EPLB or redundant experts"
        )


def check_lora_speculative_compatibility(server_args: Any):
    """Validate LoRA + speculative decoding combinations.

    Adapters apply to the target only; a shared draft runs unadapted.
    Matches resolved algorithm names (NEXTN has collapsed to EAGLE).
    """
    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm in ["NGRAM", None]:
        return

    # These algorithms present a uniform per-request token width during
    # verify, which is what the LoRA segment layout assumes.
    lora_spec_algorithms = ("EAGLE", "EAGLE3", "DFLASH", "DSPARK")
    if cfg.speculative_algorithm not in lora_spec_algorithms:
        promoted = (
            " (NEXTN/EAGLE with a Gemma4 assistant draft is automatically "
            "promoted to FROZEN_KV_MTP, which does not support LoRA)"
            if cfg.speculative_algorithm == "FROZEN_KV_MTP"
            else ""
        )
        raise ValueError(
            "LoRA is only compatible with NGRAM, EAGLE, NEXTN, EAGLE3, "
            "DFLASH, or DSPARK speculative decoding, not "
            f"{cfg.speculative_algorithm}{promoted}."
        )

    ragged_mode = envs.SGLANG_RAGGED_VERIFY_MODE.get()

    # Each entry: (is unsupported, why). Reasons are appended to a shared
    # prefix so the message names the combination, not just the flag.
    unsupported = [
        (
            cfg.speculative_algorithm == "DSPARK" and ragged_mode != "static",
            f"does not support SGLANG_RAGGED_VERIFY_MODE={ragged_mode!r}: "
            "the per-request verify lengths it schedules break the "
            "uniform-width LoRA segment layout",
        ),
        (
            cfg.speculative_adaptive,
            "does not support --speculative-adaptive: the draft is built "
            "from a static ServerArgs snapshot, and the runtime-state "
            "swap does not rebuild LoRA cuda-graph metadata",
        ),
        (
            "experimental_sgl_trtllm"
            in (cfg.moe_runner_backend, cfg.speculative_moe_runner_backend),
            "does not support the experimental_sgl_trtllm MoE runner: its "
            "TopK reads the LoRA config per forward, which the draft "
            "resolves against the target's after its own publish ended",
        ),
        (
            envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get(),
            "does not support SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1: LoRA "
            "batch preparation would run on the plan stream, unordered "
            "against in-flight forwards",
        ),
    ]
    for is_unsupported, reason in unsupported:
        if is_unsupported:
            raise ValueError(
                f"LoRA with EAGLE/NEXTN/EAGLE3 speculative decoding {reason}."
            )
