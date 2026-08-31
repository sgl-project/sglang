# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the LoRA adapters."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_late_resolution,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef

logger = logging.getLogger(__name__)


def check_lora_server_args(server_args: Any):
    cfg = resolving_view(server_args)

    assert cfg.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

    # Enable LoRA if any LoRA paths are provided for backward compatibility.
    if cfg.lora_paths:
        if cfg.enable_lora is None:
            declare_late_resolution(
                server_args, "check_lora_server_args", enable_lora=True
            )
            logger.warning(
                "--enable-lora is set to True because --lora-paths is provided."
            )
        elif cfg.enable_lora is False:
            logger.warning(
                "--enable-lora is set to False, any provided lora_paths will be ignored."
            )

    if cfg.enable_lora:
        if cfg.enable_lora_overlap_loading is None:
            declare_late_resolution(
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

        # Parse lora_paths
        if isinstance(cfg.lora_paths, list):
            parsed_lora_paths = []
            for lora_path in cfg.lora_paths:
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
                    assert (
                        "lora_name" in lora_path and "lora_path" in lora_path
                    ), f"When providing LoRA paths as a list of dict, each dict should contain 'lora_name' and 'lora_path' keys. Got: {lora_path}"
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
            declare_late_resolution(
                server_args, "check_lora_server_args", lora_paths=parsed_lora_paths
            )
        elif isinstance(cfg.lora_paths, dict):
            declare_late_resolution(
                server_args,
                "check_lora_server_args",
                lora_paths=[
                    LoRARef(
                        lora_id=LoRARef.deterministic_id(k, v),
                        lora_name=k,
                        lora_path=v,
                        pinned=False,
                    )
                    for k, v in cfg.lora_paths.items()
                ],
            )
        elif cfg.lora_paths is None:
            declare_late_resolution(
                server_args, "check_lora_server_args", lora_paths=[]
            )
        else:
            raise ValueError(
                f"Invalid type for --lora-paths: {type(cfg.lora_paths)}. "
                "Expected a list or a dictionary."
            )

        # Normalize target modules to a set; keep {"all"} as a sentinel
        # that gets resolved model-awarely in lora_manager.init_lora_shapes().
        if cfg.lora_target_modules:
            declare_late_resolution(
                server_args,
                "check_lora_server_args",
                lora_target_modules=set(cfg.lora_target_modules),
            )
            if "all" in cfg.lora_target_modules:
                assert (
                    len(cfg.lora_target_modules) == 1
                ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."

        # Ensure sufficient information is provided for LoRA initialization.
        assert cfg.lora_paths or (
            cfg.max_lora_rank and cfg.lora_target_modules
        ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

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

        assert (
            cfg.lora_drain_wait_threshold >= 0.0
        ), "--lora-drain-wait-threshold must be non-negative."


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
