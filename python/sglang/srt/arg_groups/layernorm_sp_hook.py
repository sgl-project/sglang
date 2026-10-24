from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import model_config_of, resolving_view
from sglang.srt.runtime_context import attn_dp_enabled_of

logger = logging.getLogger(__name__)


def handle_layernorm_sp(server_args: ServerArgs) -> None:
    """Validate --enable-layernorm-sp against the resolved parallelism config.

    Runs in the resolution pipeline rather than in the layers so a model that
    does not use layer boundaries rejects the flag instead of ignoring it.
    """
    cfg = resolving_view(server_args)
    use_megamoe = getattr(cfg, "enable_megamoe_for_layernorm_sp", False)
    if not cfg.enable_layernorm_sp:
        if use_megamoe:
            raise ValueError(
                "--enable-megamoe-for-layernorm-sp requires --enable-layernorm-sp."
            )
        return
    architectures = model_config_of(server_args).hf_config.architectures
    validate_layernorm_sp(
        architecture=architectures[0] if architectures else None,
        tp_size=cfg.tp_size,
        ep_size=cfg.ep_size,
        quantization=cfg.quantization,
        moe_a2a_backend=cfg.moe_a2a_backend,
        moe_runner_backend=cfg.moe_runner_backend,
        attn_dp_enabled=attn_dp_enabled_of(cfg),
        speculative_algorithm=cfg.speculative_algorithm,
        pp_size=cfg.pp_size,
        enable_megamoe_for_layernorm_sp=use_megamoe,
    )
    if architectures and architectures[0] in ("MuseSparkForConditionalGeneration",):
        from sglang.srt.environ import envs

        if use_megamoe:
            if cfg.enable_mixed_chunk:
                raise ValueError(
                    "--enable-megamoe-for-layernorm-sp is incompatible with "
                    "--enable-mixed-chunk: mixed batches bypass SP prefill."
                )
            from sglang.srt.arg_groups.overrides import declare_resolution
            from sglang.srt.model_executor.cuda_graph_config import (
                Backend,
                Phase,
                with_phase,
            )

            validate_muse_megamoe_token_budget(
                tp_size=cfg.tp_size,
                chunked_prefill_size=cfg.chunked_prefill_size,
                max_prefill_tokens=cfg.max_prefill_tokens,
                capacity=envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get(),
            )
            if cfg.pp_size != 1:
                raise ValueError(
                    "Muse MegaMoE LayerNorm SP requires pipeline parallel size 1."
                )
            # Breakable capture warms every bucket before capturing, so SP
            # workspaces, MegaMoE symmetric buffers and JIT kernels are ready.
            # Preserve the CUDA default as well as explicit breakable selection.
            # Other backends still use eager prefill; decode is unchanged.
            prefill_backend = Backend.DISABLED
            if cfg.cuda_graph_config.prefill.backend == Backend.BREAKABLE:
                if cfg.chunked_prefill_size <= 0:
                    raise ValueError(
                        "Muse MegaMoE breakable prefill requires a positive "
                        "--chunked-prefill-size to bound SP workspace growth."
                    )
                capture_bs = cfg.cuda_graph_config.prefill.bs
                if not capture_bs or max(capture_bs) < cfg.chunked_prefill_size:
                    raise ValueError(
                        "Muse MegaMoE breakable prefill requires the largest "
                        "prefill capture bucket to cover --chunked-prefill-size; "
                        "a larger eager prefill could replace captured SP workspaces."
                    )
                prefill_backend = Backend.BREAKABLE
            declare_resolution(
                server_args,
                "handle_layernorm_sp",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=prefill_backend
                ),
            )
            logger.info(
                "Muse LayerNorm SP uses MegaMoE for prefill (CUDA graph: %s) "
                "and TRT-LLM with all-reduce for decode; sharing expert weights.",
                prefill_backend,
            )
            return

        if envs.SGLANG_MOE_NVFP4_DISPATCH.get():
            raise ValueError(
                "Muse Spark --enable-layernorm-sp requires "
                "SGLANG_MOE_NVFP4_DISPATCH=0 so FlashInfer A2A transports "
                "BF16 activations and sizes its workspace for the BF16 payload."
            )


def validate_layernorm_sp(
    *,
    architecture: Optional[str],
    tp_size: int,
    attn_dp_enabled: bool,
    speculative_algorithm: Optional[str],
    pp_size: int = 1,
    ep_size: int = 1,
    quantization: Optional[str] = None,
    moe_a2a_backend: str = "none",
    moe_runner_backend: str = "auto",
    enable_megamoe_for_layernorm_sp: bool = False,
) -> None:
    """Fail loud for unsupported / incompatible configs. Callers gate on the flag."""
    from sglang.srt.layers.layernorm_sp import (
        SP_SUPPORTED_ARCHITECTURES,
        is_muse_spark_architecture,
    )

    if enable_megamoe_for_layernorm_sp and not is_muse_spark_architecture(architecture):
        raise ValueError("--enable-megamoe-for-layernorm-sp supports only Muse Spark.")
    if architecture not in SP_SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "--enable-layernorm-sp is only supported for "
            f"{sorted(SP_SUPPORTED_ARCHITECTURES)}; got {architecture}."
        )
    if tp_size <= 1:
        raise ValueError(
            "--enable-layernorm-sp requires tp_size > 1: there is no sequence to "
            "shard across a single TP rank."
        )
    if is_muse_spark_architecture(architecture):
        if (tp_size, ep_size) not in ((4, 4), (8, 8)):
            raise ValueError(
                "Muse Spark --enable-layernorm-sp requires TP4/EP4 or TP8/EP8; "
                f"got TP{tp_size}/EP{ep_size}."
            )
        # MegaMoE shares TRT-LLM expert weights for both BF16 and NVFP4.
        # The standalone FlashInfer A2A path remains restricted to NVFP4.
        if quantization != "modelopt_fp4" and not (
            quantization is None and enable_megamoe_for_layernorm_sp
        ):
            raise ValueError(
                "Muse Spark --enable-layernorm-sp requires serialized NVFP4 "
                "weights with --quantization modelopt_fp4, or BF16 weights with "
                "--enable-megamoe-for-layernorm-sp."
            )
        if enable_megamoe_for_layernorm_sp:
            if moe_a2a_backend not in ("none", "flashinfer"):
                raise ValueError(
                    "Muse MegaMoE prefill requires --moe-a2a-backend none or flashinfer; "
                    "MegaMoE performs its own dispatch/combine and decode uses all-reduce."
                )
        elif moe_a2a_backend != "flashinfer":
            raise ValueError(
                "Muse Spark --enable-layernorm-sp requires "
                "--moe-a2a-backend flashinfer."
            )
        if moe_runner_backend != "flashinfer_trtllm":
            raise ValueError(
                "Muse Spark --enable-layernorm-sp requires "
                "--moe-runner-backend flashinfer_trtllm."
            )
    if attn_dp_enabled:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with attention DP: "
            "SP shards the sequence across the full TP group, which under DP "
            "attention spans data-parallel groups holding different sequences."
        )
    if speculative_algorithm is not None and not (
        is_muse_spark_architecture(architecture) and speculative_algorithm == "DFLASH"
    ):
        raise ValueError(
            "--enable-layernorm-sp supports speculative decoding only with "
            "Muse Spark DFLASH: other models/algorithms do not gather the "
            "sequence-sharded auxiliary hidden states."
        )
    if pp_size > 1:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with pipeline parallelism: "
            "only the first rank's stack starts the sequence-sharded region, so "
            "the next rank would receive a stream it does not expect."
        )


def validate_muse_megamoe_token_budget(
    *, tp_size: int, chunked_prefill_size: int, max_prefill_tokens: int, capacity: int
) -> None:
    """Size the shared MegaMoE buffer for the largest SP prefill shard."""
    budget = chunked_prefill_size if chunked_prefill_size > 0 else max_prefill_tokens
    if capacity <= 0 or budget <= 0:
        raise ValueError(
            "Muse MegaMoE requires positive prefill and per-rank token budgets."
        )
    per_rank = (budget + tp_size - 1) // tp_size
    if per_rank > capacity:
        raise ValueError(
            f"Muse MegaMoE needs {per_rank} tokens per rank, exceeding capacity {capacity}; "
            "increase SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK "
            "or reduce --chunked-prefill-size."
        )
