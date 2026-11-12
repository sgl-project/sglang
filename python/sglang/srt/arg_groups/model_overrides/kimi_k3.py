"""Config-time override declarations for kimi_k3."""

import inspect
import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _dspark_verify_on_decode_backend,
    _is_mxfp4_pack_quantized,
    _register_for,
    attention_backends_of,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def _require_kimi_k3_cutedsl_dcp_support() -> None:
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        parameters = inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='cutedsl_mla' requires "
            "FlashInfer 0.6.17 or newer with "
            "trtllm_batch_decode_with_kv_cache_mla exposing enable_dcp."
        ) from exc

    if "enable_dcp" not in parameters:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='cutedsl_mla' requires "
            "enable_dcp in the signature of "
            "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla; upgrade "
            "to FlashInfer 0.6.17 or newer."
        )


def _require_kimi_k3_aiter_gluon_dcp_support() -> None:
    from sglang.srt.layers.attention.aiter_mla_gluon import _gluon_fn

    if _gluon_fn() is None:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='aiter' requires the aiter "
            "gluon mla kernel, which is unavailable. See above aborting reasons."
        )


@_register_for("KimiK3ForConditionalGeneration")
def _kimi_k3_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.enable_prefill_cp:
        raise ValueError(
            "Kimi-K3 does not support --enable-prefill-cp yet (Kimi-K3.1 does)."
        )
    if cfg.dcp_size > 1:
        overrides = {}
        if cfg.enable_symm_mem:
            logger.warning(
                "Kimi-K3 DCP disables --enable-symm-mem due to decode CUDA "
                "graph correctness issues."
            )
            overrides["enable_symm_mem"] = False

        if cfg.speculative_algorithm == "DSPARK":
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            ragged_mode = read_ragged_verify_mode()
            if ragged_mode is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "Kimi-K3 DCP + DSPARK currently requires "
                    "SGLANG_RAGGED_VERIFY_MODE=static; compact/cap-accept are "
                    f"not validated under DCP (got {ragged_mode.value!r})."
                )

            # DSPARK target-verify + draft-extend must run on the decode
            # (cutedsl_mla) backend, whose _run_decode_kernel implements the DCP
            # signature (causal_seqs / cp_world / cp_rank). The default
            # "prefill" routes verify to trtllm_mla, whose base _run_decode_kernel
            # lacks that DCP path (TypeError: unexpected kwarg 'causal_seqs').
            overrides["speculative_attention_mode"] = "decode"

        if get_platform().is_npu:
            overrides.update(
                dcp_comm_backend="a2a",
                dcp_replicate_q_proj=False,
            )
            return overrides

        prefill_backend, decode_backend = attention_backends_of(cfg)
        if decode_backend == "cutedsl_mla" or decode_backend is None:
            _require_kimi_k3_cutedsl_dcp_support()
            logger.info(
                "Kimi-K3 DCP keeps decode attention backend 'cutedsl_mla' "
                f"(prefill={prefill_backend!r} -> 'trtllm_mla')."
            )
            overrides.update(
                prefill_attention_backend="trtllm_mla",
                decode_attention_backend="cutedsl_mla",
            )
        elif decode_backend == "aiter":
            _require_kimi_k3_aiter_gluon_dcp_support()
            # Override prefill backend to aiter by default
            # if users don't explicitly specify triton
            prefill_ab = "triton" if prefill_backend == "triton" else "aiter"
            logger.info(
                "Kimi-K3 DCP uses aiter MLA decode: "
                f"prefill={prefill_backend!r} -> {prefill_ab!r}, "
                f"decode={decode_backend!r} -> 'aiter'."
            )
            overrides.update(
                prefill_attention_backend=prefill_ab,
                decode_attention_backend="aiter",
            )
        else:
            raise AssertionError(
                f"Decode attention backend for Kimi-K3 DCP must be 'cutedsl_mla' or 'aiter', got {decode_backend!r}."
            )

        if cfg.dcp_replicate_q_proj is None and cfg.dcp_comm_backend in (
            "a2a",
            "fi_a2a",
        ):
            logger.info("Kimi-K3 DCP enables replicated Q projection by default.")
            overrides["dcp_replicate_q_proj"] = True
        return overrides

    if not (get_platform().is_sm100 and get_platform().device_sm in (100, 103)):
        return {}
    backends_unset = is_attention_backend_not_set(cfg)
    if cfg.speculative_algorithm != "DSPARK":
        if not backends_unset:
            return {}
        logger.info(
            "Use trtllm_mla as the default prefill and decode attention "
            "backend for Kimi-K3 on SM100/SM103."
        )
        return {
            "decode_attention_backend": "trtllm_mla",
            "prefill_attention_backend": "trtllm_mla",
        }
    # DSPARK: verify runs on the decode backend (mode=decode below), so this
    # picks the verify kernel -- mode=prefill routes it to flashinfer, which is
    # slow and syncs, while plain decode is cold under dspark.
    q_len = cfg.speculative_num_draft_tokens or (
        cfg.speculative_dspark_block_size + 1
        if cfg.speculative_dspark_block_size is not None
        # Checkpoint auto-infer happens after overrides; K3 draft uses block 7.
        else 8
    )
    overrides = {}
    if backends_unset:
        backend = "trtllm_mla"
        overrides["decode_attention_backend"] = backend
        overrides["prefill_attention_backend"] = "trtllm_mla"
    else:
        # Explicit backend knobs keep priority, but the mode is a separate knob
        # that still needs declaring -- else verify stays on the prefill backend,
        # whose host-side plan (flashinfer by default) forces a per-step D2H.
        _, backend = attention_backends_of(cfg)
    if _dspark_verify_on_decode_backend(backend, q_len):
        overrides["speculative_attention_mode"] = "decode"
        logger.info(
            "Kimi-K3 DSPARK on SM100/SM103: decode/verify attention backend "
            f"{backend} (speculative_attention_mode=decode)."
        )
    else:
        logger.warning(
            f"Kimi-K3 DSPARK: decode attention backend {backend!r} cannot serve "
            f"target verify at q_len={q_len}, so verify runs on the prefill "
            "backend (speculative_attention_mode=prefill). A host-plan prefill "
            "backend costs a per-step seq_lens D2H sync; leave the attention "
            "backend knobs unset for the sync-free default."
        )
    return overrides


@_register_for("KimiK31ForConditionalGeneration")
def _kimi_k31_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.enable_unified_memory:
        raise ValueError("Kimi-K3.1 does not support --enable-unified-memory yet.")
    overrides = {}
    if cfg.dcp_size > 1:
        if cfg.speculative_algorithm not in (None, "DSPARK", "DFLASH"):
            raise ValueError(
                "Kimi-K3.1 DCP supports only DSPARK and DFLASH speculative decoding."
            )
        if (
            cfg.disaggregation_mode != "null"
            and cfg.disaggregation_transfer_backend != "mooncake"
        ):
            raise ValueError("Kimi-K3.1 DCP supports PD only over Mooncake.")
        if cfg.speculative_algorithm == "DSPARK":
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "Kimi-K3.1 DCP + DSPARK requires SGLANG_RAGGED_VERIFY_MODE=static."
                )
        if cfg.speculative_algorithm == "DFLASH":
            if cfg.speculative_draft_attention_backend not in (None, "cutedsl_mla"):
                raise ValueError(
                    "Kimi-K3.1 DCP + DFLASH needs "
                    "--speculative-draft-attention-backend cutedsl_mla."
                )
            overrides["speculative_draft_attention_backend"] = "cutedsl_mla"
        if cfg.speculative_algorithm is not None:
            # NOTE(yhyang201): DCP target verify runs the DSA decode path; each
            # rank attends over its KV shard and the (o, lse) partials are merged.
            overrides["speculative_attention_mode"] = "decode"
        if (
            cfg.dsa_topk_backend != "sgl-kernel"
            or not envs.SGLANG_OPT_USE_TOPK_V2.get()
            or not envs.SGLANG_DSA_FUSE_TOPK.get()
        ):
            raise ValueError(
                "Kimi-K3.1 DCP needs the fused sgl-kernel v2 DSA top-k "
                "(--dsa-topk-backend sgl-kernel, SGLANG_OPT_USE_TOPK_V2 and "
                "SGLANG_DSA_FUSE_TOPK on)."
            )
        if cfg.enable_return_indexer_topk or envs.SGLANG_DSA_TOPK_BROADCAST.get():
            # the sharded indexer returns only this rank's local KV slots
            raise ValueError(
                "Kimi-K3.1 DCP does not support --enable-return-indexer-topk "
                "or SGLANG_DSA_TOPK_BROADCAST."
            )
        if cfg.dcp_replicate_q_proj is None and cfg.dcp_comm_backend in (
            "a2a",
            "fi_a2a",
        ):
            logger.info("Kimi-K3.1 DCP enables replicated Q projection by default.")
            overrides["dcp_replicate_q_proj"] = True
    return overrides


@_register_for("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration")
def _kimi_k3_moe_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    # MoE runner default, independent of the attention-backend gate above.
    # trtllm-gen fused MoE (flashinfer_mxfp4) beats marlin on both the decode
    # (M=bs) and the target-verify (M=bs*(gamma+1)) regimes on SM100/SM103.
    # SM107 uses the same packed-MXFP4 runner; leaving auto unresolved falls
    # back to BF16 weight materialization during model loading.
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "auto":
        return {}
    if not (get_platform().is_sm100 and get_platform().device_sm in (100, 103, 107)):
        return {}
    if not _is_mxfp4_pack_quantized(hf_config):
        return {}
    logger.info(
        "Kimi-K3 on SM100/SM103/SM107: moe_runner_backend=flashinfer_mxfp4 "
        "(FlashInfer SiTU kernels)."
    )
    return {"moe_runner_backend": "flashinfer_mxfp4"}
