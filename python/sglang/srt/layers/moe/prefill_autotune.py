"""FlashInfer autotune of the Qwen NVFP4 TRT-LLM MoE at prefill token counts.

The startup autotune runs a decode-shaped dummy forward, so FlashInfer profiles
``trtllm_fp4_block_scale_moe`` only up to the decode token count; every larger
prefill MoE call misses the tactic cache and runs the default tactic. Like the
GLM and DeepSeek hooks, this makes one routed-experts call of the first eligible
``Qwen2MoeSparseMoeBlock`` at the prefill ceiling inside the runner's autotune
context. The call takes the block's own routed path with the pre-quant and fused
finalize choices a captured prefill segment makes at that size, so it stores the
cache keys served prefill calls look up. The router GEMM and the shared expert are
not called, so no other op is tuned at prefill sizes.

Only the choice among FlashInfer's precompiled tactics changes. A tuning file
(``SGLANG_FLASHINFER_TACTIC_PIN_FILE``, see ``flashinfer_tactic_pins``) can pin
the tactic per prefill bucket and exclude split-K tactics. Each boot with the hook
on runs one extra tuning pass over the prefill buckets, unless the autotune cache
already holds them.
"""

from __future__ import annotations

import functools
import logging
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.flashinfer_tactic_pins import (
    DEFAULT_TACTIC,
    MOE_OP,
    active_moe_tactic_pins,
    settle_prefill_tactics,
    valid_tactics_by_bucket,
)

logger = logging.getLogger(__name__)

# The dual-stream MoE forward's router-front paths serve at most 16 tokens; the
# hook mirrors its larger-M branch.
_MIN_TOKENS = 17


@functools.lru_cache(maxsize=None)
def _log_inert_once(reason: str) -> None:
    logger.debug("Qwen MoE prefill autotune inactive: %s", reason)


@functools.lru_cache(maxsize=None)
def _warn_table_unmatched_once(shape) -> None:
    logger.warning(
        "The tuning file's prefill MoE table (shape %s) matched no tuned entry; "
        "the tuner's picks are kept",
        shape,
    )


def _eligible_block(model):
    """First NVFP4 TRT-LLM MoE block on the TP1/EP1 glue path, or None."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    for module in model.modules():
        if isinstance(module, Qwen2MoeSparseMoeBlock) and module._uses_nvfp4_trtllm_glue():
            return module
    return None


def wants_qwen_moe_prefill_autotune(model) -> bool:
    if not envs.SGLANG_ENABLE_QWEN_MOE_PREFILL_AUTOTUNE.get():
        return False
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

    if not get_moe_runner_backend().is_flashinfer_trtllm():
        _log_inert_once("the MoE runner backend is not flashinfer_trtllm")
        return False
    if not get_moe_a2a_backend().is_none() or is_dp_attention_enabled():
        _log_inert_once("needs no MoE all-to-all and no DP attention")
        return False
    block = _eligible_block(model)
    if block is None:
        _log_inert_once("no NVFP4 TRT-LLM MoE block on the TP1/EP1 path")
        return False
    if getattr(block.experts, "_nvfp4_cute_prefill", None) is not None:
        # No trtllm-gen MoE runs on this worker, so its tactic table (pins included) does not apply.
        logger.info(
            "Qwen MoE prefill autotune skipped: the routed experts run the CuTe-DSL path "
            "(SGLANG_QWEN_MOE_PREFILL_CUTE_RUBIN); the MoE tactic table is not used"
        )
        return False
    return True


def _route_one_call(block, num_tokens: int, dtype: torch.dtype) -> tuple[bool, bool]:
    """The routed-experts call of the TP1 dual-stream branch; returns its
    (fused_finalize, prequant) choices."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_nvfp4_routed_input,
    )
    from sglang.srt.models import qwen2_moe

    nvfp4_glue = block.tp_size == 1 and block._uses_nvfp4_trtllm_glue()
    use_fused_gate = (
        block.shared_expert_gate is not None
        and not qwen2_moe.use_intel_amx_backend(block.shared_expert_gate)
        and not qwen2_moe.is_npu()
    )
    prequant = nvfp4_glue and qwen2_moe._NVFP4_PREQUANT_OVERLAP
    fused_finalize = (
        nvfp4_glue
        and qwen2_moe._FUSED_FINALIZE_GATE
        and use_fused_gate
        and block.shared_expert is not None
        and num_tokens <= qwen2_moe._FUSED_FINALIZE_GATE_MAX_TOKENS
    )
    device = block.gate.weight.device
    with torch.no_grad():
        hidden_states = torch.randn(
            (num_tokens, block.gate.weight.shape[1]), dtype=dtype, device=device
        )
        # The router output's shape and dtype; the tuner profiles on its own
        # synthesized logits, and skipping the GEMM keeps it untuned at these sizes.
        router_logits = torch.randn(
            (num_tokens, block.gate.weight.shape[0]),
            dtype=block.gate.weight.dtype,
            device=device,
        )
        pre_quant = (
            quantize_nvfp4_routed_input(hidden_states, block.experts.w13_input_scale_quant)
            if prequant
            else None
        )
        block._forward_router_experts(
            hidden_states,
            precomputed_router=(router_logits, pre_quant),
            fused_finalize=fused_finalize,
        )
        torch.cuda.current_stream().synchronize()
    return fused_finalize, prequant


def _moe_entries(winners: dict) -> set:
    return {key for key in winners if key.custom_op == MOE_OP}


def autotune_qwen_moe_prefill(model, num_tokens: int, *, dtype: torch.dtype) -> int:
    from flashinfer.autotuner import AutoTuner

    block = _eligible_block(model)
    if block is None or num_tokens < _MIN_TOKENS:
        _log_inert_once(f"no eligible block, or {num_tokens} tokens < {_MIN_TOKENS}")
        return 0
    # Read before any GPU work: a malformed tuning file fails the boot.
    pins = active_moe_tactic_pins()

    tuner = AutoTuner.get()
    before = _moe_entries(tuner._winner_cache())
    started = time.perf_counter()
    oom = False
    fused_finalize = prequant = None
    try:
        fused_finalize, prequant = _route_one_call(block, num_tokens, dtype)
    except torch.OutOfMemoryError:
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            _autotune_tactic_sync_group,
        )
        from sglang.srt.runtime_context import get_parallel

        if _autotune_tactic_sync_group(get_parallel().tp_group) is not None:
            raise  # Tuning is collective: a rank that stops alone hangs its peers.
        oom = True
    finally:
        torch.cuda.empty_cache()  # Free the dummy rows before graph capture sizes memory.

    with tuner._lock:
        winners = tuner._winner_cache()
        new_keys = _moe_entries(winners) - before
        valid = {}
        if pins is not None and pins.prefill is not None:
            from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner

            valid = valid_tactics_by_bucket(MoERunner.valid_tactics_dict, pins.shape)
        rows = settle_prefill_tactics(winners, new_keys, pins=pins, valid=valid)
    for bucket, final, tuned, source in rows:
        logger.debug(
            "Qwen MoE prefill autotune: bucket=%d tactic=%s tuned=%s source=%s",
            bucket,
            final,
            tuned,
            source,
        )
    from_table = sum(1 for row in rows if row[3].startswith("pin"))
    if pins is not None and pins.prefill is not None and rows and not from_table:
        _warn_table_unmatched_once(pins.shape)
    logger.info(
        "Qwen MoE prefill autotune: %d tokens, %d buckets tuned this boot, %d set "
        "from the tuning file, %d on the default tactic; fused_finalize=%s, "
        "prequant=%s, %.1f s%s",
        num_tokens,
        len(rows),
        from_table,
        sum(1 for row in rows if row[1] == DEFAULT_TACTIC),
        fused_finalize,
        prequant,
        time.perf_counter() - started,
        ", skipped after OOM" if oom else "",
    )
    if oom:
        return 0
    if envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return 0  # An explicitly requested whole-model EXTEND pass still runs.
    return max(1, len(rows))
