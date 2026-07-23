"""Local env registry for the experimental TRT-LLM LoRA fast path.

Every flag here is gated by the single global master switch
``SGLANG_EXPERIMENTAL_LORA_OPTI`` (defined in ``sglang.srt.environ``). When the master
switch is OFF (the default), every flag reads ``False`` (its default is
suppressed), so the no-LoRA path, other MoE backends, and the default
(non-experimental) LoRA path are byte-identical to upstream.

Keeping these flags out of the global ``Envs`` class is deliberate: the only
sglang-global addition for this feature is ``SGLANG_EXPERIMENTAL_LORA_OPTI``; all the
fine-grained opt switches live here, next to the code that consumes them.

Default policy (applies only when ``SGLANG_EXPERIMENTAL_LORA_OPTI=1``):
  * **common** flags — used by BOTH the qwen3.5 (FP8) and kimi (NVFP4) configs —
    default ``True`` so they need not be repeated on every launch command.
  * **non-shared** flags default ``False`` and must be set explicitly in the
    launch environment for the model that needs them.

C++-getenv flags (read via ``getenv`` in the JIT launcher, not in Python) are
listed at the bottom for documentation only; set them in the launch env.
"""

import os

from sglang.srt.environ import envs


def experimental_lora_enabled() -> bool:
    """Master gate. All flags below are forced off unless this is set."""
    return envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()


_TRUE = {"1", "true", "yes", "on"}


class _GatedBool:
    def __init__(self, name: str, default: bool):
        self._name = name
        self._default = default

    def get(self) -> bool:
        if not experimental_lora_enabled():
            return False
        raw = os.environ.get(self._name)
        if raw is None:
            return self._default
        return raw.strip().lower() in _TRUE


class _GatedInt:
    def __init__(self, name: str, default: int):
        self._name = name
        self._default = default

    def get(self) -> int:
        # Only consulted on the experimental path; return the default otherwise.
        if not experimental_lora_enabled():
            return self._default
        raw = os.environ.get(self._name)
        return int(raw) if raw is not None else self._default


class _LoraEnvs:
    # ---- common (qwen3.5 ∩ kimi): default True when experimental is on ----
    SGLANG_ENABLE_LORA_SHRINK_SPLIT_K = _GatedBool(
        "SGLANG_ENABLE_LORA_SHRINK_SPLIT_K", True
    )
    SGLANG_OPT_LORA_FUSED_MERGED_ALIGN = _GatedBool(
        "SGLANG_OPT_LORA_FUSED_MERGED_ALIGN", True
    )
    SGLANG_OPT_LORA_FUSED_TOPK_PACK = _GatedBool(
        "SGLANG_OPT_LORA_FUSED_TOPK_PACK", True
    )
    SGLANG_OPT_LORA_QKV_B_STORE = _GatedBool("SGLANG_OPT_LORA_QKV_B_STORE", True)
    # F1-①: prefill routing reuse — unify the A (shrink) stage's routing BLOCK_SIZE_M with
    # the B stage's at prefill (>=512 tokens) so the per-layer routing_cache key matches
    # across stages and the Triton align/sort runs once per layer-forward instead of once
    # per stage (4x at prefill). Dtype-agnostic (the chain is shared by fp8/nvfp4/bf16).
    # Decode (<512) keeps the opt1 fused merged-align path and its tuned shrink block.
    SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE = _GatedBool(
        "SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE", True
    )

    # ---- correctness fixes: on by default when experimental ----
    # gate_up gated-split fix (up_A shrink for the up half); set =0 only to A/B bisect.
    SGLANG_ENABLE_LORA_MOE_GATEUP_GATED_SPLIT = _GatedBool(
        "SGLANG_ENABLE_LORA_MOE_GATEUP_GATED_SPLIT", True
    )
    # feed bf16 router logits straight to the JIT kimi gate (bitwise-identical).
    SGLANG_OPT_KIMI_GATE_BF16_INPUT = _GatedBool(
        "SGLANG_OPT_KIMI_GATE_BF16_INPUT", True
    )

    # ---- non-shared: default False, set explicitly in the launch env ----
    # kimi (NVFP4):
    SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE = _GatedBool(
        "SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE", False
    )
    SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN = _GatedBool(
        "SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN", False
    )
    # qwen3.5 (FP8):
    SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC = _GatedBool(
        "SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC", False
    )
    # (SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP removed: net-neutral + base/decode-corruption hazard; serial down-LoRA only.)
    SGLANG_OPT_LORA_SHARED_ADD_OVERLAP = _GatedBool(
        "SGLANG_OPT_LORA_SHARED_ADD_OVERLAP", False
    )
    SGLANG_OPT_LORA_CUBLAS = _GatedBool("SGLANG_OPT_LORA_CUBLAS", False)
    SGLANG_OPT_LORA_CUBLAS_A = _GatedBool("SGLANG_OPT_LORA_CUBLAS_A", False)
    SGLANG_OPT_LORA_CUBLAS_B = _GatedBool("SGLANG_OPT_LORA_CUBLAS_B", False)
    SGLANG_OPT_LORA_CUBLAS_GATE_UP = _GatedBool("SGLANG_OPT_LORA_CUBLAS_GATE_UP", False)
    SGLANG_OPT_LORA_CUBLAS_QKV = _GatedBool("SGLANG_OPT_LORA_CUBLAS_QKV", False)
    SGLANG_OPT_LORA_CUBLAS_KV_B = _GatedBool("SGLANG_OPT_LORA_CUBLAS_KV_B", False)
    # diagnostics / tuning:
    SGLANG_OPT_LORA_SHRINK_TUNE = _GatedBool("SGLANG_OPT_LORA_SHRINK_TUNE", False)

    # kimi NVFP4 permute+quant fuse — read in jit_kernel/trtllm_lora_temp/core.py (Python) to pass
    # a bool to the kernel, AND C++-side via getenv in the launcher. Default off (kimi-only).
    SGLANG_OPT_FUSED_PERMUTE_QUANT = _GatedBool("SGLANG_OPT_FUSED_PERMUTE_QUANT", False)

    # ---- integer knob ----
    # decode two-stream token ceiling (consulted only on the experimental path).
    SGLANG_TWO_STREAM_MAX_TOKENS = _GatedInt("SGLANG_TWO_STREAM_MAX_TOKENS", 256)


lora_envs = _LoraEnvs()

# ---------------------------------------------------------------------------
# C++-getenv-only flags (read via getenv in jit_kernel .cu launchers, NOT in Python).
# Set them in the launch env on the model that needs them; default off:
#   SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE (kimi NVFP4 act+down-quant fuse)
#   SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC      (kimi NVFP4 vectorized activation)
# ---------------------------------------------------------------------------
