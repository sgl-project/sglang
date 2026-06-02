"""Draft-model inventory: single source of truth for routed-experts capture
opt-out.

A draft model whose MoE layers can write into the target's R3 capture buffer
must explicitly opt out via `TopKConfig.allow_routed_experts_capture=False`
during construction. A draft model that contains no `TopK` (`dense_no_topk`)
needs no opt-out but must be listed here so the runtime guard can recognize
it instead of failing closed.

This module is the single source of truth consumed by:
  - the fail-closed runtime guard in the draft `ModelRunner` init path
  - inventory completeness tests that cross-check against
    `_config_draft_model()` in `python/sglang/srt/configs/model_config.py`
    plus on-disk `*Eagle* / *MTP* / *NextN*` model files

Adding a new MoE-bearing draft architecture without adding an entry here
makes the runtime guard refuse to start, by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DraftInventoryEntry:
    """One draft-architecture row in the inventory.

    Fields:
      - `source_architectures`: target-side architecture(s) that resolve to
        this draft architecture via `_config_draft_model()`. Empty tuple for
        EAGLE-style standalone drafts that are loaded directly via
        `--speculative-draft-model-path`.
      - `draft_architecture`: the resolved class name (matches
        `hf_config.architectures[0]` after `_config_draft_model()` or
        `type(self.model).__name__` for EAGLE drafts).
      - `moe_bearing`: True when the draft model constructs at least one
        `TopK` MoE block; False when the draft is dense (no `TopK`).
      - `draft_signal`: where the per-layer opt-out value comes from at
        construction time. One of `is_nextn` (shared base block keyed on
        `is_nextn`), `is_mtp` (similar but via an MTP wrapper), `always_draft`
        (the class itself is always a draft, so opt-out is hardcoded), or
        `dense_no_topk` (not applicable; dense allowlist).
      - `opt_out_injection_point`: the `file:Class` location where the
        opt-out is set (or is planned to be set, when `opted_out=False`).
        `None` for dense allowlist entries.
      - `rationale`: short evidence string for why this classification is
        correct (cite a code path / signal). Required so future readers can
        re-verify the claim without recomputing.
      - `opted_out`: True when the family's draft `TopK` instances are
        already constructed with `allow_routed_experts_capture=False`. False for
        MoE-bearing entries whose opt-out plumbing is still pending. The
        fail-closed runtime guard treats `opted_out=False` (with
        `moe_bearing=True`) the same as an unknown architecture: it refuses
        to start the draft worker. This makes pending work explicit and
        fail-loud rather than silent. Dense entries set `opted_out=True`
        trivially.
    """

    source_architectures: Tuple[str, ...]
    draft_architecture: str
    moe_bearing: bool
    draft_signal: str
    opt_out_injection_point: Optional[str]
    rationale: str
    opted_out: bool = True


# Order: first the 15 outputs of `_config_draft_model()` (matching the
# source order in `python/sglang/srt/configs/model_config.py:413-487`),
# then the independent EAGLE architectures loaded via
# `--speculative-draft-model-path`.
INVENTORY: Tuple[DraftInventoryEntry, ...] = (
    # ---- _config_draft_model() outputs ----
    DraftInventoryEntry(
        source_architectures=("DeepseekV3ForCausalLM",),
        draft_architecture="DeepseekV3ForCausalLMNextN",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/deepseek_v2.py:DeepseekV2MoE",
        rationale=(
            "deepseek_nextn.py instantiates DeepseekV2DecoderLayer with "
            "is_nextn=True; the shared DeepseekV2MoE passes "
            "allow_routed_experts_capture=not is_nextn to TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("DeepseekV4ForCausalLM",),
        draft_architecture="DeepseekV4ForCausalLMNextN",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/deepseek_v2.py:DeepseekV2MoE",
        rationale=(
            "deepseek_v4_nextn.py constructs DeepseekV4DecoderLayer with "
            "is_nextn=True; DeepseekV4DecoderLayer reuses DeepseekV2MoE which "
            "already keys allow_routed_experts_capture on is_nextn."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("Glm4MoeForCausalLM",),
        draft_architecture="Glm4MoeForCausalLMNextN",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/glm4_moe.py:Glm4MoeSparseMoeBlock",
        rationale=(
            "glm4_moe_nextn.py instantiates Glm4MoeDecoderLayer with "
            "is_nextn=True; the shared Glm4MoeSparseMoeBlock passes "
            "allow_routed_experts_capture=not is_nextn to TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("GlmOcrForConditionalGeneration",),
        draft_architecture="GlmOcrForConditionalGenerationNextN",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale=(
            "glm_ocr_nextn.py uses the dense Glm4DecoderLayer (from glm4.py), "
            "not Glm4MoeDecoderLayer, so the draft constructs zero TopK modules."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("LongcatFlashForCausalLM",),
        draft_architecture="LongcatFlashForCausalLMNextN",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale=(
            "longcat_flash_nextn.py:110 defines LongcatFlashDenseDecoderLayer "
            "whose mlp is LongcatFlashMLP (not the MoE block); the draft has "
            "no MoE TopK to opt out."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("MiMoForCausalLM",),
        draft_architecture="MiMoMTP",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale=(
            "mimo_mtp.py:43 instantiates Qwen2DecoderLayer (dense base "
            "model) for the MTP block; no MoE TopK is constructed."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("MiMoV2ForCausalLM",),
        draft_architecture="MiMoV2MTP",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale=(
            "mimo_v2_nextn.py:57 defines MiMoV2MTPLayer using MiMoV2MLP-style "
            "dense feedforward; no MoE TopK is constructed."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("Step3p5ForCausalLM",),
        draft_architecture="Step3p5MTP",
        moe_bearing=True,
        draft_signal="is_mtp",
        opt_out_injection_point="python/sglang/srt/models/step3p5.py:Step3p5MoEMLP",
        rationale=(
            "step3p5_mtp.py:83 constructs Step3p5DecoderLayer(..., "
            "allow_routed_experts_capture=False); the kwarg threads to Step3p5MoEMLP "
            "and lands as `allow_routed_experts_capture=False` on the MoE TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("BailingMoeForCausalLM",),
        draft_architecture="BailingMoeForCausalLMNextN",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/bailing_moe.py:BailingMoESparseMoeBlock",
        rationale=(
            "bailing_moe_nextn.py:107 passes is_nextn=True to BailingMoEBlock; "
            "the block threads is_nextn to BailingMoESparseMoeBlock which sets "
            "allow_routed_experts_capture=not is_nextn on its TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("Ernie4_5_MoeForCausalLM",),
        draft_architecture="Ernie4_5_MoeForCausalLMMTP",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale=(
            "ernie4_eagle.py passes is_mtp=True; ernie4.py:183 guards "
            "`if (not is_mtp) and ...:` around Ernie4Moe construction so the "
            "MTP path uses dense MLP and constructs no TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("Qwen3NextForCausalLM",),
        draft_architecture="Qwen3NextForCausalLMMTP",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock",
        rationale=(
            "qwen3_next.py reuses Qwen2MoeSparseMoeBlock with is_nextn=True; "
            "the shared block passes allow_routed_experts_capture=not is_nextn to TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("Qwen3_5ForCausalLM",),
        draft_architecture="Qwen3_5ForCausalLMMTP",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock",
        rationale=(
            "qwen3_5.py reuses Qwen2MoeSparseMoeBlock with is_nextn=True via "
            "the same construction path as Qwen3Next."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("ExaoneMoEForCausalLM",),
        draft_architecture="ExaoneMoEForCausalLMMTP",
        moe_bearing=True,
        draft_signal="is_mtp",
        opt_out_injection_point="python/sglang/srt/models/exaone_moe.py:ExaoneMoESparseMoEBlock",
        rationale=(
            "exaone_moe_mtp.py constructs ExaoneMoEModel(..., "
            "allow_routed_experts_capture=False); the kwarg threads through "
            "ExaoneMoEModel / ExaoneMoEDecoderLayer / ExaoneMoESparseMoEBlock "
            "and lands as `allow_routed_experts_capture=False` on the MoE TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("NemotronHForCausalLM",),
        draft_architecture="NemotronHForCausalLMMTP",
        moe_bearing=True,
        draft_signal="is_mtp",
        opt_out_injection_point="python/sglang/srt/models/nemotron_h.py:NemotronHMoE",
        rationale=(
            "nemotron_h_mtp.py:115 NemotronHMTPMoEDecoderLayer calls "
            "super().__init__(..., allow_routed_experts_capture=False); the kwarg "
            "threads through NemotronHMoEDecoderLayer to NemotronHMoE and "
            "lands as `allow_routed_experts_capture=False` on the MoE TopK."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=("HYV3ForCausalLM",),
        draft_architecture="HYV3ForCausalLMNextN",
        moe_bearing=True,
        draft_signal="is_nextn",
        opt_out_injection_point="python/sglang/srt/models/hunyuan_v3.py:HYV3MoEFused",
        rationale=(
            "hunyuan_v3_nextn.py:67 constructs HYV3DecoderLayer(..., "
            "allow_routed_experts_capture=False); the kwarg threads to HYV3MoEFused "
            "and lands as `allow_routed_experts_capture=False` on the MoE TopK."
        ),
    ),
    # ---- Independent EAGLE-style draft architectures ----
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="LlamaForCausalLMEagle",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale="llama_eagle.py extends LlamaForCausalLM which is dense (no MoE).",
    ),
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="LlamaForCausalLMEagle3",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale="llama_eagle3.py uses LlamaMLP at line 78 (dense feedforward).",
    ),
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="MistralForCausalLMEagle",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale="mistral_eagle.py extends LlamaForCausalLMEagle (dense).",
    ),
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="MistralLarge3ForCausalLMEagle",
        moe_bearing=True,
        draft_signal="always_draft",
        opt_out_injection_point="python/sglang/srt/models/mistral_large_3_eagle.py:MistralLarge3EagleModel",
        rationale=(
            "mistral_large_3_eagle.py constructs DeepseekV2DecoderLayer with "
            "is_nextn=False, then marks every constructed TopK with "
            "allow_routed_experts_capture=False at the always-draft wrapper "
            "boundary. DeepseekV2MoE itself keeps the `not is_nextn` contract."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="Qwen2ForCausalLMEagle",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale="qwen2_eagle.py extends Qwen2ForCausalLM (dense, no MoE).",
    ),
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="Eagle3DeepseekV2ForCausalLM",
        moe_bearing=False,
        draft_signal="dense_no_topk",
        opt_out_injection_point=None,
        rationale=(
            "kimi_k25_eagle3.py defines Eagle3MLADecoderLayer with self.mlp = "
            "DeepseekV2MLP (line 126); no MoE TopK is constructed."
        ),
    ),
    DraftInventoryEntry(
        source_architectures=(),
        draft_architecture="Gemma4AssistantForCausalLM",
        moe_bearing=True,
        draft_signal="always_draft",
        opt_out_injection_point="python/sglang/srt/models/gemma4_causal.py:Gemma4MoE",
        rationale=(
            "gemma4_mtp.py:95 constructs Gemma4TextModel(..., "
            "allow_routed_experts_capture=False); the kwarg threads through "
            "Gemma4DecoderLayer / Gemma4MoE and lands as "
            "`allow_routed_experts_capture=False` on the MoE TopK. This is the "
            "only assistant arch that auto-promotes to FROZEN_KV_MTP in "
            "server_args._resolve_speculative_algorithm_alias."
        ),
    ),
)


_BY_ARCH = {entry.draft_architecture: entry for entry in INVENTORY}


def lookup_draft_arch(arch_name: str) -> Optional[DraftInventoryEntry]:
    """Return the inventory entry for the given draft architecture name, or
    `None` if the architecture is not registered. The runtime guard treats
    `None` as a hard failure (refuse to start).
    """
    return _BY_ARCH.get(arch_name)
