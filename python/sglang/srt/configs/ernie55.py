# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config for ERNIE 5.5 Flash, a hybrid MLA + HCA mixture-of-experts model.

The arch string and ``model_type`` name the architecture, not the tier, so
"Flash" appears only in user-facing messages and served-model names.

``compress_ratios[i]`` names the layer kind: ``128`` is an HCA layer
(attention over the whole context compressed 128:1 plus an uncompressed
``sliding_window``-token window), ``-2`` a full-context MLA layer with a DSA
lightning indexer.

Glossary for the checkpoint's own field names: *HCA* is compressed
sliding-window attention, *MLA* full-context latent attention, *DSA* the sparse
indexer that selects the positions MLA reads, *mHC* the manifold-constrained
hyper-connections (the ``hc_*`` fields), *VHA* the cross-head low-rank mix
applied to the attention output, *SiTU* the gated activation
(``activation_situ_*``), ``wo_a``/``wo_b`` the two halves of the grouped output
projection, and ``multimax`` a four-segment transform of the logits.
"""

from typing import Optional, Tuple

from transformers import PretrainedConfig

ERNIE5_5_MOE_ARCH = "Ernie5_5_MoeForCausalLM"
ERNIE5_5_DSPARK_ARCH = "Ernie5_5_MoeForCausalLMDSpark"
ERNIE5_5_NEXTN_ARCH = "Ernie5_5_MoeForCausalLMNextN"
ERNIE5_5_ARCHS = (ERNIE5_5_MOE_ARCH, ERNIE5_5_DSPARK_ARCH, ERNIE5_5_NEXTN_ARCH)
COMPRESS_RATIO_MLA = -2
COMPRESS_RATIO_HCA = 128

# Read directly by the model, backend and pool; PretrainedConfig defaults none of them.
_REQUIRED_CHECKPOINT_FIELDS = (
    "activation_situ_beta",
    "activation_situ_linear_beta",
    "hc_eps",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hybrid_mla_kv_lora_rank",
    "hybrid_mla_num_attention_heads",
    "hybrid_mla_q_lora_rank",
    "hybrid_mla_qk_nope_head_dim",
    "hybrid_mla_qk_rope_head_dim",
    "hybrid_mla_v_head_dim",
    "index_head_dim",
    "index_topk",
    "latent_moe_use_norm",
    "moe_latent_size",
    "moe_split_feature_routing",
    "multimax",
    "o_lora_rank",
    "routed_scaling_factor_learnable",
)

_REQUIRED_CHECKPOINT_VALUES = {
    "hidden_act": "situ",
    "latent_moe_use_norm": True,
    "moe_split_feature_routing": True,
    "multimax": "lm_head",
    "n_group": 1,
    "routed_scaling_factor_learnable": True,
    "topk_group": 1,
}


def to_dspark_draft_config(hf_config) -> None:
    """Reshape a target config, in place, into the DSpark draft's it bundles."""
    from sglang.srt.speculative.dspark_components.dspark_config import (
        checkpoint_bundles_dspark_draft,
    )

    if not checkpoint_bundles_dspark_draft(hf_config):
        raise ValueError(
            "this ERNIE 5.5 Flash checkpoint bundles no DSpark draft "
            "(its config has no dspark_* keys)"
        )
    num_stages = len(hf_config.dspark_target_layer_ids)
    hf_config.architectures = [ERNIE5_5_DSPARK_ARCH]
    hf_config.num_hidden_layers = num_stages
    hf_config.first_k_dense_replace = 0
    hf_config.compress_ratios = [0] * num_stages
    hf_config.checkpoint_compress_ratios = [0] * num_stages
    # The HCA backend accepts 512 or 1024; the draft has no DSA layer.
    hf_config.index_topk = 512


def to_nextn_draft_config(hf_config) -> None:
    """Reshape a target config, in place, into its MTP (NextN) draft's:
    the one decoder layer the checkpoint stores after the main stack, as layer 0."""
    mtp_layer = hf_config.num_hidden_layers
    ratios = hf_config.checkpoint_compress_ratios
    if hf_config.num_nextn_predict_layers != 1 or len(ratios) <= mtp_layer:
        raise ValueError(
            "this ERNIE 5.5 Flash checkpoint has no MTP layer "
            f"(num_nextn_predict_layers={hf_config.num_nextn_predict_layers})"
        )
    if ratios[mtp_layer] != COMPRESS_RATIO_MLA:
        raise NotImplementedError(
            f"the ERNIE 5.5 Flash MTP layer is served as an MLA layer, got "
            f"compress ratio {ratios[mtp_layer]}"
        )
    hf_config.architectures = [ERNIE5_5_NEXTN_ARCH]
    hf_config.mtp_checkpoint_layer_id = mtp_layer
    hf_config.num_hidden_layers = 1
    hf_config.first_k_dense_replace = 0
    hf_config.checkpoint_compress_ratios = [COMPRESS_RATIO_MLA]
    hf_config.compress_ratios = [0]


class Ernie5_5_MoeConfig(PretrainedConfig):
    # The _CONFIG_REGISTRY key; get_config re-parses the checkpoint through this class.
    model_type = "ernie5_5_moe"
    # __init__ reads num_hidden_layers, which only the checkpoint supplies.
    has_no_defaults_at_init = True

    def __init__(
        self,
        sliding_window: int = 128,
        compress_ratios=(),
        num_nextn_predict_layers: int = 1,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        **kwargs,
    ):
        rs = rope_scaling or {}
        kwargs.setdefault(
            "rope_parameters",
            {
                **rs,
                "rope_theta": rope_theta,
                "rope_type": rs.get("rope_type", rs.get("type", "default")),
            },
        )
        super().__init__(**kwargs)
        self.sliding_window = sliding_window
        self.num_nextn_predict_layers = num_nextn_predict_layers
        # Fixed by the architecture, not shipped in the checkpoint: the mHC
        # post-mapping scale (2 * sigmoid).
        self.hc_post_mult_value = 2.0
        # Fields the HCA attention and KV pool read.
        self.q_head_norm = True
        self.kv_source_layer_ids = ()
        self.is_hybrid_swa = True
        # Fields the MLA layers' DSA indexer reads: run its fused NeoX-rope kernels,
        # and keep wk in bf16 (the fp8 release lists no ignored_layers for it).
        self.index_fused_neox_rope = True
        self.index_wk_unquantized = True

        ratios = [int(r) for r in compress_ratios]
        unsupported = sorted(set(ratios) - {COMPRESS_RATIO_MLA, COMPRESS_RATIO_HCA})
        if unsupported:
            raise ValueError(
                f"ERNIE 5.5 Flash supports compress_ratios {COMPRESS_RATIO_HCA} (HCA) "
                f"and {COMPRESS_RATIO_MLA} (MLA), got {unsupported}"
            )
        num_hidden_layers = self.num_hidden_layers
        num_nextn = self.num_nextn_predict_layers
        # The checkpoint carries one extra entry per MTP (NextN) layer.
        if ratios and len(ratios) not in (
            num_hidden_layers,
            num_hidden_layers + num_nextn,
        ):
            raise ValueError(
                f"len(compress_ratios)={len(ratios)} must be num_hidden_layers "
                f"({num_hidden_layers}) or num_hidden_layers + num_nextn "
                f"({num_hidden_layers + num_nextn})"
            )
        # The checkpoint's own list, kept verbatim: `-2` marks an MLA layer.
        self.checkpoint_compress_ratios = ratios
        # What the compressed-KV pool and backend read: ratio 0 means no compressed
        # state, which is right for an MLA layer since it keeps its KV elsewhere.
        self.compress_ratios = [0 if r == COMPRESS_RATIO_MLA else r for r in ratios]

    def validate_checkpoint_fields(self) -> None:
        """Raise if the checkpoint omits a field the model reads,
        or sets one to a value the model does not implement; ``__init__`` skips this."""
        missing = [f for f in _REQUIRED_CHECKPOINT_FIELDS if not hasattr(self, f)]
        if missing:
            raise ValueError(
                f"this ERNIE 5.5 Flash checkpoint is missing {len(missing)} config "
                f"field(s): {', '.join(missing)}"
            )
        wrong = {
            name: getattr(self, name)
            for name, want in _REQUIRED_CHECKPOINT_VALUES.items()
            if getattr(self, name) != want
        }
        if wrong:
            raise ValueError(
                f"this ERNIE 5.5 Flash checkpoint sets {wrong}, but the model "
                f"implements only {_REQUIRED_CHECKPOINT_VALUES}"
            )

    @property
    def mla_layer_ids(self) -> Tuple[int, ...]:
        """MLA layers of the main stack; the trailing MTP (NextN) entries are excluded."""
        return tuple(
            i
            for i, r in enumerate(
                self.checkpoint_compress_ratios[: self.num_hidden_layers]
            )
            if r == COMPRESS_RATIO_MLA
        )
