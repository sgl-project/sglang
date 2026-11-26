from __future__ import annotations

from typing import Optional

from sglang.srt.configs.gigachat35 import GigaChat35Config
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)


class GigaChat4Config(GigaChat35Config):
    model_type = "gigachat4"
    dsa_draft_attends_to_own_chain = True
    gdn_round_beta_to_input_dtype = True

    def __init__(
        self,
        index_topk: Optional[int] = None,
        index_n_heads: Optional[int] = None,
        index_head_dim: Optional[int] = None,
        indexer_norm_type: str = "LlamaRMSNorm",
        indexer_rotate_activation_enabled: Optional[bool] = None,
        dense_attention_layers: Optional[list[int]] = None,
        nextn_is_linear: bool = False,
        final_logit_softcapping: Optional[float] = None,
        norm_type: str = "GatedNorm",
        swiglu_limit: Optional[float] = 10.0,
        num_nextn_predict_layers: int = 1,
        **kwargs,
    ):
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        for name, default in (
            ("index_topk_freq", 1),
            ("index_topk_pattern", None),
            ("indexer_types", None),
            ("cli_factor", 1),
        ):
            value = kwargs.get(name, default)
            if value is not None and value != default:
                raise NotImplementedError(
                    "DSA index sharing is not supported for GigaChat 4 "
                    f"({name}={value!r}); each full-attention layer must run "
                    "its own indexer."
                )
        if indexer_norm_type != "LlamaRMSNorm":
            raise ValueError(
                f"Unsupported indexer_norm_type {indexer_norm_type!r} for GigaChat 4; "
                "its DSA indexer keys are normalized with a plain RMSNorm."
            )
        self.index_k_norm_type = "rms"
        self.index_apply_hadamard = bool(indexer_rotate_activation_enabled)
        if dense_attention_layers:
            raise NotImplementedError(
                "dense_attention_layers is not supported yet: every GigaChat 4 "
                "full-attention layer runs the DSA indexer."
            )
        if nextn_is_linear:
            raise NotImplementedError(
                "A linear-attention (GDN) MTP block is not supported: draft GDN "
                "state init and rollback are not wired up."
            )
        self.final_logit_softcapping = final_logit_softcapping
        super().__init__(
            norm_type=norm_type,
            swiglu_limit=swiglu_limit,
            num_nextn_predict_layers=num_nextn_predict_layers,
            **kwargs,
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=GigaChat4Config,
        config_predicate=lambda config: (
            "GigaChat4DFlash2DraftModel" not in (config.architectures or ())
        ),
        backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
        arch_names=[
            "GigaChat4ForCausalLM",
            "GigaChat4ForCausalLMNextN",
        ],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
    )
)
