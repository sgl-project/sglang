from __future__ import annotations

from typing import Optional

from sglang.srt.configs.gigachat35 import GigaChat35Config
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)


class GigaChat4Config(GigaChat35Config):
    """GigaChat 3.5 plus DSA sparse attention (enabled by ``index_topk``,
    MTP/NextN layer included) and final-logit soft-capping."""

    model_type = "gigachat4"

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
        **kwargs,
    ):
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        if indexer_norm_type != "LlamaRMSNorm":
            raise ValueError(
                f"Unsupported indexer_norm_type {indexer_norm_type!r} for GigaChat 4; "
                "its DSA indexer keys are normalized with a plain RMSNorm."
            )
        # Translate GigaChat's indexer field names for the shared DSA Indexer.
        self.index_k_norm_type = "rms"
        self.index_rotate_activation = bool(indexer_rotate_activation_enabled)
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
        super().__init__(**kwargs)


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=GigaChat4Config,
        backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
        arch_names=[
            "GigaChat4ForCausalLM",
            "GigaChat4ForCausalLMNextN",
        ],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
    )
)
