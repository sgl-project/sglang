"""Agens Volundr configs.

VolundrTextConfig subclasses Qwen3_5TextConfig for the shared projection / rope plumbing and adds
what Volundr does differently:

  * the layer map comes from `mixer_types` ("kda" | "bcsa" | "attn"), the checkpoint's own record,
    not from `full_attention_interval`: the attention layers do not sit on a fixed interval.
  * the recurrent-state layout is KDA's (KimiLinearCacheParams: conv state [kernel-1, dim], sliced
    by q/k/v sub-block), not GDN's.
  * the BCSA, Engram and mHC hyper-parameters, with their defaults.

VolundrConfig (model_type "volundr") is the vision + text checkpoint config; it mirrors
Qwen3_5Config with VolundrTextConfig as the text sub-config.
"""

from typing import List

from transformers import PretrainedConfig

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    mamba2_state_dtype,
)
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from sglang.srt.runtime_context import get_parallel


class VolundrTextConfig(Qwen3_5TextConfig):
    model_type = "volundr_text"

    def __init__(
        self,
        mixer_types=None,
        layer_types=None,
        kda_gate_rank=128,
        output_gate_type="swish",
        bcsa_window=4096,
        bcsa_compress=4,
        bcsa_topk=512,
        bcsa_index_heads=4,
        bcsa_index_head_dim=128,
        engram_layers=None,
        engram_orders=(2, 3),
        engram_rows=0,
        engram_dim=512,
        mhc_streams=0,
        mhc_sinkhorn_iters=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mixer_types:
            self.mixer_types = list(mixer_types)
        elif layer_types:
            self.mixer_types = [
                "kda" if t == "linear_attention" else "bcsa" for t in layer_types
            ]
        else:
            self.mixer_types = ["kda"] * self.num_hidden_layers
        if len(self.mixer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mixer_types has {len(self.mixer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )
        self.layer_types = [
            "linear_attention" if m == "kda" else "full_attention"
            for m in self.mixer_types
        ]
        self.kda_gate_rank = kda_gate_rank
        self.output_gate_type = output_gate_type
        self.bcsa_window = bcsa_window
        self.bcsa_compress = bcsa_compress
        self.bcsa_topk = bcsa_topk
        self.bcsa_index_heads = bcsa_index_heads
        self.bcsa_index_head_dim = bcsa_index_head_dim
        self.engram_layers = list(engram_layers or [])
        self.engram_orders = list(engram_orders)
        self.engram_rows = engram_rows
        self.engram_dim = engram_dim
        self.mhc_streams = mhc_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters

    # ---- layer map: from the tensors, not from full_attention_interval --------------------------
    @property
    def layers_block_type(self) -> List[str]:
        return [
            "linear_attention" if m == "kda" else "full_attention"
            for m in self.mixer_types
        ]

    @property
    def linear_layer_ids(self) -> List[int]:
        return [i for i, m in enumerate(self.mixer_types) if m == "kda"]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        return [i for i, m in enumerate(self.mixer_types) if m != "kda"]

    # ---- recurrent state: KDA layout, not GDN ---------------------------------------------------
    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        return KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=get_parallel().attn_tp_size,
                num_heads=self.linear_num_value_heads,
                head_dim=self.linear_value_head_dim,
                num_k_heads=self.linear_num_key_heads,
                head_k_dim=self.linear_key_head_dim,
                conv_kernel_size=self.linear_conv_kernel_dim,
            ),
            dtype=mamba2_state_dtype(self),
            layers=self.linear_layer_ids,
        )


class VolundrConfig(PretrainedConfig):
    """Full checkpoint: Volundr text tower + vision tower."""

    model_type = "volundr"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": VolundrTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


# Route Volundr's linear layers to the KDA backend rather than the GDN one: KDA's forget gate is
# per channel, GDN's is one scalar per head. hybrid_arch.hybrid_gdn_config would otherwise claim
# this config through its Qwen3NextConfig base; an explicit registration wins there.
def _register_volundr_linear_attn() -> None:
    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        get_linear_attn_spec_by_arch,
        register_linear_attn_model,
    )

    if get_linear_attn_spec_by_arch("VolundrForCausalLM") is not None:
        return
    register_linear_attn_model(
        LinearAttnModelSpec(
            config_class=VolundrTextConfig,
            backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
            arch_names=["VolundrForCausalLM", "VolundrForConditionalGeneration"],
            uses_mamba_radix_cache=True,
            support_mamba_cache=True,
            # the full checkpoint's hf_config is VolundrConfig; match on its text sub-config
            # (get_text_config() is the identity on a text-only VolundrTextConfig)
            unwrap_text_config=True,
        )
    )


_register_volundr_linear_attn()
