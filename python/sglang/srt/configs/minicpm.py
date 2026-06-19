from dataclasses import dataclass, field

import numpy as np
from transformers import PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.distributed.utils import divide
from sglang.srt.runtime_context import get_parallel


@dataclass(kw_only=True, frozen=True)
class SimpleGLAStateShape:
    conv: list[tuple[int, int]] = field(default_factory=list)
    temporal: tuple[int, int, int]

    num_heads: int
    head_dim: int
    state_size: int

    @staticmethod
    def create(
        *,
        tp_world_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
    ) -> "SimpleGLAStateShape":
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return SimpleGLAStateShape(
            temporal=temporal_state_shape,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
        )


@dataclass(kw_only=True, frozen=True)
class SimpleGLACacheParams(BaseLinearStateParams):
    shape: SimpleGLAStateShape

    @property
    def mamba_cache_per_req(self) -> int:
        ssm_numel = int(np.prod(self.shape.temporal))
        return ssm_numel * self.dtype.temporal.itemsize * len(self.layers)


class MiniCPMHybridConfig(PretrainedConfig):
    """
    Configuration class for hybrid MiniCPM models.

    This config extends PretrainedConfig to match the pattern used by other
    hybrid/linear attention models (Falcon H1, Nemotron H, Kimi Linear, etc.)
    and provides cache parameters for the Simple GLA attention mechanism.
    """

    model_type = "minicpm_sala"

    def __init__(
        self,
        # Base model config fields
        vocab_size=150528,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        intermediate_size=14336,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        rope_scaling=None,
        # MiniCPM-specific hybrid config fields
        mixer_types=None,
        minicpm4=None,
        lightning=None,
        lightning_nh=16,
        lightning_nkv=16,
        lightning_head_dim=64,
        lightning_scale="1/sqrt(d)",
        attn_use_rope=True,
        attn_use_output_gate=False,
        # Sparse attention config fields
        sparse_block_size=32,
        sparse_dense_len=512,
        sparse_init_blocks=1,
        sparse_kernel_size=32,
        sparse_kernel_stride=16,
        sparse_topk=8,
        sparse_window_size=64,
        sparse_use_nope=False,
        **kwargs,
    ):
        # Pop structured MiniCPM fields before PretrainedConfig stores unknown kwargs.
        sparse_config = kwargs.pop("sparse_config", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # Hybrid config fields
        self.mixer_types = mixer_types if mixer_types is not None else None
        self.minicpm4 = minicpm4
        self.lightning = lightning
        self.lightning_nh = lightning_nh
        self.lightning_nkv = lightning_nkv
        self.lightning_head_dim = lightning_head_dim
        self.lightning_scale = lightning_scale
        self.attn_use_rope = attn_use_rope
        self.attn_use_output_gate = attn_use_output_gate
        # Sparse attention config fields
        self.sparse_block_size = sparse_block_size
        self.sparse_dense_len = sparse_dense_len
        self.sparse_init_blocks = sparse_init_blocks
        self.sparse_kernel_size = sparse_kernel_size
        self.sparse_kernel_stride = sparse_kernel_stride
        self.sparse_topk = sparse_topk
        self.sparse_window_size = sparse_window_size
        self.sparse_use_nope = sparse_use_nope
        # Load sparse_config from original config if available (for backward compatibility)
        self.has_sparse_config = sparse_config is not None
        if sparse_config is not None:
            self.sparse_block_size = sparse_config.get(
                "block_size", self.sparse_block_size
            )
            self.sparse_dense_len = sparse_config.get(
                "dense_len", self.sparse_dense_len
            )
            self.sparse_init_blocks = sparse_config.get(
                "init_blocks", self.sparse_init_blocks
            )
            self.sparse_kernel_size = sparse_config.get(
                "kernel_size", self.sparse_kernel_size
            )
            self.sparse_kernel_stride = sparse_config.get(
                "kernel_stride", self.sparse_kernel_stride
            )
            self.sparse_topk = sparse_config.get("topk", self.sparse_topk)
            self.sparse_window_size = sparse_config.get(
                "window_size", self.sparse_window_size
            )
            self.sparse_use_nope = sparse_config.get("use_nope", self.sparse_use_nope)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def mamba2_cache_params(self):
        """Return Simple GLA cache parameters for lightning attention layers."""
        if self.mixer_types is None:
            lightning_layer_ids = []
        else:
            lightning_layer_ids = [
                i
                for i, mixer_type in enumerate(self.mixer_types)
                if mixer_type in ["lightning", "lightning_attn", "lightning-attn"]
            ]

        if (
            not lightning_layer_ids
            or not self.lightning_nkv
            or not self.lightning_head_dim
        ):
            return None

        shape = SimpleGLAStateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            num_heads=self.lightning_nkv,
            head_dim=self.lightning_head_dim,
            state_size=self.lightning_head_dim,
        )

        return SimpleGLACacheParams(shape=shape, layers=lightning_layer_ids)

    @property
    def full_attention_layer_ids(self):
        if self.mixer_types is None:
            return list(range(self.num_hidden_layers))
        else:
            return [
                i
                for i, mixer_type in enumerate(self.mixer_types)
                if mixer_type
                in ["minicpm4", "minicpm", "standard", "attention", "attn"]
            ]

    @property
    def has_minicpm_sparse_attention(self) -> bool:
        """Check if this config has MiniCPM sparse attention layers."""
        return self.has_sparse_config and (
            self.mixer_types is None or any(mt == "minicpm4" for mt in self.mixer_types)
        )

    @property
    def has_lightning_layers(self) -> bool:
        """Check if this config has lightning attention layers."""
        return self.mixer_types is not None and any(
            mt in ["lightning", "lightning_attn", "lightning-attn"]
            for mt in self.mixer_types
        )

    @property
    def sparse_layer_ids(self) -> list:
        """Get the indices of layers with sparse attention."""
        if self.has_sparse_config:
            if self.mixer_types is None:
                return list(range(self.num_hidden_layers))
            else:
                return [i for i, mt in enumerate(self.mixer_types) if mt == "minicpm4"]
        else:
            return []

    @property
    def lightning_layer_ids(self) -> list:
        """Get the indices of layers with lightning attention."""
        if self.mixer_types is None:
            return []
        else:
            return [
                i
                for i, mt in enumerate(self.mixer_types)
                if mt in ["lightning", "lightning_attn", "lightning-attn"]
            ]


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=MiniCPMHybridConfig,
        backend_class_name=(
            "sglang.srt.layers.attention.minicpm.simple_gla.SimpleGLAAttnBackend"
        ),
        arch_names=["MiniCPMForCausalLM", "MiniCPMSALAForCausalLM"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        req_to_token_pool_factory_name=(
            "sglang.srt.layers.attention.minicpm.cache.create_req_to_token_pool"
        ),
    )
)
