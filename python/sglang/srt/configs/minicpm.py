from transformers import PretrainedConfig

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.runtime_context import get_parallel

_MIXER_TYPE_ALIASES = {
    "minicpm4": "minicpm4",
    "minicpm": "minicpm4",
    "standard": "minicpm4",
    "attention": "minicpm4",
    "attn": "minicpm4",
    "lightning": "lightning-attn",
    "lightning_attn": "lightning-attn",
    "lightning-attn": "lightning-attn",
}


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
        head_dim=None,
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
        scale_emb=12,
        scale_depth=1.4,
        dim_model_base=256,
        # MiniCPM-specific hybrid config fields
        mixer_types=None,
        lightning_nh=None,
        lightning_nkv=None,
        lightning_head_dim=None,
        lightning_scale="1/sqrt(d)",
        lightning_layerwise_decay=False,
        lightning_use_rope=True,
        use_output_gate=False,
        attention_bias=False,
        use_output_norm=False,
        qk_norm=True,
        attn_use_rope=True,
        attn_use_output_gate=False,
        sparse_config=None,
        **kwargs,
    ):
        for unused_field in ("minicpm4", "lightning", "sparse_use_nope"):
            kwargs.pop(unused_field, None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.scale_emb = scale_emb
        self.scale_depth = scale_depth
        self.dim_model_base = dim_model_base
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # Hybrid config fields
        if not mixer_types:
            mixer_types = ["minicpm4"]
        elif len(mixer_types) > num_hidden_layers:
            raise ValueError(f"Invalid number of mixer types: {len(mixer_types)}")
        try:
            mixer_types = [
                _MIXER_TYPE_ALIASES[mixer_type] for mixer_type in mixer_types
            ]
        except KeyError as exc:
            raise ValueError(f"Unsupported mixer type: {exc.args[0]}") from exc
        repeats = (num_hidden_layers + len(mixer_types) - 1) // len(mixer_types)
        self.mixer_types = (mixer_types * repeats)[:num_hidden_layers]
        self.lightning_nh = (
            lightning_nh if lightning_nh is not None else num_attention_heads
        )
        self.lightning_nkv = (
            lightning_nkv if lightning_nkv is not None else num_key_value_heads
        )
        self.lightning_head_dim = (
            lightning_head_dim if lightning_head_dim is not None else self.head_dim
        )
        if (
            "lightning-attn" in self.mixer_types
            and self.lightning_nh != self.lightning_nkv
        ):
            raise ValueError(
                "MiniCPM Lightning attention requires equal query and KV head "
                "counts because the seg_la backend does not support GQA: "
                f"lightning_nh={self.lightning_nh}, "
                f"lightning_nkv={self.lightning_nkv}"
            )
        self.lightning_scale = lightning_scale
        self.lightning_layerwise_decay = lightning_layerwise_decay
        self.lightning_use_rope = lightning_use_rope
        self.use_output_gate = use_output_gate
        self.attention_bias = attention_bias
        self.use_output_norm = use_output_norm
        self.qk_norm = qk_norm
        self.attn_use_rope = attn_use_rope
        self.attn_use_output_gate = attn_use_output_gate
        self.sparse_config = sparse_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_linear_key_value_heads(self) -> int:
        return self.lightning_nkv

    @property
    def mamba2_cache_params(self):
        """Return linear-attention cache parameters for lightning layers."""
        lightning_layer_ids = self.lightning_layer_ids

        if (
            not lightning_layer_ids
            or not self.lightning_nkv
            or not self.lightning_head_dim
        ):
            return None

        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=0,
            n_groups=0,
            num_heads=self.lightning_nkv,
            head_dim=self.lightning_head_dim,
            state_size=self.lightning_head_dim,
            conv_kernel=1,
        )

        return Mamba2CacheParams(shape=shape, layers=lightning_layer_ids)

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i, mixer_type in enumerate(self.mixer_types)
            if mixer_type == "minicpm4"
        ]

    @property
    def has_minicpm_sparse_attention(self) -> bool:
        """Check if this config has MiniCPM sparse attention layers."""
        return self.sparse_config is not None and any(
            mt == "minicpm4" for mt in self.mixer_types
        )

    @property
    def has_lightning_layers(self) -> bool:
        """Check if this config has lightning attention layers."""
        return any(mt == "lightning-attn" for mt in self.mixer_types)

    @property
    def lightning_layer_ids(self) -> list:
        """Get the indices of layers with lightning attention."""
        return [i for i, mt in enumerate(self.mixer_types) if mt == "lightning-attn"]
