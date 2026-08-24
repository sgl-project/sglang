"""PLaMo3 model configuration."""

import warnings
from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig


def is_full_attn(sliding_window_pattern: int, layer_idx: int) -> bool:
    return not bool((layer_idx + 1) % sliding_window_pattern)


class Plamo3Config(PretrainedConfig):  # type: ignore[misc]
    model_type: str = "plamo3"

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        rms_norm_eps: float = 1e-6,
        # Embedding
        scale_embedding: bool = False,
        tie_word_embeddings: bool = True,
        # Attention
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 2048,
        window_size: int = 128,
        sliding_window_pattern: int = 8,
        rope_theta: int = 150_000,
        rope_local_theta: int = 10_000,
        rope_scaling_factor: float = 1,
        initial_context_length: Optional[int] = None,
        # MLP
        intermediate_size: int = 13312,
        # Tokenizer
        vocab_size: int = 32000,
        tokenizer_class: str = "Plamo3Tokenizer",
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Evaluation
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        legacy_rope_theta = kwargs.pop("rope_global_theta", None)
        if legacy_rope_theta is not None and rope_theta == 150_000:
            rope_theta = legacy_rope_theta
        legacy_sliding_window = kwargs.pop("sliding_window", None)
        if isinstance(legacy_sliding_window, list):
            non_full_attention_windows = [
                value for value in legacy_sliding_window if value is not None
            ]
            if non_full_attention_windows:
                first_window_size = non_full_attention_windows[0]
                if all(
                    value == first_window_size for value in non_full_attention_windows
                ):
                    window_size = first_window_size
        elif isinstance(legacy_sliding_window, int):
            window_size = legacy_sliding_window

        self.window_size = window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.initial_context_length = initial_context_length
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding

        super().__init__(
            tokenizer_class=tokenizer_class,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def sliding_window(self) -> int:
        return self.window_size

    @property
    def interleaved_sliding_window(self) -> list[int | None]:
        interleaved_sliding_window: list[int | None] = []
        for i in range(self.num_hidden_layers):
            if is_full_attn(self.sliding_window_pattern, i):
                interleaved_sliding_window.append(None)
            else:
                interleaved_sliding_window.append(self.window_size)
        assert len(interleaved_sliding_window) == self.num_hidden_layers
        return interleaved_sliding_window

    @property
    def layer_types(self) -> list[str]:
        return [
            "full_attention" if sliding_window_size is None else "sliding_attention"
            for sliding_window_size in self.interleaved_sliding_window
        ]

    @property
    def layers_block_type(self) -> list[str]:
        return ["attention" for _ in range(self.num_hidden_layers)]

    @property
    def rope_parameters(self) -> dict[str, Any]:
        rope_parameters_all = {
            "full_attention": {},
            "sliding_attention": {
                "rope_theta": self.rope_local_theta,
                "rope_type": "default",
            },
        }
        if self.rope_scaling_factor == 1:
            assert self.initial_context_length is None
            rope_parameters_all["full_attention"] = {
                "rope_theta": self.rope_theta,
                "rope_type": "default",
            }
        else:
            assert self.initial_context_length is not None
            rope_parameters_all["full_attention"] = {
                "rope_theta": self.rope_theta,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": self.rope_scaling_factor,
                "original_max_position_embeddings": self.initial_context_length,
                "rope_type": "yarn",
                "truncate": False,
            }
        return {
            layer_type: rope_parameters_all[layer_type]
            for layer_type in set(self.layer_types)
        }

    @rope_parameters.setter
    def rope_parameters(self, rope_parameters: dict[str, Any]) -> None:
        # Transformers may normalize this property during config initialization.
        pass

    def _validate_yarn_rope_parameters(
        self,
        rope_parameters: dict[str, Any],
        ignore_keys: set[str] | None = None,
    ) -> None:
        required_keys = {
            "rope_type",
            "factor",
            "original_max_position_embeddings",
        }
        optional_keys = {
            "rope_theta",
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        }
        received_keys = set(rope_parameters)
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type,
            received_keys,
            required_keys,
            optional_keys,
            ignore_keys=ignore_keys,
        )

        factor = rope_parameters["factor"]
        if not isinstance(factor, (float, int)) or factor < 1.0:
            warnings.warn(
                f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}",
                stacklevel=2,
            )

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (
            not isinstance(attention_factor, float) or attention_factor < 0
        ):
            warnings.warn(
                "`rope_parameters`'s attention_factor field must be a float "
                f"greater than 0, got {attention_factor}",
                stacklevel=2,
            )
        beta_fast = rope_parameters.get("beta_fast")
        if beta_fast is not None and not isinstance(beta_fast, (float, int)):
            warnings.warn(
                "`rope_parameters`'s beta_fast field must be a float or int, "
                f"got {beta_fast}",
                stacklevel=2,
            )
        beta_slow = rope_parameters.get("beta_slow")
        if beta_slow is not None and not isinstance(beta_slow, (float, int)):
            warnings.warn(
                "`rope_parameters`'s beta_slow field must be a float or int, "
                f"got {beta_slow}",
                stacklevel=2,
            )
        if (beta_fast or 32) < (beta_slow or 1):
            warnings.warn(
                "`rope_parameters`'s beta_fast field must be greater than "
                f"beta_slow, got beta_fast={beta_fast} (defaults to 32 if None) "
                f"and beta_slow={beta_slow} (defaults to 1 if None)",
                stacklevel=2,
            )

    @property
    def rope_local_base_freq(self) -> int:
        return self.rope_local_theta
