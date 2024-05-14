from typing import Optional

from sglang.srt.hf_transformers_utils import get_config, get_context_length


class ModelConfig:
    def __init__(
        self,
        path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        model_overide_args: Optional[dict] = None,
    ) -> None:
        self.path = path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.hf_config = get_config(self.path, trust_remote_code, revision)

        if model_overide_args is not None:
            self.hf_config.update(model_overide_args)

        if context_length is not None:
            self.context_len = context_length
        else:
            self.context_len = get_context_length(self.hf_config)

        # Unify the config keys for hf_config
        self.head_dim = getattr(
            self.hf_config,
            "head_dim",
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = getattr(self.hf_config, "num_key_value_heads", None)

        # for Dbrx and MPT models
        if self.hf_config.model_type in ["dbrx", "mpt"]:
            self.num_key_value_heads = getattr(
                self.hf_config.attn_config, "kv_n_heads", None
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.hf_config.hidden_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.vocab_size = self.hf_config.vocab_size
