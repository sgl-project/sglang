from dataclasses import dataclass
from typing import Optional

from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


@dataclass
class ModelInfo:
    hf_config: Optional[dict] = None
    model_path: Optional[str] = None

    attention_arch: Optional[str] = None  # MLA | MHA
    context_len: Optional[int] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    v_head_dim: Optional[int] = None
    vocab_size: Optional[int] = None

    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None

    torch_dtype: Optional[str] = None

    def is_mla(self) -> bool:
        return self.attention_arch == "MLA"
