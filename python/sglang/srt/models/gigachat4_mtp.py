"""GigaChat 4 multi-head MTP.

The GigaChat 3.5 NextN block built from the GigaChat 4 decoder layer: the
NextN layer is DSA like the target, with its own indexer and per-step top-k.
The constructors mirror their 3.5 counterparts with the GigaChat 4 classes
swapped in.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.gigachat4 import GigaChat4Config
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.gigachat35 import build_norm
from sglang.srt.models.gigachat35_mtp import (
    GigaChat35ForCausalLMNextN,
    GigaChat35ModelNextN,
)
from sglang.srt.models.gigachat4 import GigaChat4DecoderLayer
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class GigaChat4ModelNextN(GigaChat35ModelNextN):
    def __init__(
        self,
        config: GigaChat4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = build_norm(config, config.hidden_size)
        self.hnorm = build_norm(config, config.hidden_size)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        with layer_stack():
            self.decoder = GigaChat4DecoderLayer(
                config=config,
                layer_id=0,
                quant_config=quant_config,
                prefix=add_prefix("decoder", prefix),
                alt_stream=self.alt_stream,
                is_nextn=True,
            )

        self.shared_head = nn.Module()
        self.shared_head.norm = build_norm(config, config.hidden_size)


class GigaChat4ForCausalLMNextN(GigaChat35ForCausalLMNextN):
    def __init__(
        self,
        config: GigaChat4Config,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.num_fused_shared_experts = 0
        self.draft_model_idx = draft_model_idx or 0

        self.model = GigaChat4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)


EntryClass = [GigaChat4ForCausalLMNextN]
