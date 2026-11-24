"""GigaChat 4 multi-head MTP.

The GigaChat 3.5 NextN block built from the GigaChat 4 decoder layer: the
NextN layer is DSA like the target, with its own indexer and per-step top-k.
"""

from __future__ import annotations

from sglang.srt.models.gigachat4 import GigaChat4DecoderLayer
from sglang.srt.models.gigachat35_mtp import (
    GigaChat35ForCausalLMNextN,
    GigaChat35ModelNextN,
)


class GigaChat4ModelNextN(GigaChat35ModelNextN):
    decoder_layer_cls = GigaChat4DecoderLayer


class GigaChat4ForCausalLMNextN(GigaChat35ForCausalLMNextN):
    model_cls = GigaChat4ModelNextN


EntryClass = [GigaChat4ForCausalLMNextN]
