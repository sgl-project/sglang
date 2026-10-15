"""Optional Mooncake storage for immutable model lookup tables."""


def enabled() -> bool:
    from sglang.srt.runtime_context import get_memory

    return get_memory().engram_backend == "mooncake"


def make_embedding(native_cls, *args, **kwargs):
    if not enabled():
        return native_cls(*args, **kwargs)

    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.mem_cache.engram.embedding import (
        MooncakeScaledEmbedding,
        MooncakeVocabParallelEmbedding,
    )

    cls = (
        MooncakeVocabParallelEmbedding
        if issubclass(native_cls, VocabParallelEmbedding)
        else MooncakeScaledEmbedding
    )
    return cls(*args, **kwargs)
