"""Compatibility entry for the SGLang adapter source guard."""


def verify_source(source_root, *, mode=None):
    from .adapters.sglang.source import verify_source as verify

    return verify(source_root, mode=mode)
