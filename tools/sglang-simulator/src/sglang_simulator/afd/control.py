"""Compatibility entry; implementation lives in the SGLang adapter."""


def run(provider, **kwargs):
    from .adapters.sglang.afd import run as execute

    return execute(provider, **kwargs)
