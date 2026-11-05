"""Explicit selection of source-specific execution adapters."""

from importlib import import_module


def create_adapter(source_root, mode, *, factory=None, routing="f-side"):
    # Plugins are explicit trusted Python code, never selected by a cost file.
    if factory is not None:
        if routing != "f-side":
            raise ValueError("AFD_ADAPTER_FACTORY_ROUTING_CONFLICT")
        module, separator, name = factory.partition(":")
        if not separator or not module or not name:
            raise ValueError("AFD_ADAPTER_FACTORY_INVALID: expected module:factory")
        return getattr(import_module(module), name)(source_root, mode)
    if routing == "direct":
        from .sglang.direct import DirectAdapter

        return DirectAdapter(source_root, mode)
    if routing != "f-side":
        raise ValueError(f"AFD_ROUTING_UNSUPPORTED: {routing}")
    from .sglang.adapter import SGLangAdapter

    return SGLangAdapter(source_root, mode)
