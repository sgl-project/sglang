from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_kernel_backend,
    initialize_linear_attn_config,
)

__all__ = [
    "GDNAttnBackend",
    "KDAAttnBackend",
    "KimiLinearAttnBackend",
    "LinearAttnKernelBackend",
    "get_linear_attn_kernel_backend",
    "initialize_linear_attn_config",
]

# Lazy imports to avoid circular dependency with hybrid_linear_attn_backend
_LAZY_IMPORTS = {
    "GDNAttnBackend": "sglang.srt.layers.attention.linear.gdn_backend",
    "KDAAttnBackend": "sglang.srt.layers.attention.linear.kda_backend",
    "KimiLinearAttnBackend": "sglang.srt.layers.attention.linear.kda_backend",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
