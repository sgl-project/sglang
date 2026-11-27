__all__ = ["HostPoolOptions"]


def __getattr__(name):
    # Resolved on first use: importing a leaf module of this package (such as
    # `kv.decl`, which the layout modules read) must not import the layouts
    # back while they are still initializing.
    if name == "HostPoolOptions":
        from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions

        return HostPoolOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
