try:
    from sglang._version import __version__, __version_tuple__
except ImportError:
    # Fallback for development without build
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")
