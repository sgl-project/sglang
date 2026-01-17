try:
    from sglang._version import __version__, __version_tuple__
except ImportError:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("sglang")
        __version_tuple__ = tuple(__version__.split("."))
    except Exception:
        try:
            import subprocess

            __version__ = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--always"], stderr=subprocess.STDOUT
                )
                .decode("utf-8")
                .strip()
            )
            if __version__.startswith("v"):
                __version__ = __version__[1:]
            __version_tuple__ = tuple(__version__.split("."))
        except Exception:
            # Fallback for development without build
            __version__ = "0.0.0.dev0"
            __version_tuple__ = (0, 0, 0, "dev0")
