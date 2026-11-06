"""Resolve the canonical SDK, retaining the historical installation fallback."""

from importlib import import_module


def load_sdk():
    """Do not hide a broken canonical install behind a different distribution."""
    try:
        package = import_module("simopus")
    except ModuleNotFoundError as error:
        if error.name != "simopus":
            raise
        package = import_module("infercast")
        return package, import_module("infercast.sdk")
    return package, import_module("simopus.sdk")
