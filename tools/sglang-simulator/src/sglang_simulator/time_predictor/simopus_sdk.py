"""Load the required SimOpus SDK without alternate installations."""

from importlib import import_module


def load_sdk():
    """Missing or broken SDK installations fail at their original cause."""
    package = import_module("simopus")
    return package, import_module("simopus.sdk")
