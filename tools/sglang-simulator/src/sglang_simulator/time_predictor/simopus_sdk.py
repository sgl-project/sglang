"""Load the required SimOpus SDK without alternate installations."""

from importlib import import_module


def load_sdk():
    """Missing or broken SDK installations fail at their original cause."""
    package = import_module("simopus")
    return package, import_module("simopus.sdk")


def open_database(sdk, system, backend_version, *, perfdb_dir, database_mode):
    """Open the SimOpus performance data of one GPU and SGLang version.

    ``PerfDB`` is the read scope of the SDK. An SDK that predates the name offers the
    same object as ``PerfDatabase.open_fidb``.
    """
    scope = getattr(sdk, "PerfDB", None)
    if scope is None:
        return sdk.PerfDatabase.open_fidb(
            system,
            "sglang",
            backend_version,
            perfdb_dir=perfdb_dir,
            database_mode=database_mode,
        )
    return scope(
        system,
        perfdb_dir=perfdb_dir,
        backend="sglang",
        version=backend_version,
        database_mode=database_mode,
    )
