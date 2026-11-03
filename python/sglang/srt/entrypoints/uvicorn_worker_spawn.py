import contextlib
import os
import sys
from multiprocessing import spawn


def _launched_by_sglang_entrypoint() -> bool:
    main = sys.modules["__main__"]
    spec = getattr(main, "__spec__", None)
    if spec is not None:
        return spec.name == "sglang.launch_server"
    return os.path.basename(getattr(main, "__file__", None) or "") == "sglang"


@contextlib.contextmanager
def skip_main_reimport_in_spawned_workers():
    """Spawned children do not re-run sglang's launcher module as ``__main__``.

    The spawn start method re-runs the parent's main module in every child
    before it unpickles the target. For ``python -m sglang.launch_server`` that
    imports torch, transformers and most of sglang before uvicorn's health-check
    thread exists in the worker. On a loaded host this exceeds
    ``SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT``, so the supervisor kills the
    worker and spawns a new one, which hits the same timeout.

    A uvicorn worker unpickles only uvicorn objects and imports the app from an
    import string, so it needs nothing from sglang's launcher module. A user's
    own launcher script still runs in every worker, since it may register
    models or patches at module level.
    """
    if not _launched_by_sglang_entrypoint():
        yield
        return

    get_preparation_data = spawn.get_preparation_data

    def get_preparation_data_without_main(name):
        data = get_preparation_data(name)
        data.pop("init_main_from_name", None)
        data.pop("init_main_from_path", None)
        return data

    spawn.get_preparation_data = get_preparation_data_without_main
    try:
        yield
    finally:
        spawn.get_preparation_data = get_preparation_data
