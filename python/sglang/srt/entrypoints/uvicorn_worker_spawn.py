import contextlib
from multiprocessing import spawn


@contextlib.contextmanager
def skip_main_reimport_in_spawned_workers():
    """Spawned children do not re-run the parent's ``__main__`` module.

    The spawn start method re-runs the parent's main module in every child
    before it unpickles the target. For ``python -m sglang.launch_server`` that
    imports torch, transformers and most of sglang before uvicorn's health-check
    thread exists in the worker. On a loaded host this exceeds
    ``SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT``, so the supervisor kills the
    worker and spawns a new one, which hits the same timeout.

    A uvicorn worker unpickles only uvicorn objects and imports the app from an
    import string, so it needs nothing from ``__main__``.
    """
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
