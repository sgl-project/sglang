import contextlib
import os
import sys
from multiprocessing import spawn


def _launched_by_sglang_entrypoint() -> bool:
    main_globals = vars(sys.modules["__main__"])
    spec = main_globals.get("__spec__")
    if spec is not None:
        return spec.name == "sglang.launch_server"
    return os.path.basename(main_globals.get("__file__") or "") == "sglang"


@contextlib.contextmanager
def skip_main_reimport_in_spawned_workers():
    """Do not re-run sglang's launcher as __main__ in spawned uvicorn workers.

    The re-import (torch, transformers) can outlast the worker health-check
    timeout. Workers import the app by string, so they need nothing from it.
    A user's own launcher script still runs in every worker.
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
