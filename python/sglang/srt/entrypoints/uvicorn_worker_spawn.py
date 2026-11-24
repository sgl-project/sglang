import contextlib
import os
import sys
from multiprocessing import spawn


def _launched_by_sglang_entrypoint() -> bool:
    main = sys.modules["__main__"]
    if main.__spec__ is not None:
        return main.__spec__.name == "sglang.launch_server"
    # __file__ is unset in an interactive session.
    return os.path.basename(vars(main).get("__file__") or "") == "sglang"


@contextlib.contextmanager
def skip_main_reimport_in_spawned_workers():
    """Do not re-run sglang's launcher as __main__ in spawned uvicorn workers.

    The re-import (torch, transformers) can outlast the worker health-check
    timeout. Workers import the app by string, so they need nothing from it.
    A user's own launcher script still runs in every worker.

    While active, the patch covers every spawn child of this process, not only
    uvicorn's workers. That is safe: sglang's launcher main modules define no
    process targets, so no child needs them.
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
