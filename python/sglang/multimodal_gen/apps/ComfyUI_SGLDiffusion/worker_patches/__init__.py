# SPDX-License-Identifier: Apache-2.0
"""ComfyUI worker patches. Keep module scope import-safe for multiprocessing spawn."""

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial, wraps
from importlib import import_module

# Version is part of the name (and therefore the generator's reuse key).
PROFILES = {"h3_comfyui_v1": "h3"}
_STARTUP_LOCK = threading.RLock()


def canonical_profiles(names):
    if names is None:
        return ()
    if not isinstance(names, (tuple, list)) or any(
        not isinstance(name, str) or name not in PROFILES for name in names
    ):
        raise ValueError("Unknown worker patch; choose a registered profile")
    names = tuple(dict.fromkeys(names))
    if len(names) > 1:
        raise ValueError("Select one worker patch; composition is not supported")
    return names


def validate_options(names, model_type, options):
    for name in canonical_profiles(names):
        import_module(f"{__name__}.{PROFILES[name]}").validate(model_type, options)


@dataclass(frozen=True)
class _PatchedArgs:
    original: object
    name: str

    def materialize(self):
        # Stock bootstrap calls this after platform and plugin initialization.
        args = self.original.materialize()
        from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

        install = import_module(f"{__name__}.{PROFILES[self.name]}").install
        original_init = GPUWorker.__init__

        @wraps(original_init)
        def init(worker, *a, **kw):
            original_init(worker, *a, **kw)
            # Synchronous: missing pipeline/failed installation aborts startup.
            install(worker.pipeline)
            logging.getLogger(__name__).info("Worker patch applied: %s", self.name)

        GPUWorker.__init__ = init
        return args


def _bootstrap(target, name, spec):
    return target(replace(spec, server_args=_PatchedArgs(spec.server_args, name)))


@contextmanager
def scheduler_profiles(names):
    names = canonical_profiles(names)
    from sglang.multimodal_gen.runtime import launch_server

    # Serialize patched and default launches; restore the parent even on failure.
    with _STARTUP_LOCK:
        original = launch_server.bootstrap_scheduler_process
        if names:
            launch_server.bootstrap_scheduler_process = partial(
                _bootstrap, original, names[0]
            )
        try:
            yield
        finally:
            launch_server.bootstrap_scheduler_process = original
