# SPDX-License-Identifier: Apache-2.0

"""Import-safe specifications and entry points for diffusion child processes.

``multiprocessing`` unpickles a target's arguments before calling the target.
Runtime objects therefore cannot cross this boundary directly: merely passing
``ServerArgs`` used to import the diffusion configuration graph before worker
bootstrap began. ``ServerArgsPayload`` keeps that object graph opaque until the
child reaches the explicit runtime-activation phase.

``spawn`` re-executes the launching script's module scope earlier still, before
any argument is unpickled, so an offline script may bind only the
``DiffGenerator`` proxy and ``_PRE_ACTIVATION_MODULES`` at module scope; every
other diffusion import belongs inside its ``if __name__ == "__main__":`` guard.
A violation is reported rather than silently tolerated.

Keep module scope limited to the standard library and import-neutral types.
"""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

_DIFFUSION_PREFIX = "sglang.multimodal_gen."
_RUNTIME_NAMESPACES = (
    "sglang.multimodal_gen.runtime",
    "sglang.multimodal_gen.runtime.managers",
)
# What may legitimately be imported this early: bootstrap's own imports, and the
# platform and plugin modules that every plugin loads and the contract keeps
# import-safe. Listing these rather than their complement keeps the check
# complete as subpackages are added.
_PRE_ACTIVATION_MODULES = (
    "sglang.multimodal_gen.envs",
    "sglang.multimodal_gen.runtime.platforms",
    "sglang.multimodal_gen.runtime.utils",
    "sglang.multimodal_gen.runtime.managers.worker_bootstrap",
)
_MAX_REPORTED_MODULES = 5


def _warn_if_runtime_imported_early() -> None:
    """Name the modules that this child imported ahead of its own lifecycle."""
    early = sorted(
        name
        for name in list(sys.modules)
        if name.startswith(_DIFFUSION_PREFIX)
        and name not in _RUNTIME_NAMESPACES
        and not name.startswith(_PRE_ACTIVATION_MODULES)
    )
    if not early:
        return

    listed = ", ".join(early[:_MAX_REPORTED_MODULES])
    if len(early) > _MAX_REPORTED_MODULES:
        listed += f" (+{len(early) - _MAX_REPORTED_MODULES} more)"
    # In a spawned child __main__ is the re-executed launching script, which is
    # the file whose imports have to move.
    script = vars(sys.modules["__main__"]).get("__file__", "the launching script")
    logging.getLogger(__name__).warning(
        "Diffusion runtime modules were imported before this worker initialized "
        "its platform: %s. spawn re-executes %s at module scope in every child, "
        "so these were built ahead of platform initialization and hook "
        "application, and the classes and registrations they created are "
        'already past reach. Move the import inside if __name__ == "__main__": '
        "or into the function that uses it.",
        listed,
        script,
    )


@dataclass(frozen=True, slots=True)
class ServerArgsPayload:
    """A deferred ``ServerArgs`` snapshot safe to unpickle before bootstrap."""

    _pickle: bytes = field(repr=False)

    @classmethod
    def capture(cls, server_args: ServerArgs) -> ServerArgsPayload:
        return cls(pickle.dumps(server_args, protocol=pickle.HIGHEST_PROTOCOL))

    def materialize(self) -> ServerArgs:
        # Importing ServerArgs pulls in pipeline configuration modules. This
        # method must only be called after the process lifecycle is initialized.
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        server_args = pickle.loads(self._pickle)
        if not isinstance(server_args, ServerArgs):
            raise TypeError("Bootstrap payload did not contain diffusion ServerArgs")
        return server_args


@dataclass(frozen=True, slots=True)
class SchedulerProcessSpec:
    """Everything a scheduler child needs, without eagerly importing runtime state."""

    local_rank: int
    rank: int
    server_args: ServerArgsPayload
    pipe_writer: Connection


def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:
    """Initialize a child in dependency order, then invoke its worker."""
    # Arm PDEATHSIG before any vendor code runs: everything below can block,
    # and a child that hangs there would outlive a dead launcher.
    from sglang.multimodal_gen.runtime.utils.process import (
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()

    # Every rank re-executes the same script, so one rank reporting is enough.
    if spec.rank == 0:
        _warn_if_runtime_imported_early()

    # Platform initialization is the first extensible runtime action. In
    # particular it precedes plugin callbacks and hook target resolution, both
    # of which may import arbitrary runtime modules.
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform

    initialize_current_platform()

    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()

    server_args = spec.server_args.materialize()

    # Resolve the function from its module after hook application. A ``from``
    # binding created earlier would retain the unpatched callable.
    from sglang.multimodal_gen.runtime.managers import gpu_worker

    gpu_worker.run_scheduler_process(
        local_rank=spec.local_rank,
        rank=spec.rank,
        server_args=server_args,
        pipe_writer=spec.pipe_writer,
    )


def bootstrap_http_server_process(server_args: ServerArgsPayload) -> None:
    from sglang.multimodal_gen.runtime.utils.process import (
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()

    _warn_if_runtime_imported_early()

    # No initialize_current_platform() here: this child serves HTTP and never
    # touches the device, so it has no reason to bring up a vendor backend.
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()

    from sglang.multimodal_gen.runtime import launch_server

    launch_server.launch_http_server_only(server_args.materialize())
