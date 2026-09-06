# SPDX-License-Identifier: Apache-2.0

"""Import-safe specifications and entry points for diffusion child processes.

``multiprocessing`` unpickles a target's arguments before calling the target.
Runtime objects therefore cannot cross this boundary directly: merely passing
``ServerArgs`` used to import the diffusion configuration graph before worker
bootstrap began. ``ServerArgsPayload`` keeps that object graph opaque until the
child reaches the explicit runtime-activation phase.

Keep module scope limited to the standard library and import-neutral types.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from sglang.multimodal_gen.runtime.server_args import ServerArgs


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
    master_port: int
    server_args: ServerArgsPayload
    pipe_writer: Connection
    task_pipe_r: Connection | None
    result_pipe_w: Connection | None
    task_pipes_to_slaves: list[Connection] | Connection | None
    result_pipes_from_slaves: list[Connection] | Connection | None


def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:
    """Initialize a child in dependency order, then invoke its worker."""
    # Arm PDEATHSIG before any vendor code runs: everything below can block,
    # and a child that hangs there would outlive a dead launcher.
    from sglang.multimodal_gen.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    # Platform initialization is the first extensible runtime action. In
    # particular it precedes plugin callbacks and hook target resolution, both
    # of which may import arbitrary runtime modules.
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform

    initialize_current_platform()

    from sglang.multimodal_gen.plugins import apply_plugin_hooks, load_plugins

    load_plugins()
    apply_plugin_hooks()

    server_args = spec.server_args.materialize()

    # Resolve the function from its module after hook application. A ``from``
    # binding created earlier would retain the unpatched callable.
    from sglang.multimodal_gen.runtime.managers import gpu_worker

    gpu_worker.run_scheduler_process(
        local_rank=spec.local_rank,
        rank=spec.rank,
        master_port=spec.master_port,
        server_args=server_args,
        pipe_writer=spec.pipe_writer,
        task_pipe_r=spec.task_pipe_r,
        result_pipe_w=spec.result_pipe_w,
        task_pipes_to_slaves=spec.task_pipes_to_slaves,
        result_pipes_from_slaves=spec.result_pipes_from_slaves,
    )


def bootstrap_http_server_process(server_args: ServerArgsPayload) -> None:
    from sglang.multimodal_gen.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    # No initialize_current_platform() here: this child serves HTTP and never
    # touches the device, so it has no reason to bring up a vendor backend.
    from sglang.multimodal_gen.plugins import apply_plugin_hooks, load_plugins

    load_plugins()
    apply_plugin_hooks()

    from sglang.multimodal_gen.runtime import launch_server

    launch_server.launch_http_server_only(server_args.materialize())
