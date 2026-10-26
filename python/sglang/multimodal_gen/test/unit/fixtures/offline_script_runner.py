# SPDX-License-Identifier: Apache-2.0

"""Shared body of the offline-script fixtures in ``test_worker_bootstrap``.

Held apart so the scripts differ only in the module-scope import each is named
for; nothing here may import what the child's bootstrap has to precede.
"""

import json
import multiprocessing as mp
import sys

from sglang.multimodal_gen.runtime.managers import worker_bootstrap

_CHILD_REPLY_TIMEOUT_S = 120
_CHILD_JOIN_TIMEOUT_S = 10


def run_offline_script(result_path: str) -> None:
    """Spawn one scheduler child and write back what it observed."""
    # Inside the guarded call on purpose: this is the import whose absence from
    # the child's module scope the test is measuring.
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    reader, writer = mp.Pipe(duplex=False)
    spec = worker_bootstrap.SchedulerProcessSpec(
        local_rank=0,
        rank=0,
        server_args=worker_bootstrap.ServerArgsPayload.capture(
            ServerArgs.__new__(ServerArgs)
        ),
        pipe_writer=writer,
    )

    process = mp.get_context("spawn").Process(
        target=worker_bootstrap.bootstrap_scheduler_process,
        args=(spec,),
    )
    process.start()
    writer.close()

    observed = None
    if reader.poll(_CHILD_REPLY_TIMEOUT_S):
        try:
            observed = reader.recv()
        except EOFError:
            pass
    process.join(_CHILD_JOIN_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        process.join(_CHILD_JOIN_TIMEOUT_S)

    with open(result_path, "w") as result_file:
        json.dump({"observed": observed, "exitcode": process.exitcode}, result_file)


def main() -> None:
    run_offline_script(sys.argv[1])
