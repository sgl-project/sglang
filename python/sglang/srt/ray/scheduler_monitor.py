# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Detects a Ray scheduler actor that died or whose event loop exited."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import ray

# Bounds how long a stopped monitor keeps its thread alive; arbitrary.
_POLL_INTERVAL_S = 5.0


class SchedulerActorMonitor:
    """Calls ``on_exit`` once when the first scheduler event loop ends.

    Each ref is a scheduler actor's ``run_event_loop`` result, which resolves
    only when the loop returns or the actor dies.
    """

    def __init__(
        self,
        *,
        event_loop_refs: List[ray.ObjectRef],
        on_exit: Callable[[RuntimeError], None],
    ):
        self._event_loop_refs = event_loop_refs
        self._on_exit = on_exit
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ray_scheduler_monitor"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop watching; an exit caused by shutdown is not reported."""
        self._stopped.set()

    def _run(self) -> None:
        error = self._wait_for_exit()
        if error is not None and not self._stopped.is_set():
            self._on_exit(error)

    def _wait_for_exit(self) -> Optional[RuntimeError]:
        # Poll so stop() and Ray teardown are not stuck behind a blocking wait.
        while not self._stopped.is_set() and ray.is_initialized():
            done, _ = ray.wait(
                self._event_loop_refs, num_returns=1, timeout=_POLL_INTERVAL_S
            )
            if done:
                return _exit_error(done[0])
        return None


def _exit_error(event_loop_ref: ray.ObjectRef) -> RuntimeError:
    try:
        ray.get(event_loop_ref)
    except Exception as e:
        return RuntimeError(f"SGLang scheduler actor died: {e}")
    return RuntimeError("SGLang scheduler event loop exited")
