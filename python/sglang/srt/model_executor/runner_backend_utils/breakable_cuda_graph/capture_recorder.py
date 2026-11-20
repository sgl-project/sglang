# Copyright 2023-2026 SGLang Team
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
"""Device-neutral recorder contract used by ``eager_on_graph``."""

from typing import Any, Callable, Protocol


class BreakableCaptureRecorder(Protocol):
    """Record eager calls between independently captured graph segments."""

    def end_current_segment(self) -> None:
        """Finish the segment captured before the eager call."""
        ...

    def begin_next_segment(self) -> None:
        """Begin the segment captured after the eager call."""
        ...

    def register_eager_call(self, callback: Callable[[], Any]) -> None:
        """Append a callback to run between the adjacent segments on replay."""
        ...

    def retain_for_replay(self, value: Any) -> Any:
        """Return a replay-safe representation of a captured Python value."""
        ...
