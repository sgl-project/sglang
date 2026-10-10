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
"""Current execution batch for eager regions in prefill graphs.

The prefill runner owns this scope. Eager callbacks resolve the batch when
executed instead of retaining capture-time request metadata. This state is
independent of ForwardContext, which selects the attention backend.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_current_batch: ContextVar["ForwardBatch | None"] = ContextVar(
    "forward_batch", default=None
)


def get_forward_batch() -> "ForwardBatch":
    batch = _current_batch.get()
    if batch is None:
        raise RuntimeError("No forward batch is set for this execution")
    return batch


@contextmanager
def set_forward_batch(forward_batch: "ForwardBatch") -> Iterator[None]:
    """Publish the current batch and restore the previous one on scope exit."""
    reset_token = _current_batch.set(forward_batch)
    try:
        yield
    finally:
        _current_batch.reset(reset_token)
