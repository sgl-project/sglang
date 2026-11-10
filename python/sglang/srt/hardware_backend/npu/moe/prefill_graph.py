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
"""Capture marker for ``ascend_ops`` prefill-graph MoE dispatch.

A captured prefill segment runs every MoE layer at the bucket width and no
Python executes on replay, so the dispatcher cannot read the allgather row
counts off live tensors.  While this marker is bound to the batch, the
dispatcher instead takes the fixed-width allgather path, which needs no host
count exchange.

Unlike the DSV4 attention metadata this state holds no tensors, so replay needs
no refresh pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_ascend_ops_config,
)
from sglang.srt.model_executor.runner_utils.forward_batch import (
    try_get_forward_batch,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MoePrefillGraphState(msgspec.Struct, frozen=True):
    """Marker: the enclosing forward is being captured into a prefill graph.

    The dispatch width is the layer's own input height, not a value carried
    here: capture makes that height identical on every rank, and it already
    accounts for any DP gather ahead of the MoE.
    """


def install_capture_state(forward_batch: ForwardBatch) -> None:
    """Bind the capture marker; no-op for backends without a graph dispatch."""
    if not get_moe_a2a_backend().is_ascend_ops():
        return
    if get_moe_ascend_ops_config().prefill != "allgather":
        return
    forward_batch.moe_prefill_graph_state = MoePrefillGraphState()


def get_moe_prefill_graph_state() -> Optional[MoePrefillGraphState]:
    """State of the forward being captured, or None outside a prefill graph."""
    batch = try_get_forward_batch()
    return None if batch is None else batch.moe_prefill_graph_state
