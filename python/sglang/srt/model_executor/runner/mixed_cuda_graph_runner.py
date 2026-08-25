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
"""MixedCudaGraphRunner — the extend-family runner when mixed chunking is on.

Constructed instead of PrefillCudaGraphRunner under --enable-mixed-chunk, so
one instance serves both pure prefill and MIXED batches. In a mixed batch,
each decode request is viewed as a 1-token extend with its context as prefix.

Mixed-only admission added here must be mirrored in the schedule-time dp
vote (managers/scheduler_components/dp_attn.py), or dp ranks can disagree
on replay-vs-eager and desynchronize collectives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MixedCudaGraphRunner(PrefillCudaGraphRunner):
    """Extend-family runner serving pure prefill and MIXED batches."""

    _logged_mixed_replay = False

    def __init__(self, model_runner: ModelRunner) -> None:
        prefill_backend = get_exec().graph.cuda_graph_config.prefill.backend
        assert prefill_backend == Backend.BREAKABLE, (
            "Mixed chunk prefill supports only the breakable prefill CUDA "
            f"graph backend; got '{prefill_backend}'. tc_piecewise and full "
            "cannot serve mixed batches yet (full's fixed request-slot "
            "geometry does not cover mixed decode tails; may be supported "
            "later). Use --cuda-graph-backend-prefill breakable, or disable "
            "the prefill CUDA graph or --enable-mixed-chunk."
        )
        super().__init__(model_runner)

    def _graph_replay_forward_mode(
        self, mode: Optional[ForwardMode]
    ) -> Optional[ForwardMode]:
        """MIXED replays the EXTEND-captured graphs; other modes pass through."""
        return ForwardMode.EXTEND if mode == ForwardMode.MIXED else mode

    def load_batch(self, forward_batch: ForwardBatch, **kwargs) -> ForwardBatch:
        if forward_batch.forward_mode.is_mixed() and not self._logged_mixed_replay:
            self._logged_mixed_replay = True
            logger.info(
                "Mixed CUDA graph replay engaged (bs=%d, num_tokens=%d).",
                forward_batch.batch_size,
                len(forward_batch.input_ids),
            )
        return super().load_batch(forward_batch, **kwargs)
