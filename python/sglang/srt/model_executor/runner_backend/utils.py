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
"""runner_backend utilities — phase → BaseCudaGraphBackend resolution.

Centralizes per-phase backend resolution so platform overrides (NPU,
out-of-tree) and future backend additions can plug in without
modifying the runner files. Phase / backend identifiers used here
live in :mod:`.cuda_graph_config`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    TcPiecewiseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )

logger = logging.getLogger(__name__)

# Track first occurrence of each fallback warning to avoid log spam.
_TC_PIECEWISE_DECODE_FALLBACK_LOGGED = False


def resolve_decode_backend(
    cuda_graph_runner: BaseCudaGraphRunner,
) -> BaseCudaGraphBackend:
    """Pick a backend instance from cuda_graph_config['decode']['backend'].

    NPU device returns NPUCudaGraphBackend regardless of mode (only
    the Full-style backend is wired for NPU today).
    """
    model_runner = cuda_graph_runner.model_runner
    cfg = model_runner.server_args.cuda_graph_config
    backend_name = cfg.decode.backend if cfg is not None else Backend.FULL

    enable_memory_saver = model_runner.server_args.enable_memory_saver

    if model_runner.device == "npu":
        from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
            NPUCudaGraphBackend,
        )

        return NPUCudaGraphBackend(
            cuda_graph_runner, enable_memory_saver=enable_memory_saver
        )
    elif model_runner.device == "xpu":
        from sglang.srt.hardware_backend.xpu.xpu_cudagraph_backend import (
            XPUCudaGraphBackend,
        )

        return XPUCudaGraphBackend(cuda_graph_runner)

    if model_runner.device == "xpu":
        if backend_name not in (Backend.FULL, Backend.DISABLED):
            raise ValueError(
                f"XPU only supports cuda_graph_config decode backend 'full', got '{backend_name}'"
            )
        from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
            FullXPUGraphBackend,
        )

        return FullXPUGraphBackend(cuda_graph_runner)

    if backend_name == Backend.BREAKABLE:
        return BreakableCudaGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
            debug_eager=model_runner.server_args.debug_cuda_graph,
        )
    if backend_name == Backend.TC_PIECEWISE:
        global _TC_PIECEWISE_DECODE_FALLBACK_LOGGED
        if not _TC_PIECEWISE_DECODE_FALLBACK_LOGGED:
            logger.warning(
                "cuda_graph_config decode='tc_piecewise' is not yet implemented; "
                "falling back to 'full'."
            )
            _TC_PIECEWISE_DECODE_FALLBACK_LOGGED = True
    return FullCudaGraphBackend(
        cuda_graph_runner, enable_memory_saver=enable_memory_saver
    )


def resolve_prefill_backend(
    cuda_graph_runner: BaseCudaGraphRunner,
) -> BaseCudaGraphBackend:
    """Pick a backend instance from cuda_graph_config['prefill']['backend']."""
    model_runner = cuda_graph_runner.model_runner
    cfg = model_runner.server_args.cuda_graph_config
    backend_name = cfg.prefill.backend if cfg is not None else Backend.TC_PIECEWISE

    if backend_name == Backend.BREAKABLE:
        return BreakableCudaGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=model_runner.server_args.enable_memory_saver,
            debug_eager=model_runner.server_args.debug_cuda_graph,
        )
    # Default: tc_piecewise. (prefill, full) is rejected at config validation.
    return TcPiecewiseCudaGraphBackend(cuda_graph_runner)
