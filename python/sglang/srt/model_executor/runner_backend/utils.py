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
"""runner_backend utilities — phase → BaseGraphBackend resolution.

Centralizes per-phase backend resolution so platform overrides (NPU,
out-of-tree) and future backend additions can plug in without
modifying the runner files. Phase / backend identifiers used here
live in :mod:`.cuda_graph_config`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend.base_graph_backend import (
    BaseGraphBackend,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_exec, get_flags, get_memory

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )

logger = logging.getLogger(__name__)


def _group_preparation_allowed(lora_manager, *, profile_enabled: bool) -> bool:
    """Keep per-shape warmups for shape-dependent LoRA scratch."""
    return (
        current_platform.is_cuda()
        and (
            lora_manager is None
            or (
                lora_manager.lora_backend.lora_workspace is None
                and not lora_manager.lora_backend.is_moe_lora
            )
        )
        and not get_exec().graph.debug_cuda_graph
        and not profile_enabled
        and not get_flags().capture.enable_torch_compile
        and not envs.SGLANG_ENABLE_GRAPH_POOL_PRECARVE.get()
        and not envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get()
    )


def _needs_hicache_event_capture(model_runner: ModelRunner) -> bool:
    """Captured KDA state reads need graph-managed HiCache dependencies."""
    # KDA captures linear extend inside BCG, so the pool accessor's Python
    # layer wait runs only during capture. Replay needs explicit event waits.
    # Other linear backends break to eager extend and rerun the pool waits.
    # Captured metadata in unrelated backends (e.g. DeepSeek V4) does not mean
    # they capture KDA state reads and need this mechanism.
    backend = model_runner.attn_backend
    if not (
        get_memory().enable_hierarchical_cache
        and backend.use_captured_forward_metadata_for_breakable_cuda_graph
    ):
        return False

    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

    return isinstance(getattr(backend, "linear_attn_backend", None), KDAAttnBackend)


def resolve_decode_backend(
    cuda_graph_runner: BaseCudaGraphRunner,
) -> BaseGraphBackend:
    """Pick a backend instance from cuda_graph_config['decode']['backend'].

    NPU device returns NPUCudaGraphBackend regardless of mode (only
    the Full-style backend is wired for NPU today).
    """
    model_runner = cuda_graph_runner.model_runner
    cfg = get_exec().graph.cuda_graph_config
    backend_name = cfg.decode.backend if cfg is not None else Backend.FULL

    enable_memory_saver = get_exec().features.enable_memory_saver

    if model_runner.device == "npu":
        from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
            NPUCudaGraphBackend,
        )

        return NPUCudaGraphBackend(
            cuda_graph_runner, enable_memory_saver=enable_memory_saver
        )
    elif model_runner.device == "xpu":
        if backend_name == Backend.BREAKABLE:
            return BreakableCudaGraphBackend(
                cuda_graph_runner,
                enable_memory_saver=enable_memory_saver,
                debug_eager=get_exec().graph.debug_cuda_graph,
            )
        if backend_name not in (Backend.FULL, Backend.DISABLED):
            raise ValueError(
                f"XPU only supports cuda_graph_config decode backends 'full' and 'breakable', got '{backend_name}'"
            )
        from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
            FullXPUGraphBackend,
        )

        return FullXPUGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
        )

    if backend_name == Backend.BREAKABLE:
        return BreakableCudaGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
            debug_eager=get_exec().graph.debug_cuda_graph,
        )

    # Callers decide whether to construct a graph runner. Explicit construction
    # with a disabled config (e.g. the draft-runner test harness) historically
    # uses the full backend; retiring TCPCG must preserve that behavior.
    if backend_name not in (Backend.FULL, Backend.DISABLED):
        raise ValueError(f"Unsupported decode graph backend: {backend_name}")

    full_backend_cls = None
    if current_platform.is_out_of_tree():
        full_backend_cls = current_platform.get_full_graph_backend_cls()

    if full_backend_cls is None:
        full_backend_cls = FullCudaGraphBackend

    backend = full_backend_cls(
        cuda_graph_runner,
        enable_memory_saver=enable_memory_saver,
    )
    if (
        full_backend_cls is FullCudaGraphBackend
        and _group_preparation_allowed(
            model_runner.lora_manager,
            profile_enabled=cuda_graph_runner.enable_profile_cuda_graph,
        )
        and not cuda_graph_runner.attn_backend.requires_per_shape_decode_graph_warmup
    ):
        backend._group_preparation_enabled = True
    return backend


def resolve_prefill_backend(
    cuda_graph_runner: BaseCudaGraphRunner,
) -> BaseGraphBackend:
    """Pick a backend instance from cuda_graph_config['prefill']['backend']."""
    model_runner = cuda_graph_runner.model_runner
    cfg = get_exec().graph.cuda_graph_config
    backend_name = cfg.prefill.backend if cfg is not None else Backend.BREAKABLE

    if model_runner.device == "xpu" and backend_name == Backend.FULL:
        from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
            FullXPUGraphBackend,
        )

        return FullXPUGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            reuse_output_buffer=True,
        )

    if backend_name == Backend.BREAKABLE:
        backend = BreakableCudaGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            debug_eager=get_exec().graph.debug_cuda_graph,
            enable_hicache_event_capture=_needs_hicache_event_capture(
                cuda_graph_runner.model_runner
            ),
        )
        if _group_preparation_allowed(
            model_runner.lora_manager,
            profile_enabled=get_exec().graph.enable_profile_cuda_graph,
        ):
            backend._group_preparation_enabled = True
        return backend
    if backend_name == Backend.FULL:
        return FullCudaGraphBackend(
            cuda_graph_runner,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            reuse_output_buffer=True,
        )
    raise ValueError(f"Unsupported prefill graph backend: {backend_name}")
