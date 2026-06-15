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
"""BaseRunner — the surface shared by every phase runner.

Two kinds of runner subclass this:

  - ``BaseCudaGraphRunner`` (and its ``Decode``/``PrefillCudaGraphRunner``
    subclasses) — capture a ``torch.cuda.CUDAGraph`` per shape and replay it.
  - ``EagerRunner`` — no capture; runs ``model.forward`` live each iteration
    over one static batch.

``BaseRunner`` holds only what both share: the run-once kernel ``warmup()``
(flashinfer autotune + PP-DeepGEMM warmup), the autotune-buffer hook, and the
abstract per-iteration entry points (``can_run_graph`` dispatch gate,
``load_batch`` copy-into-static, ``execute`` run). All capture/shape machinery
(``prepare`` / ``reserve_batch`` / ``_prepare_one`` / bucket padding / the
graph-only ``ExecutionBackend``) lives on ``BaseCudaGraphRunner``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """Abstract base shared by the cuda-graph runners and the eager runner.

    Methods:
      - can_run_graph(forward_batch) — should forward_batch go through cuda
        graph replay (vs eager fallback)? Dispatch gate; the eager runner
        always returns True.
      - warmup() — one-time kernel warmup / flashinfer autotune, run once
        across the decode + prefill runners.
      - load_batch(forward_batch, ...) — copy the live fb into the runner's
        static buffers and refresh dynamic attention metadata.
      - execute(forward_batch, ...) — run one batch (graph replay for the
        cuda-graph runners; model.forward for eager) and slice to raw size.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tbo_plugin = TboCudaGraphRunnerPlugin()

    def warmup(self) -> None:
        """Warm up + autotune kernels once, before this runner captures (graph)
        or reserves (eager) — part of the Runner lifecycle, called from
        prepare().

        Run-once across the decode and prefill runners via a flag on the shared
        ModelRunner: whichever runner prepares first does the warmup; the other
        is a no-op. (Replaces the unconditional ModelRunner.kernel_warmup call.)
        """
        model_runner = self.model_runner
        if getattr(model_runner, "_kernel_warmed_up", False):
            return
        model_runner._kernel_warmed_up = True

        if model_runner.device != "cuda":
            return

        if model_runner._should_run_flashinfer_autotune():
            # Autotune always reuses a prepared static decode buffer instead of
            # allocating a throwaway set. The decode runner provides it; a
            # prefill-only (embedding) runner does not -- its buffers lack the
            # decode fields the dummy forward reads -- so we assert rather than
            # silently fall back to a fresh allocation.
            buffers, batch_size = self._autotune_buffers()
            assert buffers is not None, (
                "flashinfer autotune requires the decode runner's prepared "
                "static buffers; none available (prefill-only / embedding model)"
            )
            model_runner._flashinfer_autotune(buffers=buffers, batch_size=batch_size)

        from sglang.srt.environ import envs
        from sglang.srt.layers import deep_gemm_wrapper

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and self.pp_size > 1
            and not model_runner.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(model_runner)

    def _autotune_buffers(self) -> Tuple[Optional[Any], Optional[int]]:
        """Static decode buffers + max captured bs for warmup() to hand the
        flashinfer-autotune dummy forward, so it reuses this runner's already-
        allocated buffers instead of allocating a throwaway set.

        Returns (None, None) by default. The decode runner overrides this: its
        buffers carry every field the dummy decode forward reads. Prefill
        buffers deliberately do not (no seq_lens / req_pool_indices / logits
        buffer), so warmup() asserts a decode buffer is present before autotune.
        """
        return None, None

    @abstractmethod
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool: ...

    @abstractmethod
    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def execute(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...
