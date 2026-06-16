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
"""Breakable CUDA graph (BCG) runner for diffusion DiT transformers.

Captures a DiT ``transformer.forward`` as a sequence of
``torch.cuda.CUDAGraph`` segments split at the attention modules (see
``layers/attention/layer.py``), so the linear/norm/FFN math of each block runs
from a static CUDA graph while sequence-parallel all-to-all, varlen packing,
and dynamic/sparse attention kernels run eagerly between segments.

The model-agnostic capture/replay engine lives in
:class:`sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.runner.BaseBreakableCudaGraphRunner`,
shared with the LLM runtime BCG primitives. This subclass adds only the
diffusion-specific docstring/contract:

Within a single generate request the DiT input shapes are fixed across all
denoising steps, so capture is keyed by the tensor-input signature and replayed
for every subsequent step. Every tensor input — including tensors nested inside
list/tuple/dict kwargs such as Wan's ``encoder_hidden_states`` prompt-embed list
— is copied into a persistent static buffer before each replay, so per-step
latents/timestep AND per-CFG-branch conditioning are refreshed correctly. The
attention break points re-run eagerly and re-read the live forward context, so
per-timestep attention metadata (e.g. sparse-video-attention masks) is also
picked up correctly on replay.

This runner shares the model-agnostic BCG primitives in
:mod:`sglang.srt.breakable_cuda_graph` with the LLM runtime.
Capture is driven explicitly at warmup (see the denoising stage), so serving
only replays and never records a fresh graph.
"""

from __future__ import annotations

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
    _CaptureEntry,
    _CaptureRejected,
    _clone_output,
    _flatten_kwargs,
    _map_tensors,
    _signature_kwargs,
    _signature_summary,
)

__all__ = [
    "DiffusionBreakableCudaGraphRunner",
    "_CaptureEntry",
    "_signature_kwargs",
]


class DiffusionBreakableCudaGraphRunner(BaseBreakableCudaGraphRunner):
    """Capture/replay a diffusion DiT ``transformer`` with breakable CUDA graphs.

    Usage::

        runner = DiffusionBreakableCudaGraphRunner(transformer, device)
        runner.capture(hidden_states=..., timestep=..., ...)  # at warmup
        noise_pred = runner(hidden_states=..., timestep=..., ...)  # serving

    Inherits the full capture/replay API from
    :class:`BaseBreakableCudaGraphRunner`; calling the runner replays a captured
    graph for the input signature, or runs the transformer eagerly when none was
    captured (it never captures while serving). Unknown attributes proxy to the
    wrapped transformer.
    """

    def _should_capture_on_call(self, key) -> bool:
        """Allow lazy capture only inside the warmup window.

        The denoising stages run the runner inside ``set_forward_context`` with
        the current request, so a call during a warmup request may capture
        (this is how warmup records graphs by simply driving the forward); a
        call while serving never does, guaranteeing no fresh capture after
        startup.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        try:
            forward_batch = get_forward_context().forward_batch
        except Exception:
            return False
        return bool(getattr(forward_batch, "is_warmup", False))
