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
        """Diffusion DiTs capture lazily on first use of each signature.

        Across denoising stages the runner is reached through several call
        paths (the base stage, and model-specific stages that enter
        ``set_forward_context`` without a ``forward_batch``), so gating capture
        on a per-call warmup signal is unreliable and would silently disable
        BCG for those stages. Instead, capture-on-first-use is always allowed,
        and the *no-recapture-while-serving* property is delivered by warmup:
        warmup runs the model's full recommended steps and (for the base stage)
        every ``--bcg-text-buckets`` bucket, so by the time real requests
        arrive every signature is already captured and serving only replays.
        """
        return True
