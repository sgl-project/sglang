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
"""EagerRunner — the no-cuda-graph phase runner.

The eager dual of the cuda-graph runners. Where ``DecodeCudaGraphRunner`` and
``PrefillCudaGraphRunner`` capture a ``torch.cuda.CUDAGraph`` per shape and
replay it, ``EagerRunner`` runs ``model.forward`` live each iteration over a
grow-on-demand static buffer set. It is used when CUDA graph is disabled for the
generation phases (``--disable-cuda-graph`` / a phase resolving to
``disabled``).

When CUDA graph is disabled, ``ModelRunner.decode_cuda_graph_runner`` and
``prefill_cuda_graph_runner`` point at ONE ``EagerRunner`` instance,
mode-dispatched on ``forward_batch.forward_mode`` (decode / extend / idle).

This is a real extraction — the eager path that used to live inline in
``ModelRunner.forward_decode`` / ``forward_extend`` (eager branch) /
``forward_idle`` (plus ``_eager_fb_view`` and the grow-on-demand eager input
registries) now lives here. The cuda-graph runners stay purely cuda-graph (no
``eager`` flag, no ``EagerBackend``). ``load_batch`` copies the live batch into
the eager static buffers (``_eager_fb_view``); ``execute`` inits attention
metadata and runs ``model.forward`` live, mode-dispatched.

Deferred (kept identical to the pre-extraction behavior, NOT yet optimized):
  - the v2 single-unified-buffer at ``max_num_tokens`` and prefill-first
    allocation — this runner keeps the two separate grow-on-demand decode /
    prefill registries exactly as before;
  - a first-class ``StaticForwardBatch`` — ``load_batch`` still returns the
    registry's ``extract_buffer`` view.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
    build_prefill_registry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import next_power_of_2

logger = logging.getLogger(__name__)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class _EagerBufferRegistry:
    # Lazily-built eager input-buffer registry plus the capacity it was sized to.
    registry: Optional[CudaGraphBufferRegistry] = None
    max_bs: int = 0
    max_num_tokens: int = 0


class EagerRunner(BaseRunner):
    """No-cuda-graph phase runner; mode-dispatched over decode + extend + idle.

    Public surface (the :class:`BaseRunner` ABC):
      - can_run_graph(forward_batch) -> False (always; the dispatch gate that
        keeps callers from routing an eager batch into a graph-replay branch).
      - warmup() — inherited; run-once kernel warmup + flashinfer autotune.
      - load_batch(forward_batch, ...) — copy the live batch into the eager
        static buffers (the grow-on-demand registry view).
      - execute(forward_batch, ...) — init attention metadata + run
        model.forward live, mode-dispatched (decode / extend / idle).
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        self._eager_decode_registry = _EagerBufferRegistry()
        self._eager_prefill_registry = _EagerBufferRegistry()

    # ------------------------------------------------------------------ #
    # Dispatch gate
    # ------------------------------------------------------------------ #
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        # Eager never runs a cuda graph; callers dispatch on isinstance(...,
        # EagerRunner) and must not route an eager batch into a replay branch.
        return False

    # ------------------------------------------------------------------ #
    # Eager input-buffer registry (grow-on-demand), formerly on ModelRunner
    # ------------------------------------------------------------------ #
    def _ensure_eager_registry(
        self,
        cache: _EagerBufferRegistry,
        raw_bs: int,
        raw_num_tokens: int,
        build: Callable[[int, int], CudaGraphBufferRegistry],
    ) -> CudaGraphBufferRegistry:
        # Built on first use and grown (next power of two) when a batch exceeds
        # the current capacity.
        if (
            cache.registry is not None
            and raw_bs <= cache.max_bs
            and raw_num_tokens <= cache.max_num_tokens
        ):
            return cache.registry
        cache.max_bs = next_power_of_2(max(raw_bs, cache.max_bs))
        cache.max_num_tokens = next_power_of_2(
            max(raw_num_tokens, cache.max_num_tokens)
        )
        cache.registry = build(cache.max_bs, cache.max_num_tokens)
        return cache.registry

    def _ensure_eager_decode_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> CudaGraphBufferRegistry:
        model_runner = self.model_runner
        is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        return self._ensure_eager_registry(
            self._eager_decode_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_decode_registry(
                device=model_runner.device,
                max_bs=bs,
                max_num_token=num_tokens,
                # Eager has no padding so this sentinel is never read; 0 avoids the
                # cuda-graph-only fill-value method that some backends lack.
                seq_len_fill_value=0,
                cache_loc_dtype=torch.int64,
                enable_mamba_track=(
                    model_runner.server_args.enable_mamba_extra_buffer()
                    and model_runner.spec_algorithm.is_none()
                ),
                is_encoder_decoder=is_encoder_decoder,
                encoder_len_fill_value=(
                    getattr(
                        model_runner.model_config.hf_config, "max_source_positions", 0
                    )
                    if is_encoder_decoder
                    else 0
                ),
                enable_num_token_non_padded=False,
                register_global_num_tokens=False,
                require_gathered_buffer=False,
                require_mlp_tp_gather=False,
                dp_size=model_runner.server_args.dp_size,
                share_pool=False,
                source=None,
            ),
        )

    def _ensure_eager_prefill_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> CudaGraphBufferRegistry:
        model_runner = self.model_runner
        return self._ensure_eager_registry(
            self._eager_prefill_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_prefill_registry(
                device=model_runner.device,
                max_bs=bs,
                max_num_token=num_tokens,
                cache_loc_dtype=torch.int64,
                is_multimodal=model_runner.is_multimodal,
                enable_mamba_track=False,
                register_input_embeds=False,
                share_pool=False,
                source=None,
            ),
        )

    def load_batch(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> ForwardBatch:
        """Copy the live batch into the eager static buffers (the grow-on-demand
        registry view) — the eager counterpart of the cuda-graph runners'
        load_batch (formerly ModelRunner._eager_fb_view)."""
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        raw_num_tokens = forward_batch.input_ids.shape[0]
        ensure = (
            self._ensure_eager_prefill_registry
            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            else self._ensure_eager_decode_registry
        )
        registry = ensure(raw_bs, raw_num_tokens)
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    # ------------------------------------------------------------------ #
    # execute — mode-dispatched eager forward (live model.forward)
    # ------------------------------------------------------------------ #
    def execute(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> Any:
        mode = forward_batch.forward_mode
        if mode.is_decode():
            return self._execute_decode(forward_batch, pp_proxy_tensors)
        if mode.is_idle():
            return self._execute_idle(forward_batch, pp_proxy_tensors)
        # extend (incl. draft_extend_v2) — returns (ret, can_run_graph=False)
        return self._execute_extend(forward_batch, pp_proxy_tensors)

    def _resolve_decode_pdmux(
        self,
    ) -> Tuple[Any, contextlib.AbstractContextManager]:
        """Resolve the (attn_backend, forward_context) the eager decode forward
        runs under. PDmux selects a per-stream backend and publishes it via an
        active ForwardContext; non-pdmux uses attn_backend + the ambient ctx."""
        model_runner = self.model_runner
        if model_runner.server_args.enable_pdmux:
            return model_runner.decode_attn_backend, forward_context(
                ForwardContext(attn_backend=model_runner.decode_attn_backend)
            )
        return model_runner.attn_backend, contextlib.nullcontext()

    def _execute_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        enable_pdmux = model_runner.server_args.enable_pdmux
        attn_backend, pdmux_ctx = self._resolve_decode_pdmux()
        if not enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
        # Set extra arguments
        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)

        # Launch forward
        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "decode"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )

        with ctx, pdmux_ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def _execute_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        model_runner = self.model_runner
        # Setup extra arguments (pp_proxy_tensors + input_embeds + get_embedding)
        kwargs = model_runner._extend_forward_kwargs(forward_batch, pp_proxy_tensors)

        if not model_runner.server_args.enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)

        # Launch model forward
        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            model_runner.attn_backend.init_forward_metadata(forward_batch)

        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "extend"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            pcg_runner = model_runner.prefill_cuda_graph_runner
            if (
                _is_hip
                and pcg_runner is not None
                and not isinstance(pcg_runner, EagerRunner)
            ):
                # AMD/HIP: when PCG is enabled but the batch exceeds max captured
                # size, run eagerly under enable_tc_piecewise_cuda_graph() and
                # set_tc_piecewise_forward_context() so that (a) Dynamo guards on
                # _in_tc_piecewise_cuda_graph stay consistent with the PCG-traced
                # graph (preventing runtime recompilation) and (b) PCG-specific
                # code paths (MoE, attention) can access their layer objects.
                with (
                    enable_tc_piecewise_cuda_graph(),
                    set_tc_piecewise_forward_context(
                        forward_batch,
                        model_runner.attention_layers,
                        getattr(model_runner.model, "quant_config", None),
                        model_runner.moe_layers,
                        model_runner.moe_fusions,
                        dsa_indexers=model_runner.dsa_indexers,
                    ),
                ):
                    ret = model_runner.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            else:
                ret = model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
        return (ret, False)

    def _execute_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            if not model_runner.server_args.enable_pdmux:
                forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
        else:
            model_runner.attn_backend.forward_metadata = None

        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)
        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "idle"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
