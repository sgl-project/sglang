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
"""Short-convolution attention backend.

Several hybrid models interleave a *causal short conv with per-request conv
state* (stored in the centralized ``MambaPool``) with softmax attention layers:

* **LFM2** (:class:`Lfm2ShortConv <sglang.srt.models.lfm2.Lfm2ShortConv>`) --
  a depthwise gated short conv (``causal_conv1d_fn`` / ``causal_conv1d_update``)
  as a standalone token mixer on its own conv layers.
* **ZAYA1** (:class:`CCA <sglang.srt.models.zaya.CCA>`) -- a two-stage grouped
  conv plus a one-token ``prev_hs`` lag, preprocessing q/k for the layer's
  softmax attention.

These share the *state plumbing* -- resolving the per-request slot indices, the
``has_initial_state`` prefix mask, the ``query_start_loc`` cu-seqlens, and the
cuda-graph static index buffers, all once per forward step -- but NOT the conv
kernel itself. ``ShortConvAttnBackend`` owns only the plumbing and hands it out
via :meth:`conv_state_metadata` as a :class:`ShortConvMetadata`; each model runs
its own conv kernel against that handle, so the model definition holds no pool
access.

The backend is a *sidecar*: it is invoked directly by the model (through
:class:`ShortConvHybridAttnBackend
<sglang.srt.layers.attention.hybrid_linear_attn_backend.ShortConvHybridAttnBackend>`),
never through the full-vs-linear ``forward_decode`` / ``forward_extend``
dispatch. Metadata + cuda-graph capture/replay come from
:class:`MambaAttnBackendBase`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class ShortConvMetadata(NamedTuple):
    """Per-(layer, step) conv-state handle handed to a model's conv kernel.

    ``layer_cache`` exposes the per-layer pool views (``conv[0]`` = conv state,
    ``conv[1]`` = an optional second state such as ZAYA1's ``prev_hs``,
    ``temporal`` = SSM state, unused by pure short convs). The device tensors are
    cuda-graph-static on the decode/replay path; the ``*_cpu`` host mirrors are
    built once per step only for models whose extend path runs a host loop
    (e.g. ZAYA1 v1) and are ``None`` on decode.
    """

    layer_cache: Any
    cache_indices: torch.Tensor
    # cu-seqlens for the varlen prefill conv (device, int32). None on decode.
    query_start_loc: Optional[torch.Tensor] = None
    # Per-request "resumes a cached prefix" mask (device bool). None on decode.
    has_initial_state: Optional[torch.Tensor] = None
    # Host mirror of cache_indices for extend host loops. None on decode.
    slot_ids_cpu: Optional[List[int]] = None
    # Host mirror of has_initial_state for extend host loops. None on decode.
    has_prefix_cpu: Optional[List[bool]] = None


class ShortConvAttnBackend(MambaAttnBackendBase):
    """Owns the short-conv per-request state plumbing (see module docstring)."""

    # State IO is index-driven; no host seq-lens plumbing required from the
    # runner. (The extend path reads ``extend_*_cpu`` off the batch, which is
    # always populated for extend regardless of this flag.)
    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        # conv[0] == conv_state: [n_layers, n_slots, conv_dim, conv_kernel - 1]
        self.conv_states_shape = mamba_cache.conv[0].shape

        # Per-step state, resolved ONCE per step in init_forward_metadata /
        # init_forward_metadata_out_graph (never per conv layer). The extend host
        # mirrors drive the extend loop; ``_cache_indices`` is the int64 slot
        # index view shared by all conv layers within the step.
        self._has_initial_state: Optional[torch.Tensor] = None
        self._slot_ids_cpu: Optional[List[int]] = None
        self._has_prefix_cpu: Optional[List[bool]] = None
        self._cache_indices: Optional[torch.Tensor] = None
        self._cache_indices_buf: Optional[torch.Tensor] = None

    def _reset_step_state(self):
        self._has_initial_state = None
        self._slot_ids_cpu = None
        self._has_prefix_cpu = None

    def _alloc_cache_indices_buf(self, max_bs: int):
        # Persistent int64 index buffer, refilled in place per step so the
        # captured (cuda or cpu) graph reads a stable address.
        self._cache_indices_buf = torch.empty(
            max_bs, dtype=torch.int64, device=self.device
        )

    def _refresh_cache_indices(self):
        # Resolve the int64 slot-index view ONCE per step, shared by every conv
        # layer. When a graph index buffer is allocated and large enough, refill
        # it IN PLACE and hand out a view -- the captured graph then reads a
        # stable address that this (pre-replay) hook keeps current, so it is
        # cuda- and cpu-graph safe. Otherwise (eager, or bs beyond the buffer)
        # a fresh cast is fine.
        md = self.forward_metadata
        idx = md.mamba_cache_indices if md is not None else None
        buf = self._cache_indices_buf
        if idx is None:
            self._cache_indices = None
        elif buf is not None and idx.shape[0] <= buf.shape[0]:
            n = idx.shape[0]
            buf[:n].copy_(idx)
            self._cache_indices = buf[:n]
        else:
            self._cache_indices = idx.to(torch.long)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cpu_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Eager path (also the CPU-graph replay path). Builds
        # self.forward_metadata and runs the deferred mamba clear/COW ops.
        super().init_forward_metadata(forward_batch)
        self._reset_step_state()
        self._refresh_cache_indices()
        mode = forward_batch.forward_mode
        if (
            mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend_v2()
        ):
            self._has_initial_state = forward_batch.extend_prefix_lens > 0
            if self._cache_indices is not None:
                self._slot_ids_cpu = self._cache_indices.tolist()
                self._has_prefix_cpu = [
                    int(p) > 0 for p in forward_batch.extend_prefix_lens_cpu
                ]

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # Decode cuda-graph capture + replay path -- no extend prefix state.
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._reset_step_state()
        self._refresh_cache_indices()

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        # Decode CPU-graph capture path. The base fills forward_metadata but not
        # the int64 view; without this the conv layers would capture a ``None``
        # index (crash / corrupt state). Replay goes through init_forward_metadata
        # and refills the SAME buffer, so the captured cpu graph reads a stable
        # address kept current at replay.
        super().init_forward_metadata_capture_cpu_graph(*args, **kwargs)
        self._reset_step_state()
        self._refresh_cache_indices()

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> ShortConvMetadata:
        """Return the conv-state handle for ``layer_id`` at the current step.

        The per-step fields are already resolved on ``self.forward_metadata`` /
        ``self._*`` (in ``init_forward_metadata`` / ``_out_graph``);
        ``forward_batch`` is accepted for interface parity with the unit-test
        mock and is not otherwise required here.
        """
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        md = self.forward_metadata

        # Slot indices are cached ONCE per step in init_forward_metadata /
        # init_forward_metadata_out_graph (int64). Hand back the cached view -- no
        # per-layer recompute. Decode is cuda-graph-safe because that view is a
        # persistent buffer refilled in place before each replay.
        return ShortConvMetadata(
            layer_cache=layer_cache,
            cache_indices=self._cache_indices,
            query_start_loc=md.query_start_loc,
            has_initial_state=self._has_initial_state,
            slot_ids_cpu=self._slot_ids_cpu,
            has_prefix_cpu=self._has_prefix_cpu,
        )

    # The short-conv layers are invoked via conv_state_metadata + the model's own
    # conv kernel, never through the HybridLinearAttnBackend full-vs-linear
    # dispatch. Mirror Mamba2AttnBackend and guard the routed entrypoints.
    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "ShortConvAttnBackend is invoked via conv_state_metadata; "
            "it does not run through forward_decode."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "ShortConvAttnBackend is invoked via conv_state_metadata; "
            "it does not run through forward_extend."
        )
