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
  conv plus a one-token ``val_proj2`` value lag, preprocessing q/k for the
  softmax attention.

These share the *state plumbing* -- the per-request slot indices, the
``has_initial_state`` prefix mask, the ``query_start_loc`` cu-seqlens and the
cuda-graph static index buffers, all resolved once per forward step -- but NOT
the conv kernel itself. This backend owns only the plumbing and hands it out via
:meth:`conv_state_metadata`; each model runs its own conv against that handle.

It is a *sidecar*: invoked directly by the model (through
:class:`ShortConvHybridAttnBackend
<sglang.srt.layers.attention.hybrid_linear_attn_backend.ShortConvHybridAttnBackend>`),
never through the full-vs-linear ``forward_decode`` / ``forward_extend``
dispatch. Metadata and cuda-graph capture/replay come from
:class:`MambaAttnBackendBase`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Sequence

import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    track_mamba_states_if_needed,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import mamba_cache_chunk_size

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class ShortConvMetadata(NamedTuple):
    """Per-(layer, step) conv-state handle handed to a model's conv kernel.

    ``layer_cache`` exposes the per-layer pool views (``conv[0]`` = conv state,
    ``conv[1]`` = an optional second state such as ZAYA1's val_proj2 lag,
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

    # int64 is canonical (the CUDA causal_conv1d narrows at its own boundary); a
    # subclass whose kernels take int32 sets int32 to skip that per-layer cast.
    cache_indices_dtype: torch.dtype = torch.int64
    # The host mirrors below cost a device->host sync per extend step, so only
    # models with a host extend loop (ZAYA1 v1) ask for them.
    needs_extend_host_mirrors: bool = True
    # Pure short conv: ``temporal`` is a zero-element tensor, so the radix
    # track never snapshots an SSM state and the base skips building its
    # (host-synchronizing) SSM track indices.
    has_temporal_state: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        # conv[0] == conv_state: [n_layers, n_slots, conv_dim, conv_kernel - 1]
        self.conv_states_shape = mamba_cache.conv[0].shape
        # Sliding-window length of EVERY conv entry (its trailing axis): ZAYA1
        # has two (the conv_qk left padding and the one-token lag), LFM2 one.
        # Each entry's state at length L is exactly that entry's last ``window``
        # INPUT rows, which makes the extend-side snapshot a plain gather.
        self.conv_window_lens: List[int] = [int(c.shape[-1]) for c in mamba_cache.conv]

        # Per-step state, resolved ONCE per step in init_forward_metadata /
        # init_forward_metadata_out_graph (never per conv layer). The extend host
        # mirrors drive the extend loop; ``_cache_indices`` is the int64 slot
        # index view shared by all conv layers within the step.
        self._has_initial_state: Optional[torch.Tensor] = None
        self._slot_ids_cpu: Optional[List[int]] = None
        self._has_prefix_cpu: Optional[List[bool]] = None
        self._cache_indices: Optional[torch.Tensor] = None
        self._cache_indices_buf: Optional[torch.Tensor] = None
        self._has_initial_state_buf: Optional[torch.Tensor] = None

        # --- mamba radix cache, extra_buffer strategy ---------------------
        # Per-step extend track state: one flattened-token index tensor per
        # conv entry ([n_tracked, window]) plus the destination track slots.
        # Both None unless this step actually tracks something.
        self._track_conv_indices: Optional[List[torch.Tensor]] = None
        self._track_dst: Optional[torch.Tensor] = None
        # Decode-side, all-layers-at-once track plumbing (see
        # _init_track_state); None when the strategy is not extra_buffer.
        self._track_layer_row_base: Optional[torch.Tensor] = None
        self._track_pairs: Optional[List[tuple]] = None
        self.enable_mamba_extra_buffer = (
            model_runner.server_args.enable_mamba_extra_buffer()
        )
        if self.enable_mamba_extra_buffer:
            self._init_track_state(model_runner.server_args, mamba_cache)

    def _init_track_state(self, server_args, mamba_cache) -> None:
        """Validate + precompute the radix track plumbing (extra_buffer only).

        The decode snapshot is a pure row copy (live slot -> track slot) that
        every conv layer needs on the same step, so instead of one launch per
        layer this flattens the pool's ``[n_layers, n_slots, ...]`` conv tensors
        to ``[n_layers * n_slots, ...]`` and does the whole model in one, with row
        ids ``layer * n_slots + slot``.
        """
        # The prefill-CUDA-graph and speculative-decoding refusals are pure
        # config combinations, so they live in
        # ServerArgs._validate_mamba_extra_buffer, which reads the resolving
        # view rather than this supplied record.
        chunk = server_args.mamba_cache_chunk_size
        max_window = max(self.conv_window_lens)
        # The extend snapshot gathers the ``window`` rows ending at the
        # chunk-aligned track position, at least mamba_cache_chunk_size into the
        # current extend. A longer window would have to reach back into the
        # cached prefix, which the gather cannot express.
        assert max_window < chunk, (
            f"short-conv extra_buffer needs every conv window "
            f"({self.conv_window_lens}) < mamba_cache_chunk_size ({chunk}); "
            f"the minimum viable chunk here is {max_window + 1}. This is "
            "derived in ServerArgs.mamba_cache_chunk_size, which must not "
            "take a conv-only model's mamba_chunk_size (its scan length, 1) "
            "as the caching granularity."
        )
        assert server_args.mamba_track_interval >= chunk, (
            f"mamba_track_interval ({server_args.mamba_track_interval}) must be "
            f">= mamba_cache_chunk_size ({chunk})"
        )

        num_layers, num_slots = mamba_cache.conv[0].shape[:2]
        entries = []
        for conv in mamba_cache.conv:
            assert tuple(conv.shape[:2]) == (num_layers, num_slots), (
                "all conv entries must share the pool's [n_layers, n_slots] "
                f"leading dims, got {tuple(conv.shape[:2])}"
            )
            if not conv.is_contiguous():
                # Page-major / envelope conv views are strided, so the
                # flatten(0, 1) row addressing below is invalid. Fail loudly
                # rather than silently snapshotting the wrong bytes.
                raise NotImplementedError(
                    "mamba extra_buffer for short-conv models requires "
                    "contiguous conv state; the page-major envelope layout is "
                    "not supported yet."
                )
            entries.append(conv.flatten(0, 1))
        if len(entries) % 2 == 1:
            # The shared track kernel copies exactly two state tensors per
            # launch. Short-conv models carry a zero-element ``temporal``, so
            # it pairs with an odd conv entry at no cost.
            temporal = mamba_cache.temporal.flatten(0, 1)
            assert temporal[0].numel() == 0, (
                "short-conv backend expects an empty temporal state; a real "
                "SSM state would be silently dropped from the radix snapshot"
            )
            entries.append(temporal)
        self._track_pairs = [
            (entries[i], entries[i + 1]) for i in range(0, len(entries), 2)
        ]
        self._track_layer_row_base = (
            torch.arange(num_layers, dtype=torch.int64, device=self.device) * num_slots
        ).unsqueeze(1)

    def _reset_step_state(self):
        self._has_initial_state = None
        self._slot_ids_cpu = None
        self._has_prefix_cpu = None
        self._track_conv_indices = None
        self._track_dst = None

    def _alloc_cache_indices_buf(self, max_bs: int):
        # Refilled in place per step so a captured graph reads a stable address.
        # Grow-only, never reallocated at the same size: the cuda- and cpu-graph
        # hooks can both run, in either order, after another phase captured.
        # ``_has_initial_state_buf`` matters once the PREFILL graph is captured:
        # a fused extend conv reads the prefix mask from inside the graph.
        buf = self._cache_indices_buf
        if buf is not None and buf.shape[0] >= max_bs:
            return
        assert buf is None, (
            f"cache-indices buffer must be sized before any graph capture: have "
            f"{buf.shape[0]}, need {max_bs}"
        )
        self._cache_indices_buf = torch.empty(
            max_bs, dtype=self.cache_indices_dtype, device=self.device
        )
        self._has_initial_state_buf = torch.empty(
            max_bs, dtype=torch.bool, device=self.device
        )

    def _refresh_cache_indices(self):
        # ONCE per step, shared by every conv layer. With a graph buffer, refill IN
        # PLACE and hand out a view so the captured address stays current; otherwise
        # (eager, or bs past the buffer) a fresh cast is fine.
        md = self.forward_metadata
        idx = md.mamba_cache_indices if md is not None else None
        buf = self._cache_indices_buf
        # Batch padding poisons unused rows' slot ids to -1, under cuda-graph bs
        # rounding and whenever DP attention pads a replica's batch. Clamp ONCE
        # per step, here where the shared int64 view is resolved, so every conv
        # layer's index_select / index_copy_ is in bounds: MambaPool reserves
        # slot 0, so padded rows land on that scratch slot and can neither read
        # out of bounds nor clobber a live request's state. An unclamped -1 is an
        # out-of-bounds device gather, which on ROCm aborts the queue with
        # HSA_STATUS_ERROR_EXCEPTION 0x1016 rather than raising.
        if idx is None:
            self._cache_indices = None
        elif buf is not None and idx.shape[0] <= buf.shape[0]:
            n = idx.shape[0]
            buf[:n].copy_(idx)
            buf[:n].clamp_(min=0)
            self._cache_indices = buf[:n]
        else:
            # ``clamp`` (not ``clamp_``): ``to()`` is a no-op alias when idx already
            # has cache_indices_dtype, so an in-place clamp would mutate the
            # backend's own mamba_cache_indices.
            self._cache_indices = idx.to(self.cache_indices_dtype).clamp(min=0)

    def _resolve_has_initial_state(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Per-request "resumes a cached prefix" mask, in a graph-stable buffer.

        Refills the persistent buffer in place when it is allocated and wide
        enough and hands out a view; otherwise (eager, or a batch beyond the
        buffer) a fresh mask is fine.
        """
        mask = forward_batch.extend_prefix_lens > 0
        buf = self._has_initial_state_buf
        if buf is None or mask.shape[0] > buf.shape[0]:
            return mask
        n = mask.shape[0]
        buf[:n].copy_(mask)
        return buf[:n]

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cpu_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ) -> List[torch.Tensor]:
        """Flattened input positions to snapshot, ONE tensor per conv entry.

        Overrides the single-conv base implementation: a short-conv model may
        carry several conv entries with different window lengths, each snapshot
        being its own window of its own input tensor. Every entry's window ENDS
        at the same chunk-aligned track position, so ``indices[j][:, -1]`` is the
        same column for every ``j``. Returned tensors are ``[n_tracked,
        window_j]`` over the flattened token axis, restricted to
        ``mamba_track_mask``.
        """
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        chunk = mamba_cache_chunk_size()
        aligned_len = (lens_to_track // chunk) * chunk
        # One past the last token whose input belongs in the snapshot.
        end = (query_start_loc[:-1] + aligned_len)[forward_batch.mamba_track_mask]
        last = query_start_loc[-1] - 1
        out: List[torch.Tensor] = []
        for window in self.conv_window_lens:
            starts = end - window
            offsets = torch.arange(window, device=self.device, dtype=starts.dtype)
            out.append((starts.unsqueeze(-1) + offsets).clamp(0, last))
        return out

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
            self._has_initial_state = self._resolve_has_initial_state(forward_batch)
            if self.needs_extend_host_mirrors and self._cache_indices is not None:
                self._slot_ids_cpu = self._cache_indices.tolist()
                self._has_prefix_cpu = [
                    int(p) > 0 for p in forward_batch.extend_prefix_lens_cpu
                ]
        # Extend-side radix track: the base only populates track_conv_indices
        # on the plain-extend branch and only when some row is tracked, so its
        # presence is the gate. mamba_track_indices was translated
        # virtual->physical in place by _forward_metadata.
        md = self.forward_metadata
        if md is not None and md.track_conv_indices is not None:
            self._track_conv_indices = md.track_conv_indices
            self._track_dst = forward_batch.mamba_track_indices[
                forward_batch.mamba_track_mask
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
        # the int64 view, so without this the conv layers capture a ``None``
        # index. Replay refills the SAME buffer through init_forward_metadata.
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

    # ------------------------------------------------------------------
    # Mamba radix cache, extra_buffer strategy: the track snapshot
    # ------------------------------------------------------------------
    # Under `no_buffer` the radix tree is handed the request's LIVE state slot,
    # current only at the exact token count the scheduler last saw -- hence
    # page_size == 1 and no overlap schedule. `extra_buffer` gives each request
    # extra pool slots (the ping-pong track buffer) and snapshots the state into
    # them at KNOWN, chunk-aligned lengths, which is what lets the cached key
    # length and the cached state agree while the scheduler runs a step ahead of
    # the GPU. Without the snapshot, a prefix hit restores garbage conv state.

    def track_conv_states_extend(
        self,
        conv_states: Sequence[Optional[torch.Tensor]],
        conv_inputs: Sequence[Optional[torch.Tensor]],
    ) -> None:
        """Snapshot this layer's conv entries at the chunk-aligned track point.

        ``conv_states[j]`` must be the EXACT tensor view the conv wrote its state
        through, not ``layer_cache.conv[j]`` re-derived here: a rank may own only
        a leading sub-slice of a rank-uniform pool entry, and the snapshot has to
        land in the same slice or a prefix restore reloads a row the decode path
        never wrote.

        ``conv_inputs[j]`` is the ``[T, C_j]`` tensor whose last ``window_j``
        rows ARE that state after the conv runs. A pair is skipped when either
        side is ``None`` -- a rank owning no lag stream has nothing to snapshot.

        Call once per conv layer on the extend path; the state slot the conv
        itself writes is a different row, so before-or-after is equivalent. A
        no-op unless this step tracks something.
        """
        index_list = self._track_conv_indices
        if index_list is None:
            return
        dst = self._track_dst
        assert len(conv_states) == len(conv_inputs) == len(index_list), (
            f"expected {len(index_list)} (state, input) pairs, got "
            f"{len(conv_states)} states and {len(conv_inputs)} inputs"
        )
        for conv_state, x, indices in zip(conv_states, conv_inputs, index_list):
            if conv_state is None or x is None:
                continue
            # Checked against the state the conv actually wrote: a channel
            # mismatch means the snapshot and the live state are different
            # quantities, and the scatter below would write the wrong rows.
            assert conv_state.shape[-2] == x.shape[-1], (
                f"conv state has {conv_state.shape[-2]} channels but its input "
                f"tensor has {x.shape[-1]}; the snapshot must cache exactly "
                "what the conv state holds"
            )
            assert conv_state.shape[-1] == indices.shape[-1], (
                f"conv state window is {conv_state.shape[-1]} but the track "
                f"index build used {indices.shape[-1]}"
            )
            # [C, T] -> [C, n_tracked, window] -> [n_tracked, C, window]
            window = x.transpose(0, 1)[:, indices].transpose(0, 1)
            conv_state[dst] = window.to(conv_state.dtype)

    def track_conv_states_decode(self, forward_batch: ForwardBatch) -> None:
        """Snapshot EVERY conv layer's state into the track slots (one launch).

        Call once per decode step, after the last conv layer has updated its
        state: ``mamba_track_mask`` is built from the POST-increment seq_lens,
        so the row is tracked on the step whose output makes the length a
        multiple of ``mamba_track_interval``.

        CUDA-graph contract. Every tensor read here is either a persistent buffer
        refilled in place before replay or a constant allocated at init, so
        capture MUST reach this call and record the scatter: during capture the
        mask buffer is all-False and the kernel is inert, but the launch is in
        the graph and the refilled mask makes it fire at replay. Skipping the
        launch because "nothing is tracked right now" silently drops every
        snapshot for the life of the graph.
        """
        if self._track_pairs is None:
            return
        if not forward_batch.forward_mode.is_decode_or_idle():
            return
        md = self.forward_metadata
        src = self._cache_indices
        mask = forward_batch.mamba_track_mask
        dst = md.mamba_track_indices if md is not None else None
        if src is None or mask is None or dst is None:
            return
        bs = src.shape[0]
        if bs == 0:
            return

        row_ok = mask[:bs]
        if self.enable_unified_memory:
            # The unified pool's v2p translate tombstones freed slots with -1.
            # Folded into the mask rather than left to the kernel's own check,
            # because `layer_base + -1` aliases the previous layer's last slot.
            # `_cache_indices` already clamped its -1s, hence the raw tensor.
            raw_src = md.mamba_cache_indices
            row_ok = row_ok & (dst[:bs] >= 0) & (raw_src[:bs] >= 0)
        base = self._track_layer_row_base  # [n_layers, 1]
        num_layers = base.shape[0]
        # Row id of (layer, slot) in the flattened [n_layers * n_slots, ...] view.
        src_rows = (base + src).reshape(-1)
        dst_rows = (base + dst[:bs]).reshape(-1)
        mask_rows = row_ok.expand(num_layers, bs).reshape(-1)
        total_rows = num_layers * bs
        for state_a, state_b in self._track_pairs:
            track_mamba_states_if_needed(
                state_a,
                state_b,
                src_rows,
                mask_rows,
                dst_rows,
                total_rows,
                # Invalid rows are already masked off above.
                check_freed_slots=False,
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
