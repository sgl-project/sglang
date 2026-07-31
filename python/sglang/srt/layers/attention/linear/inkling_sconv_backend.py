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
"""Inkling's short-conv state backend.

Inkling is a short-conv hybrid in the sense of
:mod:`~sglang.srt.layers.attention.linear.short_conv_backend`: every layer is
softmax attention, and four causal short convs per decoder layer -- ``k_sconv`` /
``v_sconv`` preprocessing that layer's attention, plus the ``attn`` and ``mlp``
output-stream convs -- carry per-request conv state in the centralized
``MambaPool`` (six streams pool-wide, since full and local layers get separate
k/v slots). So this backend is a *sidecar*:
:class:`InklingShortConvHybridAttnBackend` drives its metadata hooks, the model
reaches it via :meth:`conv_state_metadata`, and it never serves
``forward_decode`` / ``forward_extend``.

On top of the plumbing :class:`ShortConvAttnBackend` owns (slot indices,
``has_initial_state``, ``query_start_loc``, graph-static buffers, once-per-step
resolution) Inkling's conv kernels consume a *precomputed* metadata set
(``cache_mask`` / ``safe_idx`` / ``cu`` / ``si``), which its fused metadata kernel
emits in the same launch as ``query_start_loc`` / ``has_initial_state``, plus the
extend prefix-cache ``track_conv_indices``. All of it is step-global -- it depends
only on ``req_pool_indices`` and the per-request lengths, never on the layer or
the conv stream -- so this backend resolves the whole set ONCE per step during
metadata prep and every conv layer just reads the cached handle.

That is the point of the migration. The pre-backend model code keyed ownership on
``layer_id == 0``, but Inkling's decoder layer 0 holds FOUR ``ShortConvolution``
modules (``k_sconv``, ``v_sconv``, ``attn_sconv``, ``mlp_sconv``), so the slot
gather+translate, the fused metadata kernel and the track-index prep each ran
four times per step instead of once.

Which metadata-prep hook does what follows the graph contract, and the split
matters for decode latency:

* ``init_forward_metadata_in_graph`` is *recorded* into the decode /
  target-verify / draft-extend graphs, so prep placed there replays for free.
  Both the slot lookup and the fused metadata launch go there -- resolving them
  out of graph instead measured ~7% off decode throughput, since a captured graph
  is precisely the place where per-step CPU launches hurt.
* ``init_forward_metadata_out_graph`` takes whatever the phase cannot record: the
  full-cuda-graph prefill (whose runner has no in-graph hook) and, for the
  unified memory pool, the slot translate -- an allocator lookup, not a table
  gather, so it must run eagerly before each replay.
* ``init_forward_metadata`` covers the eager path and does both.

Because the resolution no longer sits in the model, every tensor a captured conv
kernel reads has to live at a stable address: the graph-static buffers below are
refilled in place per step, the same contract as
``ShortConvAttnBackend._cache_indices_buf`` and the full-attention backend's
``swa_out_cache_loc_buf``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    ShortConvHybridAttnBackend,
)
from sglang.srt.layers.attention.linear.short_conv_backend import ShortConvAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.inkling_common.kernels.sconv import (
    HIS_ONES,
    HIS_PREFIX,
    HIS_SEQ_MINUS_EXT,
    HIS_ZEROS,
    SconvDecodeMetadata,
    SconvExtendMetadata,
    SconvMetadataOut,
    fused_decode_sconv_metadata,
    fused_extend_sconv_metadata,
    precompute_helion_extend_metadata,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class InklingShortConvMetadata(NamedTuple):
    """Per-(layer, step) conv-state handle handed to Inkling's conv kernels.

    ``layer_cache`` exposes this layer's pool views: ``conv[SconvType]`` is one
    stream's conv state and ``intermediate_conv_window[SconvType]`` its
    per-draft-token snapshot buffer. Everything else is step-global and already
    resolved; on the graph path the device tensors are static buffers refilled in
    place before each replay.
    """

    layer_cache: Any
    # Per-request conv-state slot ids (int32).
    cache_indices: torch.Tensor
    # cu-seqlens for the varlen conv (device, int32).
    query_start_loc: Optional[torch.Tensor] = None
    # Per-request "resumes a cached prefix" mask (device bool).
    has_initial_state: Optional[torch.Tensor] = None
    # cache_mask / safe_idx / cu / si for the conv kernel.
    precomputed: Optional[SconvExtendMetadata | SconvDecodeMetadata] = None
    # [B, conv_kernel - 1] flattened input positions whose conv window feeds the
    # prefix cache. Extend only; None elsewhere and when tracking is off.
    track_conv_indices: Optional[torch.Tensor] = None


class InklingShortConvAttnBackend(ShortConvAttnBackend):
    """Owns Inkling's per-step short-conv state plumbing (see module docstring)."""

    # Inkling's conv kernels take int32 slot ids (also the pool's dtype), so keep
    # the shared index view int32: an int64 view would re-run a narrowing cast in
    # every conv layer.
    cache_indices_dtype: torch.dtype = torch.int32

    # Inkling's extend path is fully device-side; the ZAYA1-style host mirrors
    # would only add a device->host sync per step.
    needs_extend_host_mirrors: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # conv[i] is [n_layers, n_slots, conv_kernel - 1, conv_dim].
        self.conv_state_len: int = self.conv_states_shape[2]
        self.mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size
        # The slot gather is a plain table lookup, so it can be recorded into the
        # graph -- unless the pool overrides the virtual->physical translate (the
        # unified memory pool resolves it through the allocator, which is host
        # work and must stay in the out-of-graph replay prep).
        self._slot_gather_recordable = (
            type(self.req_to_token_pool).translate_mamba_indices
            is HybridReqToTokenPool.translate_mamba_indices
        )

        # Per-step state, resolved ONCE per step (never per conv layer).
        self._query_start_loc: Optional[torch.Tensor] = None
        self._precomputed: Optional[SconvExtendMetadata | SconvDecodeMetadata] = None
        self._track_conv_indices: Optional[torch.Tensor] = None

        self._alloc_graph_buffers()

    # ------------------------------------------------------------------
    # graph-static metadata buffers
    # ------------------------------------------------------------------

    def _alloc_graph_buffers(self):
        """Allocate the per-step metadata destinations for the graph path.

        Bounds come from the *configured* capture shapes rather than from
        observed batches: growing a buffer after a graph has been captured would
        move the address that graph reads (and the prefill graph is captured
        before the decode runner reports its own bounds), so these are sized once,
        up front, and never reallocated.
        """
        server_args = get_server_args()
        cuda_graph_config = server_args.cuda_graph_config
        decode_bs: list[int] = []
        prefill_tokens: list[int] = []
        decode_max_bs = 0
        if cuda_graph_config is not None:
            decode_bs = list(cuda_graph_config.decode.bs or [])
            prefill_tokens = list(cuda_graph_config.prefill.bs or [])
            decode_max_bs = cuda_graph_config.decode.max_bs or 0
        draft_token_num = server_args.speculative_num_draft_tokens or 1
        # req_to_token_pool.size is the runner's max_bs for both graph phases.
        max_bs = max([self.req_to_token_pool.size, decode_max_bs, *decode_bs])
        max_tokens = max([max_bs, *prefill_tokens, max_bs * draft_token_num])

        dev = self.device
        self._graph_bufs = SconvMetadataOut(
            query_start_loc=torch.empty(max_bs + 1, dtype=torch.int32, device=dev),
            has_initial_state=torch.empty(max_bs, dtype=torch.bool, device=dev),
            cache_mask=torch.empty((max_bs, 1, 1), dtype=torch.bool, device=dev),
            safe_idx=torch.empty(max_bs, dtype=torch.int64, device=dev),
            cu=torch.empty(max_bs + 1, dtype=torch.int64, device=dev),
            si=torch.empty(max_tokens, dtype=torch.int32, device=dev),
        )
        self._graph_track_conv_indices = torch.zeros(
            (max_bs, self.conv_state_len), dtype=torch.int64, device=dev
        )
        # The slot-index view must be address-stable for the same reason. The base
        # only allocates it from init_cuda_graph_state, which the prefill graph
        # never calls, so size it here instead.
        self._alloc_cache_indices_buf(max_bs)
        # Reused constants for the in-place track-index build.
        self._track_window_offsets = torch.arange(
            self.conv_state_len, dtype=torch.int64, device=dev
        )
        self._track_index_floor = torch.zeros((1,), dtype=torch.int64, device=dev)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        # The buffers were sized in __init__ from the configured capture shapes;
        # check the bound here so a phase that wants more fails at startup rather
        # than at its first replay.
        self._graph_metadata_out(B=max_bs, T=max_num_tokens)

    def _graph_metadata_out(self, *, B: int, T: int) -> SconvMetadataOut:
        """Graph-static destinations sliced to this step.

        Fails loud rather than falling back to a fresh allocation: this is only
        reached from the graph path, where a fresh allocation would silently leave
        the captured kernels reading a dead address.
        """
        bufs = self._graph_bufs
        assert B + 1 <= bufs["query_start_loc"].shape[0] and T <= bufs["si"].shape[0], (
            f"short-conv metadata buffers too small for a captured shape: "
            f"B={B}, T={T} vs bs bound {bufs['query_start_loc'].shape[0] - 1}, "
            f"token bound {bufs['si'].shape[0]}"
        )
        return SconvMetadataOut(
            query_start_loc=bufs["query_start_loc"][: B + 1],
            has_initial_state=bufs["has_initial_state"][:B],
            cache_mask=bufs["cache_mask"][:B],
            safe_idx=bufs["safe_idx"][:B],
            cu=bufs["cu"][: B + 1],
            si=bufs["si"][:T],
        )

    # ------------------------------------------------------------------
    # per-step metadata prep
    # ------------------------------------------------------------------

    def _forward_metadata(self, forward_batch: ForwardBatch) -> ForwardMetadata:
        """Resolve the per-request conv-state slot ids for this step.

        Deliberately leaner than ``MambaAttnBackendBase._forward_metadata``:
        Inkling has no SSM state (so no ``track_ssm_*`` prep, which also syncs),
        its conv window sits on a different axis than the shared
        ``_init_track_conv_indices`` assumes, and its ``query_start_loc`` comes
        out of the fused metadata kernel. This is the exact gather + virtual->
        physical translate the model used to run inside its layer-0 conv modules.
        """
        return ForwardMetadata(
            query_start_loc=None,
            mamba_cache_indices=self._translate_mamba_indices(
                self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
            ),
        )

    def _reset_step_state(self):
        super()._reset_step_state()
        self._query_start_loc = None
        self._precomputed = None
        self._track_conv_indices = None

    @staticmethod
    def _phase_records_metadata(forward_batch: ForwardBatch) -> bool:
        """True when this phase's runner records ``init_forward_metadata_in_graph``.

        The decode / target-verify / draft-extend graph runners do, so metadata
        prep placed there replays for free instead of sitting on the per-step CPU
        path -- which matters: resolving it out of graph measured ~7% off decode
        throughput, and a captured graph is exactly where per-step launches hurt.
        The full-cuda-graph *prefill* runner offers no such hook, so that phase
        resolves out of graph into the same static buffers.
        """
        mode = forward_batch.forward_mode
        return (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager path: nothing downstream is captured, so let the kernels
        allocate."""
        self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=False)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        """Out-of-graph replay prep: whatever this phase cannot record.

        Kept as short as possible -- this runs before *every* graph replay, and on
        the common path (a recordable pool in a phase with an in-graph hook) it is
        a single predicate and a return.
        """
        del in_capture
        if self._phase_records_metadata(forward_batch):
            if not self._slot_gather_recordable:
                self._prepare_slot_indices(forward_batch)
            return
        self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=True)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recorded prep: writes into the graph-static destinations, so the
        recorded launches refill them at every replay -- no allocation, honoring
        the hook's contract."""
        if not self._phase_records_metadata(forward_batch):
            return
        if self._slot_gather_recordable:
            self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=True)

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "Inkling's short-conv backend has no CPU-graph path; its conv "
            "kernels are CUDA/Triton only."
        )

    def _prepare_slot_indices(self, forward_batch: ForwardBatch):
        self._reset_step_state()
        req_pool_indices = forward_batch.req_pool_indices
        n = req_pool_indices.shape[0]
        buf = self._cache_indices_buf
        if self._slot_gather_recordable and n <= buf.shape[0]:
            # ONE launch: gather the slot table straight into the graph-static
            # index buffer. Going through the base's gather-then-copy would add a
            # second (recorded) kernel to every decode step for no benefit -- the
            # pool's table already has this buffer's dtype.
            torch.index_select(
                self.req_to_token_pool.req_index_to_mamba_index_mapping,
                0,
                req_pool_indices,
                out=buf[:n],
            )
            self._cache_indices = buf[:n]
            self.forward_metadata = ForwardMetadata(
                query_start_loc=None, mamba_cache_indices=self._cache_indices
            )
            return
        self.forward_metadata = self._forward_metadata(forward_batch)
        self._refresh_cache_indices()

    def _refresh_sconv_metadata(
        self, forward_batch: ForwardBatch, *, on_graph_path: bool
    ):
        if self._cache_indices is None:
            return
        mode = forward_batch.forward_mode
        if mode.is_decode_or_idle():
            self._refresh_decode_metadata(forward_batch, on_graph_path)
        elif mode.is_target_verify():
            self._refresh_extend_metadata(forward_batch, on_graph_path)
        elif mode.is_extend(include_draft_extend_v2=True):
            self._refresh_extend_metadata(forward_batch, on_graph_path)
            self._refresh_track_conv_indices(forward_batch, on_graph_path)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

    def _refresh_decode_metadata(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):
        B = forward_batch.batch_size
        (
            self._query_start_loc,
            self._has_initial_state,
            self._precomputed,
        ) = fused_decode_sconv_metadata(
            B=B,
            cache_indices=self._cache_indices,
            out=self._graph_metadata_out(B=B, T=B) if on_graph_path else None,
        )

    def _refresh_extend_metadata(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):
        """Fused ``(query_start_loc, has_initial_state, SconvExtendMetadata)`` in
        one launch, with the unfused op sequence as the off-CUDA / large-batch
        fallback."""
        B = forward_batch.batch_size
        if forward_batch.forward_mode.is_target_verify():
            # target_verify does not populate extend_seq_lens/extend_prefix_lens;
            # the lens are a constant draft_token_num per request.
            draft_token_num = forward_batch.spec_info.draft_token_num
            T = B * draft_token_num
            his_kwargs = dict(his_mode=HIS_ONES, draft_token_num=draft_token_num)
        else:
            T = forward_batch.extend_num_tokens
            spec_info = forward_batch.spec_info
            if (
                isinstance(spec_info, EagleDraftExtendInput)
                and spec_info.num_front_tokens > 0
            ):
                # Boundary-KV fix: run conv fresh so warm-up rows rebuild the
                # window.
                his_mode, his_src = HIS_ZEROS, None
            elif forward_batch.extend_prefix_lens is not None:
                his_mode, his_src = HIS_PREFIX, forward_batch.extend_prefix_lens
            else:
                # draft_extend_v2 capture has no extend_prefix_lens.
                his_mode, his_src = HIS_SEQ_MINUS_EXT, forward_batch.seq_lens
            his_kwargs = dict(
                his_mode=his_mode,
                extend_seq_lens=forward_batch.extend_seq_lens,
                his_src=his_src,
            )

        # The captured conv kernels bake their token extent at CAPTURE (the
        # prefill bucket), while replay only tells us the live token count. So on
        # the graph path fill the whole seq-index buffer: the kernel writes the
        # real mapping for t < T and clamps the tail to B - 1, exactly the values
        # the pre-migration in-graph prep produced past the live extent.
        # target_verify's extent is B * draft_token_num, exact at capture and
        # replay alike, so it keeps the tight slice.
        fill_T = T
        out = None
        if on_graph_path:
            if not forward_batch.forward_mode.is_target_verify():
                fill_T = self._graph_bufs["si"].shape[0]
            out = self._graph_metadata_out(B=B, T=fill_T)

        fused = fused_extend_sconv_metadata(
            B=B,
            T=fill_T,
            cache_indices=self._cache_indices,
            out=out,
            **his_kwargs,
        )
        if fused is not None:
            query_start_loc, has_initial_state, precomputed = fused
        else:
            # The unfused fallback allocates, so it can only serve the eager path;
            # on a captured shape it would leave the graph reading dead addresses.
            assert not on_graph_path, (
                "the fused extend metadata kernel declined a captured shape "
                f"(B={B}); its unfused fallback is not cuda-graph safe"
            )
            query_start_loc, has_initial_state = self._unfused_extend_metadata(
                forward_batch
            )
            precomputed = precompute_helion_extend_metadata(
                B=B,
                T=T,
                W=self.conv_state_len + 1,
                cache_indices=self._cache_indices,
                has_initial_state=has_initial_state,
                query_start_loc=query_start_loc,
            )
        if fill_T != T:
            # Hand the model the live extent; the address is what the captured
            # kernels hold onto.
            precomputed = SconvExtendMetadata(
                cache_mask=precomputed["cache_mask"],
                safe_idx=precomputed["safe_idx"],
                cu=precomputed["cu"],
                si=precomputed["si"][:T],
            )
        self._query_start_loc = query_start_loc
        self._has_initial_state = has_initial_state
        self._precomputed = precomputed

    def _unfused_extend_metadata(self, forward_batch: ForwardBatch):
        """Multi-kernel query_start_loc/has_initial_state prep; fused fallback
        only."""
        device = forward_batch.req_pool_indices.device
        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = forward_batch.spec_info.draft_token_num
            query_start_loc = torch.arange(
                0,
                (forward_batch.batch_size + 1) * draft_token_num,
                draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            has_initial_state = torch.ones(
                forward_batch.batch_size, dtype=torch.bool, device=device
            )
            return query_start_loc, has_initial_state
        query_start_loc = torch.zeros(
            forward_batch.batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        query_start_loc[1:] = forward_batch.extend_seq_lens.cumsum(dim=0)
        spec_info = forward_batch.spec_info
        if (
            isinstance(spec_info, EagleDraftExtendInput)
            and spec_info.num_front_tokens > 0
        ):
            has_initial_state = torch.zeros(
                forward_batch.batch_size, dtype=torch.bool, device=device
            )
        elif forward_batch.extend_prefix_lens is not None:
            has_initial_state = forward_batch.extend_prefix_lens > 0
        else:
            has_initial_state = (
                forward_batch.seq_lens[: forward_batch.batch_size]
                - forward_batch.extend_seq_lens
            ) > 0
        return query_start_loc, has_initial_state

    def _refresh_track_conv_indices(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):
        """Flattened input positions of the conv windows to snapshot for prefix
        caching.

        The conv layer keeps a sliding window of recent inputs, so after a prefill
        chunk the last ``conv_kernel - 1`` tokens of the tracked region must be
        saved. Tracking only reaches the last complete ``mamba_cache_chunk_size``
        boundary, matching where the chunk scan checkpoints.

        Under the padded prefill graph the captured gather reads all
        ``batch_size`` (= capture slot count) rows while the per-request track
        lengths are only as long as the live request count, so the tail rows are
        zeroed rather than left stale: every row the captured kernel may read has
        to index inside *this* replay's token buffer, which is exactly what the
        pre-migration in-graph build got from clamping the whole padded range to
        ``query_start_loc[-1] - 1``. Whether a tail row is read at all still comes
        down to the equally padded tail of ``mamba_track_mask``.
        """
        if forward_batch.mamba_track_mask is None:
            return
        rows = forward_batch.batch_size
        query_start_loc = self._query_start_loc
        # The graph path pads batch_size past the live per-request lengths.
        live = min(
            rows,
            forward_batch.mamba_track_seqlens.shape[0],
            forward_batch.extend_prefix_lens.shape[0],
        )

        lens_to_track = (
            forward_batch.mamba_track_seqlens[:live]
            - forward_batch.extend_prefix_lens[:live]
        )
        chunk_aligned = (
            lens_to_track // self.mamba_cache_chunk_size
        ) * self.mamba_cache_chunk_size
        start_indices = query_start_loc[:live] + chunk_aligned - self.conv_state_len

        if on_graph_path:
            assert rows <= self._graph_track_conv_indices.shape[0], (
                f"track-index buffer too small for a captured shape: rows={rows} "
                f"vs bound {self._graph_track_conv_indices.shape[0]}"
            )
            out = self._graph_track_conv_indices[:rows]
        else:
            out = torch.empty(
                (rows, self.conv_state_len),
                dtype=torch.int64,
                device=start_indices.device,
            )
        torch.add(
            start_indices.unsqueeze(-1).to(torch.int64),
            self._track_window_offsets,
            out=out[:live],
        )
        # Clamp with 1-element tensors, never [-1] scalars: a 0-d -> Python
        # conversion would sync (and is unrecordable under capture).
        torch.clamp(
            out[:live],
            min=self._track_index_floor,
            max=query_start_loc[-1:].to(torch.int64) - 1,
            out=out[:live],
        )
        if live < rows:
            out[live:].zero_()
        self._track_conv_indices = out

    # ------------------------------------------------------------------
    # speculative-decoding state commit
    # ------------------------------------------------------------------

    def commit_conv_state_after_mtp_verify(
        self,
        *,
        req_pool_indices: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
    ) -> None:
        """Commit the per-step conv windows saved during TARGET_VERIFY into the
        persistent conv caches at each request's last accepted step.

        The slot ids are re-derived from ``req_pool_indices`` instead of reusing
        the per-step ``self._cache_indices``: this runs after the forward context
        has exited, by which point another forward may already have refilled that
        buffer. Trusting it is what makes the generic mamba commit fail here
        (``dst_indices=15 vs step_indices=16``), so the authoritative request
        identity has to come in as an argument.
        """
        pool = self.req_to_token_pool
        scatter_mamba_states_after_mtp_verify(
            pool.get_speculative_mamba2_params_all_layers(),
            self._translate_mamba_indices(pool.get_mamba_indices(req_pool_indices)),
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )

    # ------------------------------------------------------------------
    # handle handed to the model
    # ------------------------------------------------------------------

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> InklingShortConvMetadata:
        """Return ``layer_id``'s conv-state handle for the current step.

        Everything but the per-layer pool view is already on ``self._*``, so this
        is a pure read: all conv layers in the step share one gather, one fused
        metadata launch and one track-index build.
        """
        del forward_batch
        return InklingShortConvMetadata(
            layer_cache=self.req_to_token_pool.mamba2_layer_cache(layer_id),
            cache_indices=self._cache_indices,
            query_start_loc=self._query_start_loc,
            has_initial_state=self._has_initial_state,
            precomputed=self._precomputed,
            track_conv_indices=self._track_conv_indices,
        )


class InklingShortConvHybridAttnBackend(ShortConvHybridAttnBackend):
    """Full-attention backend plus Inkling's conv-state sidecar.

    Inkling has NO linear-attention layers -- the short convs preprocess q/k and
    the attn/mlp output streams around softmax attention -- so every layer routes
    to the full-attention child and the sidecar is reached only through
    :meth:`conv_state_metadata`. That makes three departures from
    :class:`ShortConvHybridAttnBackend` necessary:

    * every layer is full attention, including the draft model's (the base's
      ``full_attn_layers = [0]`` draft-worker assumption does not hold);
    * DRAFT_EXTEND_V2 must still init the sidecar -- the draft model runs its own
      conv layers, unlike the mamba models the base's skip was written for;
    * the capability surface the model and the runners read off the full-attention
      backend has to stay visible through the wrapper;
    * the MTP-verify state commit is Inkling's own (see
      :meth:`update_mamba_state_after_mtp_verify`), not the generic mamba scatter.
    """

    def _is_full_attn(self, layer=None, layer_id: Optional[int] = None) -> bool:
        del layer, layer_id
        return True

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model=None,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """Commit accepted verify state through the conv sidecar.

        Overrides the generic mamba scatter, which sources its slot ids from
        ``linear_attn_backend.forward_metadata`` -- stale by the time this runs,
        since the commit happens after the forward context exits. The sidecar
        re-derives them from ``req_pool_indices`` instead.
        """
        del model
        assert req_pool_indices is not None, (
            "Inkling's conv-state commit needs req_pool_indices; the caller must "
            "pass the verify batch's request slots."
        )
        self.short_conv_backend.commit_conv_state_after_mtp_verify(
            req_pool_indices=req_pool_indices,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    @property
    def forward_metadata(self):
        # Unambiguous here: the conv sidecar's metadata is reached through
        # conv_state_metadata, so `forward_metadata` means the attention one (KV
        # write locations, the SWA loc translate).
        return self.full_attn_backend.forward_metadata

    @property
    def supports_ragged_verify_graph(self) -> bool:
        return self.full_attn_backend.supports_ragged_verify_graph

    @property
    def supports_full_cuda_graph_chunked_prefix(self) -> bool:
        return self.full_attn_backend.supports_full_cuda_graph_chunked_prefix

    def prepare_full_cuda_graph_chunked_prefix(self, *args, **kwargs):
        return self.full_attn_backend.prepare_full_cuda_graph_chunked_prefix(
            *args, **kwargs
        )

    def draft_extend_metadata_captured_in_graph(self) -> bool:
        return self.full_attn_backend.draft_extend_metadata_captured_in_graph()
