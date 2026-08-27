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

A :mod:`~sglang.srt.layers.attention.linear.short_conv_backend` sidecar. Four short
convs per decoder layer keep per-request conv state in the centralized
``MambaPool``; the model reaches this via :meth:`conv_state_metadata` for the
step's metadata and :meth:`sconv_state` for a layer's own conv stream, never
through ``forward_decode`` / ``forward_extend``.

On top of what :class:`ShortConvAttnBackend` owns, Inkling's kernels take a
precomputed ``cache_mask`` / ``safe_idx`` / ``cu`` / ``si`` set plus the extend
``track_conv_indices``. All of it is step-global, so it is resolved once per step
and shared by every conv module in the step (a decoder layer holds four).

The hook split is a decode-latency decision. ``init_forward_metadata_in_graph`` is
*recorded* into the decode / target-verify / draft-extend graphs, so prep placed
there replays for free; out of graph it lands on the per-step CPU path that a
captured graph exists to avoid. ``init_forward_metadata_out_graph`` therefore takes
only what a phase cannot record: full-cuda-graph prefill (no in-graph hook) and the
unified pool's slot translate. Prep consequently sits outside the graph, so every
tensor a captured kernel reads lives in a graph-static buffer refilled in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
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
from sglang.srt.runtime_context import (
    get_exec,
    get_spec,
    mamba_cache_chunk_size,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class InklingShortConvMetadata(msgspec.Struct):
    """The step's conv-state metadata, filled during metadata prep. On the graph
    path every tensor here is a static buffer refilled in place.
    """

    cache_indices: Optional[torch.Tensor] = None  # per-request slot ids, int32
    query_start_loc: Optional[torch.Tensor] = None  # cu-seqlens, int32
    has_initial_state: Optional[torch.Tensor] = None  # "resumes a cached prefix"
    precomputed: Optional[SconvExtendMetadata | SconvDecodeMetadata] = None
    # [B, conv_kernel - 1] input positions whose conv window feeds the prefix
    # cache. Extend only, and only when tracking is on.
    track_conv_indices: Optional[torch.Tensor] = None


class InklingShortConvAttnBackend(ShortConvAttnBackend):
    """Owns Inkling's per-step short-conv state plumbing (see module docstring)."""

    # int32 matches the pool and the conv kernels; an int64 view would re-run a
    # narrowing cast in every conv layer.
    cache_indices_dtype: torch.dtype = torch.int32
    # Fully device-side extend path, so the ZAYA1-style host mirrors would only
    # add a device->host sync per step.
    needs_extend_host_mirrors: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # Pool-wide, bound at pool construction: conv[stream] is
        # [n_layers, n_slots, conv_kernel - 1, conv_dim].
        self._mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        self.conv_state_len: int = self.conv_states_shape[2]
        self.mamba_cache_chunk_size = mamba_cache_chunk_size()
        # A plain table lookup is recordable; the unified pool's translate is an
        # allocator lookup and must stay in the out-of-graph replay prep.
        self._slot_gather_recordable = (
            type(self.req_to_token_pool).translate_mamba_indices
            is HybridReqToTokenPool.translate_mamba_indices
        )

        self.sconv_metadata = InklingShortConvMetadata()

        self._alloc_graph_buffers()

    def _alloc_graph_buffers(self):
        """Sized from the CONFIGURED capture shapes, once, never reallocated:
        growing a buffer after a graph captured it moves the address that graph
        reads, and prefill captures before the decode runner reports its bounds."""
        cuda_graph_config = get_exec().graph.cuda_graph_config
        decode_bs: list[int] = []
        prefill_tokens: list[int] = []
        decode_max_bs = 0
        if cuda_graph_config is not None:
            decode_bs = list(cuda_graph_config.decode.bs or [])
            prefill_tokens = list(cuda_graph_config.prefill.bs or [])
            decode_max_bs = cuda_graph_config.decode.max_bs or 0
        draft_token_num = get_spec().speculative_num_draft_tokens or 1
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
        # Inert track fields for graph capture: a capture warmup batch that
        # carries no tracking metadata must still LAUNCH the track scatter
        # (all rows masked off), or the python-level `if` specializes the
        # scatter out of the captured graph.
        self._graph_track_inert_mask = torch.zeros(max_bs, dtype=torch.bool, device=dev)
        self._graph_track_inert_indices = torch.zeros(
            max_bs, dtype=torch.int64, device=dev
        )
        self._graph_track_inert_seqlens = torch.zeros(
            max_bs, dtype=torch.int64, device=dev
        )
        # Same address-stability requirement; the base only sizes this from
        # init_cuda_graph_state, which the prefill graph never calls.
        self._alloc_cache_indices_buf(max_bs)
        self._track_window_offsets = torch.arange(
            self.conv_state_len, dtype=torch.int64, device=dev
        )
        self._track_index_floor = torch.zeros((1,), dtype=torch.int64, device=dev)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        # Fail now, not at the first replay, if a phase outgrew __init__'s bounds.
        self._graph_metadata_out(B=max_bs, T=max_num_tokens)

    def _graph_metadata_out(self, *, B: int, T: int) -> SconvMetadataOut:
        """Graph-static destinations sliced to this step. Asserts rather than
        allocating, which would leave captured kernels on a dead address."""
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

    def _forward_metadata(self, forward_batch: ForwardBatch) -> ForwardMetadata:
        """Slot ids only. Leaner than ``MambaAttnBackendBase._forward_metadata``
        on purpose: no SSM state (whose track prep also syncs), a conv window on a
        different axis, and ``query_start_loc`` from the fused kernel."""
        return ForwardMetadata(
            query_start_loc=None,
            mamba_cache_indices=self._translate_mamba_indices(
                self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
            ),
        )

    def _reset_step_state(self):
        super()._reset_step_state()
        self.sconv_metadata = InklingShortConvMetadata()

    @staticmethod
    def _phase_records_metadata(forward_batch: ForwardBatch) -> bool:
        """True when this phase's runner records ``init_forward_metadata_in_graph``
        (decode / target-verify / draft-extend do; full-cuda-graph prefill does
        not)."""
        mode = forward_batch.forward_mode
        return (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager path: nothing downstream is captured, so kernels may allocate."""
        self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=False)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        """Whatever this phase cannot record. Runs before EVERY replay, so the
        common path is one predicate and a return."""
        del in_capture
        if self._phase_records_metadata(forward_batch):
            if not self._slot_gather_recordable:
                self._prepare_slot_indices(forward_batch)
            return
        self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=True)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Recorded into the graph: writes the static destinations, so the launches
        refill them every replay and never allocate (the hook's contract)."""
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
            # One launch; the base's gather-then-copy would add a second recorded
            # kernel per step (the pool's table already has this dtype). No PAD
            # sentinel needed: MambaSlotAllocator.clear reserves slot 0 as the dummy
            # write target, so zero-filled padded rows already land there.
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
        self.sconv_metadata.cache_indices = self._cache_indices
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
        md = self.sconv_metadata
        (
            md.query_start_loc,
            md.has_initial_state,
            md.precomputed,
        ) = fused_decode_sconv_metadata(
            B=B,
            cache_indices=self._cache_indices,
            out=self._graph_metadata_out(B=B, T=B) if on_graph_path else None,
        )

    def _refresh_extend_metadata(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):
        """One fused launch; unfused fallback off-CUDA / past the batch bound."""
        B = forward_batch.batch_size
        if forward_batch.forward_mode.is_target_verify():
            # target_verify has no extend_seq_lens/extend_prefix_lens; the lens are
            # a constant draft_token_num per request.
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
                # Boundary-KV fix: run conv fresh so warm-up rows rebuild the window.
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

        # Captured kernels bake their token extent at CAPTURE (the prefill bucket)
        # while replay reports only the live count, so fill the WHOLE seq-index
        # buffer; the kernel clamps the tail to B - 1. target_verify's
        # B * draft_token_num is exact either way.
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
            # The unfused fallback allocates, so it cannot serve a captured shape.
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
            # Hand back the live extent; only the address matters to the graph.
            precomputed = SconvExtendMetadata(
                cache_mask=precomputed["cache_mask"],
                safe_idx=precomputed["safe_idx"],
                cu=precomputed["cu"],
                si=precomputed["si"][:T],
            )
        md = self.sconv_metadata
        md.query_start_loc = query_start_loc
        md.has_initial_state = has_initial_state
        md.precomputed = precomputed

    def _unfused_extend_metadata(self, forward_batch: ForwardBatch):
        """Unfused query_start_loc / has_initial_state prep; fallback only."""
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
        """Input positions of the conv windows to snapshot for prefix caching: the
        last ``conv_kernel - 1`` tokens up to the last complete
        ``mamba_cache_chunk_size`` boundary.

        The padded tail is ZEROED, not left stale: the captured gather reads all
        ``batch_size`` rows while the track lengths cover only live requests, and
        every row it may read must index inside *this* replay's token buffer.
        """
        if forward_batch.mamba_track_mask is None:
            if not on_graph_path:
                self.sconv_metadata.track_conv_indices = None
                return
            # Graph capture must still launch the track scatter (see the inert
            # buffers in __init__).
            rows = forward_batch.batch_size
            forward_batch.mamba_track_mask = self._graph_track_inert_mask[:rows]
            forward_batch.mamba_track_indices = self._graph_track_inert_indices[:rows]
            forward_batch.mamba_track_seqlens = self._graph_track_inert_seqlens[:rows]
        rows = forward_batch.batch_size
        query_start_loc = self.sconv_metadata.query_start_loc
        # A capture batch is built directly rather than through
        # ForwardBatch.init_new, so it carries no prefix lengths; its rows are
        # masked off, so zeros keep the indices in bounds. A replayed or eager
        # batch always has them, and must still fail rather than track against
        # an invented prefix.
        prefix_lens = forward_batch.extend_prefix_lens
        if prefix_lens is None and on_graph_path:
            prefix_lens = torch.zeros_like(forward_batch.mamba_track_seqlens)
        live = min(
            rows,
            forward_batch.mamba_track_seqlens.shape[0],
            prefix_lens.shape[0],
        )

        lens_to_track = forward_batch.mamba_track_seqlens[:live] - prefix_lens[:live]
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
        # 1-element tensors, never [-1]: a 0-d -> Python conversion would sync.
        torch.clamp(
            out[:live],
            min=self._track_index_floor,
            max=query_start_loc[-1:].to(torch.int64) - 1,
            out=out[:live],
        )
        if live < rows:
            out[live:].zero_()
        self.sconv_metadata.track_conv_indices = out

    def commit_conv_state_after_mtp_verify(
        self,
        *,
        req_pool_indices: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
    ) -> None:
        """Commit the TARGET_VERIFY conv windows at each request's last accepted step.

        Slot ids come from ``req_pool_indices``, not the per-step
        ``self._cache_indices``: this runs after the forward context exits, so that
        buffer may already belong to a later forward.
        """
        pool = self.req_to_token_pool
        scatter_mamba_states_after_mtp_verify(
            pool.get_speculative_mamba2_params_all_layers(),
            self._translate_mamba_indices(pool.get_mamba_indices(req_pool_indices)),
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> InklingShortConvMetadata:
        """The step's metadata: resolved once during prep, so this is a pure read."""
        del layer_id, forward_batch
        return self.sconv_metadata

    def sconv_state(self, *, layer_id: int, stream: int) -> torch.Tensor:
        """``layer_id``'s conv state for one ``SconvType`` stream."""
        pool_layer = self.req_to_token_pool.mamba2_layer_index(layer_id)
        return self._mamba_cache.conv[stream][pool_layer]

    def sconv_intermediate_window(self, *, layer_id: int, stream: int) -> torch.Tensor:
        """One stream's per-draft-token conv windows. TARGET_VERIFY only."""
        pool_layer = self.req_to_token_pool.mamba2_layer_index(layer_id)
        return self._mamba_cache.intermediate_conv_window[stream][pool_layer]


class InklingShortConvHybridAttnBackend(ShortConvHybridAttnBackend):
    """Full-attention backend plus Inkling's conv-state sidecar.

    Inkling has NO linear-attention layers, so every layer routes to the
    full-attention child and the sidecar is reached only through its metadata and
    conv-state accessors. Four departures from
    :class:`ShortConvHybridAttnBackend`: every layer is full attention (including
    the draft's, so the base's ``full_attn_layers = [0]`` does not hold);
    DRAFT_EXTEND_V2 still inits the sidecar (the draft runs its own convs, unlike
    the mamba models the base's skip was written for); the full-attention backend's
    capability surface stays visible through the wrapper; and the MTP-verify commit
    is Inkling's own, not the generic mamba scatter.
    """

    def sconv_state(self, *, layer_id: int, stream: int) -> torch.Tensor:
        return self.short_conv_backend.sconv_state(layer_id=layer_id, stream=stream)

    def sconv_intermediate_window(self, *, layer_id: int, stream: int) -> torch.Tensor:
        return self.short_conv_backend.sconv_intermediate_window(
            layer_id=layer_id, stream=stream
        )

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
        """Overrides the generic mamba scatter, which sources slot ids from
        ``forward_metadata`` -- stale once the forward context has exited."""
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
        # The sidecar's is reached via conv_state_metadata, so this is the attention
        # one (KV write locs, the SWA loc translate).
        return self.full_attn_backend.forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value):
        self.full_attn_backend.forward_metadata = value

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
