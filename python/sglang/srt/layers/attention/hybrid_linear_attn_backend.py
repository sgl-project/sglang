import logging
from typing import Optional, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
    track_mamba_states_if_needed,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.enable_unified_memory_pool = model_runner.enable_unified_memory_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        # Radix-prefix-cache mamba track DESTINATION slots. A SINGLE backend-owned
        # static buffer mirroring the runner's DecodeInputBuffers.mamba_track_indices
        # slot (shape (max_bs,), captured into the graph by pointer and refreshed
        # in-place each replay with the virtual->physical translate). Exposed as
        # ForwardMetadata.mamba_track_indices so the captured decode track-save
        # reads THIS buffer instead of mutating the InputBuffer registry slot.
        # Sized like the slot (NOT per-bs): the fb_view hands us the full slot
        # un-sliced and the consumers index it relative to its full length
        # ([:batch_size] for GDN, [-num_decodes:] for Mamba2), so a (max_bs,)
        # buffer keeps that indexing byte-identical to the old in-place path.
        self.mamba_track_indices_buf = None
        # GDN ReplaySSM (slice 1b): per-bs STATIC per-row write-cursor buffers
        # for cuda-graph. Allocated lazily in init_cuda_graph_state only when
        # --enable-linear-replayssm is set; stays None otherwise.
        self.replayssm_write_pos_list = None
        # GDN ReplaySSM (slice 2b): per-bs STATIC per-row force-flush buffers
        # for cuda-graph, parallel to replayssm_write_pos_list. Same lifetime
        # (None unless the flag is on).
        self.replayssm_force_flush_list = None
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _execute_deferred_mamba_cow_and_clear(self, forward_batch: ForwardBatch):
        """Run deferred clear/COW ops on the forward stream to avoid races."""
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
            or self.is_draft_worker
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            # mamba_pool is a pure PHYSICAL store; translate the (virtual, for the
            # unified memory pool) slot ids before zeroing — else clear_slots would zero
            # the WRONG physical slots. Identity for the non-unified memory pool.
            self.req_to_token_pool.mamba_pool.clear_slots(
                self._translate_mamba_indices(forward_batch.mamba_clear_indices)
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            ckpt_pool = getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)
            if ckpt_pool is not None:
                # int8 checkpoints: dequantize the cached state (src = int8 ckpt slot)
                # into the request's active bf16 slot (dst).
                ckpt_pool.load_to_active(
                    self.req_to_token_pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                # mamba_pool is a pure PHYSICAL store; translate both COW slot
                # ids virtual->physical (identity for the non-unified memory pool).
                self.req_to_token_pool.mamba_pool.copy_from(
                    self._translate_mamba_indices(forward_batch.mamba_cow_src_indices),
                    self._translate_mamba_indices(forward_batch.mamba_cow_dst_indices),
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        """Virtual->physical mamba slot-id translate (delegates to
        ``req_to_token_pool.translate_mamba_indices``: identity for the
        non-unified memory pool, allocator v2p for the unified memory pool).

        Apply EVERYWHERE mamba ids feed the SSM/conv kernels or the mamba-pool
        state ops (``clear_slots`` / ``copy_from``): the eager
        ``_forward_metadata`` AND the cuda-graph replay-prep (which copies the
        result into the captured graph's stable ``state_indices_list``). Missing
        it on the cuda-graph path feeds VIRTUAL ids to the captured kernels →
        wrong state slot → garbled output.
        """
        return self.req_to_token_pool.translate_mamba_indices(mamba_indices)

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        # Unified memory pool: get_mamba_indices returns *virtual* per-request ids;
        # translate to *physical* slot ids once per batch (no-op for the
        # non-shared HybridReqToTokenPool). MUST also happen on the cuda-graph
        # capture/replay metadata path — see `_translate_mamba_indices`.
        # Translate BEFORE the cuda-graph padding sentinel below, so the
        # virtual->physical gather reads only real ids; padded rows are then
        # poisoned to -1 (the mamba kernels skip -1).
        mamba_cache_indices = self._translate_mamba_indices(mamba_cache_indices)
        if forward_batch.mamba_track_indices is not None:
            # The *_track_* index derivations below index by mamba slot too
            # (speculative path; spec is off for the unified memory pool today). Identity
            # for the non-unified memory pool.
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
        _real_bs = forward_batch._original_batch_size
        if _real_bs is not None and _real_bs < mamba_cache_indices.shape[0]:
            mamba_cache_indices = mamba_cache_indices.clone()
            mamba_cache_indices[_real_bs:] = -1

        replayssm_write_pos = None
        replayssm_force_flush = None
        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
            # GDN ReplaySSM (slice 1a): the ring cursor is a per-slot
            # decode-position counter shared by ALL GDN layers in this forward.
            # Manage it exactly ONCE here (not per-layer): snapshot this step's
            # value for the batch's slots, hand it to the layers, then advance
            # the persistent buffer mod L for the next step.
            mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            write_pos_buf = (
                getattr(mamba_pool, "replayssm_write_pos", None)
                if mamba_pool is not None
                else None
            )
            if write_pos_buf is not None:
                slots = mamba_cache_indices.to(torch.long)
                # Padded rows carry slot == -1; clamp so the per-row gather stays
                # in-bounds (the kernel zeroes padded rows via state_idx < 0).
                safe_slots = slots.clamp(min=0)
                replayssm_write_pos = write_pos_buf[safe_slots].clone()
                L = mamba_pool.linear_replayssm_cache_len
                # KDA (per-K gate) ships without radix coordination for now: no
                # track-boundary force-flush, so the ring flushes only at the
                # natural write_pos == L-1 wrap. GDN keeps the radix-aligned
                # force-flush (slice 2b). Gate on the pool's recorded gate type.
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                # GDN ReplaySSM (slice 2b): per-row force-flush at the radix
                # track boundary. THE alignment: the radix mamba track snapshots
                # temporal[slot] when seq_lens_cpu % mamba_track_interval == 0
                # (extra_buffer: schedule_batch.prepare_for_decode builds
                # `mamba_track_mask = (seq_lens_cpu % mamba_track_interval == 0)`
                # off the SAME post-increment seq_lens_cpu used here). We source
                # the flush from the identical seq_lens + condition so the kernel
                # folds the ring into temporal[slot] on EXACTLY the steps the
                # snapshot reads it. seq_lens_cpu is the committed length AFTER
                # this decode token (incremented in prepare_for_decode before the
                # forward), matching the track. int32, one entry per batch row.
                if not is_kda:
                    force_flush_bool = self._replayssm_track_flush_mask(
                        forward_batch.seq_lens_cpu, bs
                    )
                    replayssm_force_flush = force_flush_bool.to(
                        device=self.device, dtype=torch.int32
                    )
                # Advance only the VALID (non-padded) slots. Scatter over the
                # unique valid slots to avoid duplicate-index races (padded rows
                # all clamp to slot 0, which a real row may also occupy). A
                # forced flush empties the ring -> next write_pos is 0 (same as
                # the natural wrap at write_pos == L-1).
                valid_mask = slots >= 0
                valid_slots = slots[valid_mask]
                if valid_slots.numel() > 0:
                    # Per-row "did this step flush?": natural wrap OR forced.
                    # (KDA has no forced flush -> force_flush is None -> pure wrap.)
                    flushed = replayssm_write_pos == (L - 1)
                    if replayssm_force_flush is not None:
                        flushed = flushed | (replayssm_force_flush != 0)
                    next_pos = torch.where(
                        flushed,
                        torch.zeros_like(replayssm_write_pos),
                        (replayssm_write_pos + 1) % L,
                    )
                    # Dedup valid slots; for duplicates a scatter picks one
                    # arbitrary row, but all rows of a given slot share the same
                    # write_pos/flush, so the value is identical regardless.
                    uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                    next_for_valid = next_pos[valid_mask]
                    new_vals = torch.empty(
                        uniq_slots.shape[0],
                        dtype=write_pos_buf.dtype,
                        device=write_pos_buf.device,
                    )
                    new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                    write_pos_buf[uniq_slots] = new_vals
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                # HybridLinearAttnBackend.init_forward_metadata calls all sub-backends
                # unconditionally, but DRAFT_EXTEND_V2 only runs full-attn layers in
                # the draft model, so mamba metadata can be skipped.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if self.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # retrieve_next_token is None during dummy run so skip tensor creation
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        has_mamba_track_mask = bool(
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        )

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            # Physical track destinations for the decode track-save (translated in
            # place above; None when tracking is off). cuda-graph supplies this via
            # the static backend buffer instead — see _replay_metadata.
            mamba_track_indices=forward_batch.mamba_track_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            has_mamba_track_mask=has_mamba_track_mask,
            replayssm_write_pos=replayssm_write_pos,
            replayssm_force_flush=replayssm_force_flush,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # seq_lens_cpu is unused by _replay_metadata for the non-target-verify
        # case but kept in the contract for compatibility.
        self.forward_metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
            in_capture=in_capture,
            mamba_track_indices=forward_batch.mamba_track_indices,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) and Mamba2 processes sequences in chunks
        of the chunk size (FLA_CHUNK_SIZE=64 for FLA, mamba_chunk_size for Mamba2).
        We only track the conv state up to the last complete chunk boundary (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The chunk size varies by model type:
        - FLA models: FLA_CHUNK_SIZE (64)
        - Mamba2 models: mamba_chunk_size (256)

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % chunk_size == 0): The to-cache state is stored in
            the last_recurrent_state.
          - Unaligned case (len % chunk_size != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example (assuming chunk_size=64):
        1. If chunked prefill length is < chunk_size, then only final state has value.
           In this case we cache `final` state.
        2. If chunked prefill length == chunk_size, then only final state has value.
           In this case we cache pos chunk_size, from `final` state.
        3. If chunked prefill length > chunk_size and < 2 * chunk_size, then both h and
           final state have value. We cache pos chunk_size from `h` state.
        4. If chunked prefill length == 2 * chunk_size, then both h and final state have
           value. We cache pos 2 * chunk_size from `final` state. Note `h` doesn't include
           the final position.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        # Move to CPU to avoid kernel launches for masking operations
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        if isinstance(self, Mamba2AttnBackend):
            num_h_states = extend_seq_lens // mamba_cache_chunk_size
        else:
            num_h_states = (extend_seq_lens - 1) // mamba_cache_chunk_size + 1

        # Calculate the starting offset for each sequence in the packed batch
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        is_aligned = (lens_masked % mamba_cache_chunk_size) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support mamba_cache_chunk_size % page size != 0, then need to modify this
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // mamba_cache_chunk_size
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def _replayssm_enabled(self) -> bool:
        """True iff --enable-linear-replayssm allocated the persistent ring cursor.

        The per-slot ``replayssm_write_pos`` buffer on MambaPool is None unless
        the flag is set, so it doubles as the on/off gate (same signal that
        ``_forward_metadata`` / ``GDNAttnBackend.forward_decode`` already use).
        """
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return False
        return getattr(mamba_pool, "replayssm_write_pos", None) is not None

    def _replayssm_track_flush_mask(
        self, seq_lens_cpu: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Per-row bool flush mask == the radix mamba-track snapshot condition.

        THE alignment (slice 2b): the radix mamba track snapshots temporal[slot]
        exactly when ``seq_lens_cpu % mamba_track_interval == 0`` (the same mask
        ``schedule_batch.prepare_for_decode`` builds for extra_buffer at
        ``mamba_track_mask = (seq_lens_cpu % mamba_track_interval == 0)``). Both
        read the SAME post-increment ``seq_lens_cpu`` (committed length AFTER
        this decode token), so the kernel force-flush fires on EXACTLY the steps
        the snapshot reads the checkpoint -- no off-by-one. Returns a CPU bool
        tensor of length ``bs`` (caller moves it to device as int32).
        """
        interval = get_global_server_args().mamba_track_interval
        if seq_lens_cpu is None:
            # Decode without a CPU seq-len mirror should not happen for the
            # supported (no_buffer, radix-on) config, but stay safe: never flush.
            return torch.zeros((bs,), dtype=torch.bool)
        mask = (seq_lens_cpu[:bs].to(torch.int64) % interval) == 0
        if mask.shape[0] < bs:
            pad = torch.zeros((bs - mask.shape[0],), dtype=torch.bool)
            mask = torch.cat([mask, pad])
        return mask.cpu()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        # GDN ReplaySSM (slice 1b): per-batch-size STATIC per-row write-cursor
        # buffers the kernel reads. Captured into the graph by pointer, so they
        # must be the SAME tensor objects refreshed in-place each replay. Sized
        # and indexed like state_indices_list ((i+1,), indexed [bs - 1]). Left
        # None when the flag is off so the dispatch falls through unchanged.
        self.replayssm_write_pos_list = [] if self._replayssm_enabled() else None
        # GDN ReplaySSM (slice 2b): static per-bs force-flush buffers, captured
        # by pointer and refreshed in-place per replay just like the write-pos
        # buffers. None when the flag is off.
        self.replayssm_force_flush_list = [] if self._replayssm_enabled() else None
        # Single (max_bs,) track-destination buffer mirroring the runner's slot.
        # int64 to match DecodeInputBuffers.mamba_track_indices + the track-save
        # kernel's int64 index load. Refreshed in-place by _replay_metadata.
        self.mamba_track_indices_buf = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            if self.replayssm_write_pos_list is not None:
                self.replayssm_write_pos_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            if self.replayssm_force_flush_list is not None:
                self.replayssm_force_flush_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Unified memory pool: virtual -> physical (no-op otherwise). The captured Mamba
        # kernels read state_indices_list as PHYSICAL slot ids, so we must
        # translate before copying — same as the eager _forward_metadata path.
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # GDN ReplaySSM (slice 1b): point at the STATIC per-bs write-cursor
        # buffer (no advance, no snapshot — capture records the pointer; its
        # zeros are overwritten in-place by _replay_metadata before each
        # replay). None when the flag is off. Same per-bs tensor object that
        # _replay_metadata refreshes, so the captured pointer stays valid.
        replayssm_write_pos = (
            self.replayssm_write_pos_list[bs - 1]
            if self.replayssm_write_pos_list is not None
            else None
        )
        # GDN ReplaySSM (slice 2b): point at the STATIC per-bs force-flush
        # buffer (same capture-by-pointer contract as write_pos; refreshed
        # in-place by _replay_metadata before each replay). None when off.
        replayssm_force_flush = (
            self.replayssm_force_flush_list[bs - 1]
            if self.replayssm_force_flush_list is not None
            else None
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and self.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrieve_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrieve_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
    ):
        if num_padding is None:
            if seq_lens_cpu is None:
                num_padding = 0
            else:
                num_padding = torch.count_nonzero(
                    seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
                )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Unified memory pool: virtual -> physical (no-op otherwise), reading the LIVE
        # v2p table in replay-prep, BEFORE the padding sentinel below. The
        # captured Mamba kernels read state_indices_list as PHYSICAL slot ids,
        # so this translate is required here.
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        # Radix-prefix-cache mamba track DESTINATION (`mamba_track_indices`):
        # refresh the captured backend-owned static buffer (the
        # state_indices_list sibling) in-place with the virtual->physical
        # translate, then expose it as ForwardMetadata.mamba_track_indices below.
        # The captured decode track-save reads THAT field, so the InputBuffer
        # registry slot we were handed stays read-only (never mutated). Runs at
        # capture too (records the static pointer + writes valid in-bounds dummy
        # slots) and at replay (live v2p -> physical). The translate is identity
        # for the non-unified memory pool, so this is a plain copy there.
        track_buf = None
        if mamba_track_indices is not None:
            track_buf = self.mamba_track_indices_buf
            track_buf[: len(mamba_track_indices)].copy_(
                self._translate_mamba_indices(mamba_track_indices)
            )
        # GDN ReplaySSM (slice 1b): refresh the STATIC per-row write cursor the
        # kernel reads, mirroring the eager snapshot-then-advance in
        # _forward_metadata but writing in-place into the captured per-bs buffer
        # so the graph's recorded pointer stays valid across replays. Done once
        # per forward here (out_graph host op), not per layer. Skipped during
        # capture (in_capture): capture runs on dummy slots, so advancing the
        # persistent counter then would corrupt real per-slot ring positions;
        # the captured buffer's contents are irrelevant at capture time anyway.
        replayssm_write_pos = None
        replayssm_force_flush = None
        if self.replayssm_write_pos_list is not None:
            mamba_pool = self.req_to_token_pool.mamba_pool
            write_pos_buf = mamba_pool.replayssm_write_pos
            static_wp = self.replayssm_write_pos_list[bs - 1]
            static_ff = self.replayssm_force_flush_list[bs - 1]
            # Hand the full captured per-bs buffers to the kernel, mirroring how
            # mamba_cache_indices = self.state_indices_list[bs - 1] is the full
            # (bs,) tensor; the kernel indexes them per decode row.
            replayssm_write_pos = static_wp
            replayssm_force_flush = static_ff
            if write_pos_buf is not None:
                # mamba_indices: this replay's per-row physical slots (padded
                # rows == -1, same tensor fed to state_indices_list above).
                slots = mamba_indices.to(torch.long)
                safe_slots = slots.clamp(min=0)
                # Snapshot THIS step's per-slot cursor into the captured buffer
                # the kernel reads (in-place copy_, never reassign the object).
                static_wp[: len(mamba_indices)].copy_(write_pos_buf[safe_slots])
                # GDN ReplaySSM (slice 2b): refresh the captured force-flush
                # buffer in-place from THIS step's seq_lens. THE alignment: same
                # `seq_lens_cpu % mamba_track_interval == 0` the radix track uses
                # (see _replayssm_track_flush_mask / schedule_batch). During
                # capture (seq_lens_cpu is None) leave it zeroed: capture content
                # is irrelevant and decode replays overwrite it below.
                force_flush_dev = None
                # KDA: no radix coordination -> leave static_ff zeroed and
                # force_flush_dev None so the advance below is a pure wrap,
                # matching the kernel (a zeroed force_flush flushes nothing).
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                if (
                    not is_kda
                    and forward_mode.is_decode_or_idle()
                    and seq_lens_cpu is not None
                ):
                    ff_mask = self._replayssm_track_flush_mask(seq_lens_cpu, bs)
                    force_flush_dev = ff_mask.to(device=self.device, dtype=torch.int32)
                    static_ff.copy_(force_flush_dev)
                else:
                    static_ff.zero_()
                if not in_capture:
                    L = mamba_pool.linear_replayssm_cache_len
                    # Advance only VALID (non-padded) slots. A forced flush
                    # empties the ring -> next write_pos is 0 (same as the
                    # natural wrap at write_pos == L-1). Use this step's snapshot
                    # cursor (write_pos_buf[safe_slots]) + the flush flag.
                    valid_mask = slots >= 0
                    valid_slots = slots[valid_mask]
                    if valid_slots.numel() > 0:
                        cur_pos = write_pos_buf[safe_slots]
                        flushed = cur_pos == (L - 1)
                        if force_flush_dev is not None:
                            flushed = flushed | (force_flush_dev != 0)
                        next_pos = torch.where(
                            flushed,
                            torch.zeros_like(cur_pos),
                            (cur_pos + 1) % L,
                        )
                        # Dedup; rows sharing a slot share write_pos+flush, so
                        # the scattered value is identical for either row.
                        uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                        next_for_valid = next_pos[valid_mask]
                        new_vals = torch.empty(
                            uniq_slots.shape[0],
                            dtype=write_pos_buf.dtype,
                            device=write_pos_buf.device,
                        )
                        new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                        write_pos_buf[uniq_slots] = new_vals
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and self.topk > 1:
            if (
                spec_info is not None
                and getattr(spec_info, "retrieve_next_token", None) is not None
            ):
                bs_without_pad = spec_info.retrieve_next_token.shape[0]
                self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_token
                )
                self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_sibling
                )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.

        The PHYSICAL track destinations come from the attention metadata
        (`self.forward_metadata.mamba_track_indices`), not forward_batch: under
        cuda-graph that is the backend-owned static buffer refreshed each replay,
        so the InputBuffer registry slot is never mutated.
        """
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                self.forward_metadata.mamba_track_indices,
                forward_batch.batch_size,
                check_freed_slots=self.enable_unified_memory_pool,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to the chunk size. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        if forward_metadata.has_mamba_track_mask:
            h = h.squeeze(0)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )

        if model_runner.server_args.enable_mamba_extra_buffer():
            assert (
                self.conv_states_shape[-1] < self.mamba_chunk_size
            ), f"{self.conv_states_shape[-1]=} should be less than {self.mamba_chunk_size}"
            assert (
                model_runner.server_args.mamba_track_interval >= self.mamba_chunk_size
            ), f"mamba_track_interval ({model_runner.server_args.mamba_track_interval}) must be >= mamba_chunk_size ({self.mamba_chunk_size})"

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
            in_capture=in_capture,
            mamba_track_indices=forward_batch.mamba_track_indices,
        )
        spec_info = forward_batch.spec_info
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        # The page-major envelope stores conv/SSM state in strided views; only the
        # stride-aware Triton causal-conv reads them correctly (CUDA causal_conv1d
        # garbles them). enable_page_major_kv_layout IS the signal that strides the
        # state, so the conv kernel follows it. A model may also force Triton
        # per-call (spec decoding's intermediate-state path in Nemotron-H / Granite).
        use_triton_causal_conv = (
            use_triton_causal_conv
            or get_global_server_args().enable_page_major_kv_layout
        )
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        mixer_out, intermediate_states = mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            forward_batch=forward_batch,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

        if forward_batch.mamba_track_mask is not None:
            if (
                intermediate_states is not None
                and forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                self._track_mamba_state_extend(
                    forward_batch,
                    intermediate_states,
                    layer_cache.temporal,
                    self.forward_metadata,
                )

            if self.forward_metadata.num_decodes > 0:
                num_decodes = self.forward_metadata.num_decodes
                track_mamba_states_if_needed(
                    layer_cache.conv[0],
                    layer_cache.temporal,
                    self.forward_metadata.mamba_cache_indices[-num_decodes:],
                    forward_batch.mamba_track_mask[-num_decodes:],
                    self.forward_metadata.mamba_track_indices[-num_decodes:],
                    num_decodes,
                    check_freed_slots=self.enable_unified_memory_pool,
                )

        return mixer_out

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]
        # Dispatcher aliases the full-attn backend's pool refs.
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool
        self.max_context_len = getattr(full_attn_backend, "max_context_len", None)
        self.needs_cpu_seq_lens = (
            full_attn_backend.needs_cpu_seq_lens
            or linear_attn_backend.needs_cpu_seq_lens
        )

    def _is_full_attn(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> bool:
        if layer is not None:
            layer_id = layer.layer_id
        assert layer_id is not None, "either layer or layer_id must be provided"
        return layer_id in self.full_attn_layers

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND_V2 only runs full-attn layers in the draft model,
            # so skip linear/mamba backend metadata which requires query_start_loc.
            self.full_attn_backend.init_forward_metadata(forward_batch)
            return
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Hybrid MLA models (Ring/Ling, Kimi-Linear) resolve this via
        # get_attn_backend(), which returns this wrapper; delegate to the
        # full-attn backend so its chunked/one-shot prefill metadata is planned.
        init = getattr(self.full_attn_backend, "init_mha_chunk_metadata", None)
        if init is not None:
            init(forward_batch, disable_flashinfer_ragged)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cpu_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cpu_graph_seq_len_fill_value()

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        is_linear_attn = not self._is_full_attn(layer, kwargs.get("layer_id"))

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]

        # Use fully fused kernel that handles masking internally
        # This avoids separate nonzero() and index_select() calls
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_state_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )
        # conv intermediate uses the deduplicated sliding-window (overlapping)
        # layout, so it needs the strided-read scatter variant.
        fused_conv_window_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )

        # Track indices used for tracking mamba states for prefix cache
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            # Use fully fused kernel for track scatter operations
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
            fused_conv_window_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
