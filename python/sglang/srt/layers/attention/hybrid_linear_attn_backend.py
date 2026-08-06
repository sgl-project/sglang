from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.kernels.ops.mamba.mamba_state_indices_triton import (
    fused_replay_state_indices,
)
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    scatter_mamba_states_after_mtp_verify,
    track_mamba_states_all_layers,
    track_mamba_states_if_needed,
)
from sglang.srt.configs.hybrid_arch import mamba2_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_exec, get_memory, get_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    from sglang.srt.layers.attention.verify_mask import VerifyMask

logger = logging.getLogger(__name__)


class MambaAttnBackendBase(AttentionBackend):
    # RecoverSSM (--gdn-mtp-cache-mode=none) is a GDN-only verify protocol;
    # GDNAttnBackend overrides this from the resolved mode. Declared at class
    # scope (not just in __init__) so the hybrid wrapper can read it off any
    # linear sidecar, including a MagicMock(spec=MambaAttnBackendBase) — spec
    # introspects the class, so an __init__-only attribute is invisible to it.
    _recover_ssm: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.enable_unified_memory = model_runner.server_args.enable_unified_memory
        # Fused replay-prep state-indices fast path (fused_replay_state_indices):
        # requires the static hybrid pool whose v2p translate is the identity —
        # the unified pool overrides translate_mamba_indices with an allocator
        # lookup that is not a flat table gather.
        self._fused_state_indices_ok = (
            str(self.device).startswith("cuda")
            and isinstance(self.req_to_token_pool, HybridReqToTokenPool)
            and type(self.req_to_token_pool).translate_mamba_indices
            is HybridReqToTokenPool.translate_mamba_indices
        )
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        # Static (max_bs,) track-dest buffer captured by pointer, refreshed in-place
        # each replay; the captured track-save reads this, not the InputBuffer slot.
        self.mamba_track_indices_buf = None
        # Per-bs static write-cursor / force-flush buffers for cuda-graph; None
        # unless --enable-linear-replayssm is set.
        self.replayssm_write_pos_list = None
        self.replayssm_force_flush_list = None
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        """Virtual->physical mamba slot-id translate (identity for the non-unified
        pool). Must run everywhere mamba ids feed the SSM/conv kernels or mamba-pool
        state ops, incl. the cuda-graph replay-prep copy into ``state_indices_list``."""
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
        # Translate virtual->physical BEFORE the padding sentinel below, so the
        # gather reads only real ids; padded rows are then poisoned to -1 (skipped).
        mamba_cache_indices = self._translate_mamba_indices(mamba_cache_indices)
        if forward_batch.mamba_track_indices is not None:
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
        # Resolve the tracked-row selection once per forward
        has_mamba_track_mask = bool(
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
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
            # The ring cursor is a per-slot decode counter shared by all GDN layers;
            # manage it once here (snapshot, hand to layers, advance mod L), not per-layer.
            # Gate on the linear_replayssm FLAG, not on cursor-tensor presence: the
            # spec-verify ring (--enable-linear-replayssm-spec) shares the write_pos
            # allocation but owns it exclusively via commit_gdn_replayssm_spec
            # (advance-by-accept-count once per verify step). Advancing it here as
            # well inserts one phantom/stale ring entry per step and cumulatively
            # poisons the reconstruction (degenerate repetition at 10k+ tokens).
            mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            write_pos_buf = (
                mamba_pool.replayssm_write_pos
                if mamba_pool is not None and mamba_pool.enable_linear_replayssm
                else None
            )
            if write_pos_buf is not None:
                slots = mamba_cache_indices.to(torch.long)
                # Padded rows carry slot == -1; clamp the gather in-bounds (kernel
                # zeroes padded rows via state_idx < 0).
                safe_slots = slots.clamp(min=0)
                replayssm_write_pos = write_pos_buf[safe_slots].clone()
                L = mamba_pool.linear_replayssm_cache_len
                # KDA has no radix coordination: flush only on the natural write_pos
                # == L-1 wrap. GDN adds the radix-aligned force-flush below.
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                # Force-flush on the radix track's seq_lens % mamba_track_interval
                # == 0 boundary so the ring folds into temporal[slot] when read.
                if not is_kda:
                    force_flush_bool = self._replayssm_track_flush_mask(
                        forward_batch.seq_lens_cpu, bs
                    )
                    replayssm_force_flush = force_flush_bool.to(
                        device=self.device, dtype=torch.int32
                    )
                # Advance only valid slots, scattered over unique slots (dup-index
                # race; padded rows clamp to 0); a forced flush -> next write_pos 0.
                valid_mask = slots >= 0
                valid_slots = slots[valid_mask]
                if valid_slots.numel() > 0:
                    flushed = replayssm_write_pos == (L - 1)
                    if replayssm_force_flush is not None:
                        flushed = flushed | (replayssm_force_flush != 0)
                    next_pos = torch.where(
                        flushed,
                        torch.zeros_like(replayssm_write_pos),
                        (replayssm_write_pos + 1) % L,
                    )
                    # Dedup: rows sharing a slot share write_pos/flush, so the
                    # scattered value is identical regardless of which row wins.
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
                # DRAFT_EXTEND_V2 runs only full-attn layers in the draft model;
                # skip mamba metadata.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                ragged_layout = forward_batch.spec_info.ragged_verify_layout
                if ragged_layout is not None:
                    # Compact ragged verify: variable per-request verify lens.
                    query_start_loc = ragged_layout.qo_indptr_device
                else:
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
                    # None during dummy run
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
                if has_mamba_track_mask:
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

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            # Physical track destinations (None when tracking off); cuda-graph
            # supplies this via the static backend buffer in _replay_metadata.
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
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
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # Plan-stream fixup: slot indices / static query_start_loc are
        # draft-independent, but tree verify (topk > 1) copies the
        # draft-produced tree links into the captured buffers on the plan
        # stream, racing the draft — re-copy after the stream join. Eager
        # verify reads the spec_info tensors directly; parent links are
        # derived from these buffers at execution time.
        if self.topk <= 1 or cuda_graph_bs is None:
            return
        if (
            not isinstance(spec_info, EagleVerifyInput)
            or spec_info.retrieve_next_token is None  # dummy / capture runs
        ):
            return
        bs_without_pad = spec_info.retrieve_next_token.shape[0]
        self.retrieve_next_token_list[cuda_graph_bs - 1][:bs_without_pad].copy_(
            spec_info.retrieve_next_token
        )
        self.retrieve_next_sibling_list[cuda_graph_bs - 1][:bs_without_pad].copy_(
            spec_info.retrieve_next_sibling
        )

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """Flattened input positions of conv states to track during extend (up to
        the last complete chunk boundary, mamba_track_mask rows only)."""
        conv_state_len = self.conv_states_shape[-1]

        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """src/dst indices to track SSM states for prefix caching: aligned seqs
        cache last_recurrent_state, unaligned cache intermediate `h` at the last
        chunk boundary."""
        mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size
        # CPU to avoid kernel launches for the masking ops
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        if isinstance(self, Mamba2AttnBackend):
            num_h_states = extend_seq_lens // mamba_cache_chunk_size
        else:
            num_h_states = (extend_seq_lens - 1) // mamba_cache_chunk_size + 1

        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        is_aligned = (lens_masked % mamba_cache_chunk_size) == 0

        # Aligned: last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Unaligned: intermediate state from h.
        # TODO: handle mamba_cache_chunk_size % page size != 0
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // mamba_cache_chunk_size
        )
        track_ssm_h_dst = dst_masked[not_aligned]

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
        """True iff --enable-linear-replayssm is on for this pool.

        Gate on the FLAG, not on ``replayssm_write_pos is not None``: the
        spec-verify ring (--enable-linear-replayssm-spec) also allocates the
        cursor tensor but owns it exclusively via commit_gdn_replayssm_spec.
        The decode-ring metadata machinery gated here (per-bs static cursor
        buffers, the per-replay snapshot + advance-by-one in _replay_metadata,
        and the decode-kernel ring rerouting downstream) must stay fully
        dormant for the spec ring.
        """
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return False
        return bool(mamba_pool.enable_linear_replayssm)

    def _replayssm_track_flush_mask(
        self, seq_lens_cpu: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Per-row (length bs) bool flush mask = the radix track's seq_lens_cpu %
        mamba_track_interval == 0, so force-flush and snapshot fire on the same
        steps (no off-by-one)."""
        interval = get_exec().mamba.mamba_track_interval
        if seq_lens_cpu is None:
            # Should not happen for the supported config; stay safe and never flush.
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
        # Per-bs static write-cursor / force-flush buffers, captured by pointer +
        # refreshed in-place each replay; sized like state_indices_list. None when off.
        self.replayssm_write_pos_list = [] if self._replayssm_enabled() else None
        self.replayssm_force_flush_list = [] if self._replayssm_enabled() else None
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
            ragged_layout = (
                spec_info.ragged_verify_layout if spec_info is not None else None
            )
            if ragged_layout is not None:
                # Ragged capture: qsl from the runner's synthetic layout.
                self.query_start_loc_list[bs - 1].copy_(ragged_layout.qo_indptr_device)
            else:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Captured Mamba kernels read state_indices_list as PHYSICAL ids; translate
        # before copying (no-op for non-unified pool).
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # Capture records the pointer to the static per-bs buffers; their zeros are
        # overwritten in-place by _replay_metadata before each replay. None when off.
        replayssm_write_pos = (
            self.replayssm_write_pos_list[bs - 1]
            if self.replayssm_write_pos_list is not None
            else None
        )
        replayssm_force_flush = (
            self.replayssm_force_flush_list[bs - 1]
            if self.replayssm_force_flush_list is not None
            else None
        )

        if forward_mode.is_target_verify() and self.topk > 1:
            # retrieve_* are None during capture, so skip the copy.
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
        if self._fused_state_indices_ok and self.replayssm_write_pos_list is None:
            # Single-launch fast path: mapping gather + padding sentinel + store
            # into the static buffer, plus zeroing padded req_pool_indices rows —
            # bit-identical to the reference chain below.
            mamba_indices = fused_replay_state_indices(
                req_pool_indices=req_pool_indices,
                mamba_index_mapping=self.req_to_token_pool.req_index_to_mamba_index_mapping,
                out_state_indices=self.state_indices_list[bs - 1],
                valid_bs=bs - int(num_padding),
                total_bs=bs,
            )
        else:
            # Make sure forward metadata is correctly handled for padding reqs
            req_pool_indices[bs - num_padding :] = 0
            mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
            # Translate using the LIVE v2p table BEFORE the padding sentinel below;
            # captured Mamba kernels read state_indices_list as PHYSICAL ids.
            mamba_indices = self._translate_mamba_indices(mamba_indices)
            mamba_indices[bs - num_padding :] = -1
            self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        # Refresh the static track-dest buffer in-place (translated); the captured
        # track-save reads it, leaving the handed-in InputBuffer slot read-only.
        # Hand out only the refreshed [:bs] prefix — Mamba2's track-save slices
        # [-num_decodes:], which on the full max_bs buffer binds the stale tail.
        track_buf = None
        if mamba_track_indices is not None:
            assert (
                len(mamba_track_indices) >= bs
            ), f"{len(mamba_track_indices)=} < {bs=}"
            track_buf = self.mamba_track_indices_buf[:bs]
            track_buf.copy_(self._translate_mamba_indices(mamba_track_indices[:bs]))
        # Refresh the static write cursor in-place (mirrors the eager
        # snapshot-then-advance). Skip the advance during capture: dummy slots
        # would corrupt real ring positions.
        replayssm_write_pos = None
        replayssm_force_flush = None
        if self.replayssm_write_pos_list is not None:
            mamba_pool = self.req_to_token_pool.mamba_pool
            write_pos_buf = mamba_pool.replayssm_write_pos
            static_wp = self.replayssm_write_pos_list[bs - 1]
            static_ff = self.replayssm_force_flush_list[bs - 1]
            # Hand the full captured per-bs buffers to the kernel; it indexes per row.
            replayssm_write_pos = static_wp
            replayssm_force_flush = static_ff
            if write_pos_buf is not None:
                # this replay's per-row physical slots (padded rows == -1)
                slots = mamba_indices.to(torch.long)
                safe_slots = slots.clamp(min=0)
                # Snapshot this step's cursor into the captured buffer in-place
                # (copy_, never reassign the object).
                static_wp[: len(mamba_indices)].copy_(write_pos_buf[safe_slots])
                # Refresh the force-flush buffer in-place from this step's seq_lens
                # (same condition as the radix track). Zeroed during capture.
                force_flush_dev = None
                # KDA: no radix coordination -> leave zeroed so the advance is a pure wrap.
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
                # Defense in depth: the decode-ring advance is only meaningful for
                # decode/idle forwards (mirrors the eager path's gating). A
                # TARGET_VERIFY replay must never advance the cursor.
                if not in_capture and forward_mode.is_decode_or_idle():
                    L = mamba_pool.linear_replayssm_cache_len
                    # Advance only valid (non-padded) slots; a forced flush empties
                    # the ring -> next write_pos 0, like the natural L-1 wrap.
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
                        # Dedup; rows sharing a slot share write_pos+flush.
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
            ragged_layout = (
                spec_info.ragged_verify_layout if spec_info is not None else None
            )
            if ragged_layout is not None:
                # Mamba kernels index dense [bs, N] scratch, so they need the
                # capped variant (see padded_to_bucket). Padding rows carry
                # mamba slot -1 and are skipped.
                if ragged_layout.bs != bs or ragged_layout.cap is None:
                    ragged_layout = ragged_layout.padded_to_bucket(
                        padded_bs=bs, cap=spec_info.draft_token_num
                    )
                self.query_start_loc_list[bs - 1].copy_(ragged_layout.qo_indptr_device)
            elif num_padding == 0:
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

    def _track_pools(self):
        """Full [num_layers, pool_size, ...] conv/ssm pools plus the pool index
        of the last mamba layer, for the fused all-layers track launch. None if
        the pool shape is not the expected layout."""
        cached = getattr(self, "_track_pools_cache", False)
        if cached is False:
            pools = None
            try:
                mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
                conv_pool = mamba_cache.conv[0]
                ssm_pool = mamba_cache.temporal
                last_pool_idx = max(self.req_to_token_pool.mamba_map.values())
                if (
                    conv_pool.dim() >= 3
                    and ssm_pool.dim() >= 3
                    and conv_pool.shape[0] == ssm_pool.shape[0] == last_pool_idx + 1
                ):
                    pools = (conv_pool, ssm_pool, last_pool_idx)
            except (AttributeError, IndexError):
                pools = None
            self._track_pools_cache = pools
            cached = pools
        return cached

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        layer_id: Optional[int] = None,
    ):
        """Copy decode conv/SSM states to track slots for prefix caching. Track
        dests come from the metadata (under cuda-graph: the static buffer), so the
        InputBuffer registry slot is never mutated.

        With a known layer_id, the per-layer launches collapse into ONE
        all-layers launch fired at the last mamba layer: the mask/src/dst
        indices are shared across layers and every layer's state is final by
        then, so the result is identical."""
        if forward_batch.mamba_track_mask is None:
            return
        if layer_id is not None:
            pools = self._track_pools()
            if pools is not None:
                conv_pool, ssm_pool, last_pool_idx = pools
                if self.req_to_token_pool.mamba_map.get(layer_id) != last_pool_idx:
                    return
                track_mamba_states_all_layers(
                    conv_pool,
                    ssm_pool,
                    cache_indices,
                    forward_batch.mamba_track_mask,
                    self.forward_metadata.mamba_track_indices,
                    forward_batch.batch_size,
                    check_freed_slots=self.enable_unified_memory,
                )
                return
        track_mamba_states_if_needed(
            conv_states,
            ssm_states,
            cache_indices,
            forward_batch.mamba_track_mask,
            self.forward_metadata.mamba_track_indices,
            forward_batch.batch_size,
            check_freed_slots=self.enable_unified_memory,
        )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """Copy extend SSM state at the last chunk boundary to track slots (source
        depends on chunk alignment; see `_init_track_ssm_indices`)."""
        if forward_metadata.has_mamba_track_mask:
            # Triton always returns h; FlashInfer returns it only when checkpoints
            # were requested. Aligned-only tracking reads the final state below.
            if forward_metadata.track_ssm_h_src.numel() > 0:
                assert h is not None
                h = h.squeeze(0)
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
        config = mamba2_config(model_runner.model_config)
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
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
        )
        spec_info = forward_batch.spec_info
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )
        # `forward` slices the track destinations from ([-num_decodes:])
        assert (
            self.forward_metadata.num_decodes == forward_batch.batch_size
        ), f"{self.forward_metadata.num_decodes=} != {forward_batch.batch_size=}"

    def init_forward_metadata(self, forward_batch: ForwardBatch):
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
        # Page-major stores state strided; only the stride-aware Triton causal-conv
        # reads it (CUDA causal_conv1d garbles it). A model may also force Triton.
        use_triton_causal_conv = (
            use_triton_causal_conv or get_memory().enable_page_major_kv_layout
        )
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        mixer_out, intermediate_states = mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

        if forward_batch.mamba_track_mask is not None:
            if intermediate_states is not None:
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
                    check_freed_slots=self.enable_unified_memory,
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
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool
        self.max_context_len = getattr(full_attn_backend, "max_context_len", None)
        self.needs_cpu_seq_lens = (
            full_attn_backend.needs_cpu_seq_lens
            or linear_attn_backend.needs_cpu_seq_lens
        )
        # Mirrors the linear sidecar's RecoverSSM mode; False for every non-GDN
        # sidecar, so the verify-commit dispatch below stays GDN-only by
        # construction rather than by inspecting pool buffers.
        self._recover_ssm = linear_attn_backend._recover_ssm
        # gdn_mtp_cache_mode=none recovery overlap: run the (eager) SSM-state
        # recovery on a dedicated side stream so its GPU work overlaps the next
        # step's draft-phase compute, then join before the next target forward
        # reads / overwrites the SSM pool. Lazily created on first recovery.
        self._recovery_stream: Optional[torch.cuda.Stream] = None
        self._recovery_event: Optional[torch.cuda.Event] = None
        self._recovery_event_pending: bool = False
        # Created with _recovery_stream and reused every step, because
        # _no_cache_mtp_recompute runs on the bs=1 host-latency critical path:
        # a bare Event.record() resolves the current stream in C++, where
        # torch.cuda.current_stream() costs ~40us of interpreter time
        # (_get_device_index -> _cuda_getDevice plus a fresh Stream wrapper),
        # and torch.cuda.stream() allocates a new StreamContext per call.
        self._recovery_join_event: Optional[torch.cuda.Event] = None
        self._recovery_stream_ctx: Optional[torch.cuda.StreamContext] = None
        # CUDA graph for FlashInfer recovery: replace the per-layer kernel
        # launches (their CPU dispatch cost) with a single graph replay. Captured
        # AND replayed on _recovery_stream so it still overlaps draft_extend + next
        # draft. Stable fixed-address buffers hold the per-step state/accepted-step
        # indices; contents are refreshed each step before replay.
        self._rec_state_idx_buf: Optional[torch.Tensor] = None
        self._rec_acc_steps_buf: Optional[torch.Tensor] = None
        # Interval-checkpoint (mamba radix track) recovery buffers, used only when
        # gdn_mtp_cache_mode=none runs with mamba radix tracking (extra_buffer):
        # a second recovery pass reconstructs the state at the exact track
        # boundary and writes it to the ping-pong track slot. Output slots come
        # from mamba_track_indices, accepted_steps from mamba_steps_to_track (both
        # -1-masked to reserved slot 0 / step 0 for non-crossing requests).
        self._rec_track_idx_buf: Optional[torch.Tensor] = None
        self._rec_track_steps_buf: Optional[torch.Tensor] = None
        self._rec_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Per-bucket graphs for the boundary pass, captured alongside _rec_graphs
        # only when mamba radix tracking is enabled. Empty when tracking is off or
        # capture failed, in which case the boundary runs eager on the side stream.
        self._rec_boundary_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Sorted bucket sizes for which a recovery graph was captured at warmup.
        # None until capture_recovery_graphs() succeeds; while None, recovery runs
        # eagerly on the side stream (the known-good overlap path).
        self._rec_capture_bs: Optional[list[int]] = None

    @property
    def data_type(self):
        # KV-cache dtype readers (e.g. the trtllm_mla fused-rope check) reach the
        # wrapper since split backends are wrapped once (#31439); the full-attn
        # side owns the KV cache, so its dtype is authoritative.
        return self.full_attn_backend.data_type

    @property
    def supports_ragged_verify_graph(self) -> bool:
        return (
            self.full_attn_backend.supports_ragged_verify_graph
            and self.linear_attn_backend.supports_ragged_verify_graph
        )

    def _is_full_attn(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> bool:
        if layer is not None:
            layer_id = layer.layer_id
        assert layer_id is not None, "either layer or layer_id must be provided"
        return layer_id in self.full_attn_layers

    def _ensure_recovery_stream(self):
        """Lazily create the recovery side stream and its per-step helpers.

        Both the warmup capture and the serving path enter through here so the
        cached join event / stream context can never be left unset by whichever
        one runs first.
        """
        if self._recovery_stream is not None:
            return
        self._recovery_stream = torch.cuda.Stream()
        self._recovery_event = torch.cuda.Event()
        self._recovery_join_event = torch.cuda.Event()
        self._recovery_stream_ctx = torch.cuda.stream(self._recovery_stream)

    def _wait_recovery_if_pending(self):
        """Join the side-stream gdn_mtp_cache_mode=none SSM recovery before a
        target forward reads the SSM pool. The wait also prevents the upcoming
        forward from overwriting the per-layer recovery stash while the
        side-stream recovery is still reading it."""
        if self._recovery_event_pending:
            # Event.wait() with no stream defaults to the current stream inside
            # C++ — same ordering as current_stream().wait_event(event), without
            # building the Python Stream wrapper.
            self._recovery_event.wait()
            self._recovery_event_pending = False

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if not in_capture and not forward_batch.forward_mode.is_draft_extend_v2():
            self._wait_recovery_if_pending()
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_in_graph(forward_batch)

    def on_after_cuda_graph_warmup(self):
        for attn_backend in self.attn_backend_list:
            attn_backend.on_after_cuda_graph_warmup()

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        # The mask lives on the full-attn child; the linear side reads none.
        return self.full_attn_backend.verify_mask

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # Plan-stream fixup after draft completes: forward to both children.
        # Sub-backends that cannot run under the plan stream keep the fail-loud
        # NotImplementedError base behavior.
        for attn_backend in self.attn_backend_list:
            attn_backend.update_verify_buffers_to_fill_after_draft(
                spec_info=spec_info, cuda_graph_bs=cuda_graph_bs
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND_V2 runs only full-attn layers in the draft model; skip
            # linear/mamba metadata (it requires query_start_loc).
            self.full_attn_backend.init_forward_metadata(forward_batch)
            return
        self._wait_recovery_if_pending()
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Hybrid MLA models resolve get_attn_backend() to this wrapper; delegate
        # so the full-attn backend plans its chunked-prefill metadata.
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
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """Update mamba states after MTP verify via a fused gather-scatter kernel.

        ``req_pool_indices`` serves implementations that must re-derive the state
        slot ids instead of reusing this step's ``forward_metadata``; the scatter
        below reads the metadata it just planned.
        """
        del req_pool_indices
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        req_pool = self.linear_attn_backend.req_to_token_pool
        mamba_caches = req_pool.get_speculative_mamba2_params_all_layers()

        # ReplaySSM-KDA: the accepted drafts live in the per-slot ring (written
        # during verify); no intermediate_ssm is allocated. Replay the accepted
        # prefix into `temporal` instead of scattering an intermediate state.
        # dspark/dflash call this method directly; the generic spec_utils commit
        # handles replayssm before reaching here (returns early), so this branch is
        # only hit by the direct callers. Chain layout only (topk <= 1), so
        # accept_lens == last_correct_step_indices + 1.
        mamba_pool = req_pool.mamba_pool
        if getattr(mamba_pool, "replayssm_is_kda", False):
            from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
                commit_kda_replayssm_after_verify,
            )

            commit_kda_replayssm_after_verify(
                spec_state=mamba_caches,
                state_batch_indices=state_indices_tensor,
                accept_lens=last_correct_step_indices + 1,
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                null_block_id=-1,
            )
            return

        # RecoverSSM (gdn_mtp_cache_mode=none) skips the per-draft SSM snapshots, so
        # rerun the recurrence from h_0 over the accepted prefix and roll back the
        # conv state. All other modes (full, ReplaySSM) go through the upstream
        # fused gather-scatter, which handles the ssm scatter, conv rollback, and
        # radix track indices. Keyed on the resolved mode, not on which buffers the
        # pool left unallocated -- ReplaySSM also skips intermediate_ssm, and which
        # of its ring buffers exist depends on its verify protocol. server_args
        # rejects none-mode alongside either ReplaySSM flag, so this is exclusive.
        if self._recover_ssm:
            conv_states = mamba_caches.conv[0]
            intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]
            # Mamba radix tracking (extra_buffer) in none-mode is supported on both
            # recovery paths (FlashInfer via native output_state_indices, Triton via
            # the output-index arg): a second boundary pass recomputes the tracked
            # prefix state, which none-mode does not cache.
            # Write h_K directly without materializing outputs; when tracking is
            # active, also recompute + write the boundary checkpoint state to the
            # ping-pong track slot.
            self._no_cache_mtp_recompute(
                accepted_steps=last_correct_step_indices,
                state_indices_tensor=state_indices_tensor,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )
            # Conv-state rollback uses the cached (deduplicated sliding-window)
            # conv windows via the strided-read scatter variant.
            fused_conv_window_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                state_indices_tensor,
                last_correct_step_indices,
            )
            # Conv boundary checkpoint: mirror the SSM boundary pass by writing the
            # tracked prefix conv window to the ping-pong track slot (same masked
            # scatter the full-mode path uses; step == -1 rows are skipped).
            if mamba_track_indices is not None:
                fused_conv_window_scatter_with_mask(
                    conv_states,
                    intermediate_conv_window_cache,
                    mamba_track_indices,
                    mamba_steps_to_track,
                )
        else:
            scatter_mamba_states_after_mtp_verify(
                mamba_caches,
                state_indices_tensor,
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
            )

    def _persist_kv(self, layer_id, conv_dims, b_rows, cache_steps):
        """FlashInfer-recovery post-conv (k, v) as zero-copy strided
        views of the persistent conv-out buffer, shaped [1, b_rows*cache_steps,
        H, D]. The buffer is [pool_size, cache_steps, conv_dim] token-major; k/v
        are the column slices [q_dim:q_dim+k_dim] / [q_dim+k_dim:...] (token
        stride = conv_dim, feature contiguous). The recovery kernel reads k/v via
        runtime strides, so no copy is materialized — and, because the view is
        pure host-side metadata over an address-stable buffer, nothing extra is
        captured into the recovery graph."""
        persist = self.linear_attn_backend._conv_out_persist[layer_id]
        q_dim, k_dim, v_dim, Hk, Dk, Hv, Dv = conv_dims
        n_tok = b_rows * cache_steps
        mixed = persist[:b_rows].reshape(n_tok, q_dim + k_dim + v_dim)
        k = mixed[:, q_dim : q_dim + k_dim].view(1, n_tok, Hk, Dk)
        v = mixed[:, q_dim + k_dim : q_dim + k_dim + v_dim].view(1, n_tok, Hv, Dv)
        return k, v

    def _fi_recovery_launch(
        self,
        n,
        stash_per_layer,
        pool,
        gated_delta_rule_mtp,
        init_idx_buf=None,
        out_idx_buf=None,
        acc_steps_buf=None,
    ):
        """Issue the per-layer FlashInfer recovery launches for the first ``n``
        rows, reading the stable index buffers and the [:n] stash slices. Shared
        by warmup graph capture, graph-less eager fallback, and the warm-compile
        pass. The stash is pre-shaped [pool_size, T, H] so [:n] is [n, T, H].
        FI recovery reads k/v from strided views of the persistent conv-out
        buffer, a/b from the stash.

        Defaults reconstruct h_K in place at the accepted length (init == output ==
        the request's working SSM slot). The interval-checkpoint boundary pass
        overrides ``out_idx_buf`` (ping-pong track slot) and ``acc_steps_buf`` (the
        per-request boundary step) while keeping ``init_idx_buf`` at the working
        slot, since h_0 lives there — so it MUST run before the in-place working
        recovery overwrites h_0 with h_K."""
        init_idx_buf = self._rec_state_idx_buf if init_idx_buf is None else init_idx_buf
        out_idx_buf = self._rec_state_idx_buf if out_idx_buf is None else out_idx_buf
        acc_steps_buf = (
            self._rec_acc_steps_buf if acc_steps_buf is None else acc_steps_buf
        )
        _init_idx = init_idx_buf[:n]
        # Reuse the SAME slice object when read and write indices alias: the FI
        # kernel picks its single-pool fast path via an identity check
        # (`output_state_indices is initial_state_indices`), and a second
        # `buf[:n]` would be a distinct view object that silently forces the
        # slower split-pool codegen (extra index LDG + separate write-side STGs).
        _out_idx = _init_idx if out_idx_buf is init_idx_buf else out_idx_buf[:n]
        _acc_steps = acc_steps_buf[:n]
        _T = self.linear_attn_backend._no_cache_draft_token_num
        for layer_id, stash in stash_per_layer.items():
            layer_ssm_states = pool.mamba2_layer_cache(layer_id).temporal
            # The FI path always stashes conv_dims (never k/v), so k/v come from
            # the persistent conv-out buffer. Reshape the [1, n*T, H, D] views to
            # the [n, T, H, D] the kernel expects (q == k: the kernel l2-norms
            # both).
            conv_dims = stash["conv_dims"]
            k_pv, v_pv = self._persist_kv(layer_id, conv_dims, n, _T)
            q_k = k_pv.view(n, _T, k_pv.shape[2], k_pv.shape[3])
            v_bat = v_pv.view(n, _T, v_pv.shape[2], v_pv.shape[3])
            gated_delta_rule_mtp(
                A_log=stash["A_log_f32"],
                a=stash["a"][:n],
                dt_bias=stash["dt_bias"],
                q=q_k,
                k=q_k,
                v=v_bat,
                b=stash["b"][:n],
                initial_state_source=layer_ssm_states,
                initial_state_indices=_init_idx,
                output_state_indices=_out_idx,
                accepted_steps=_acc_steps,
                disable_state_update=False,
                disable_output=True,
                use_qk_l2norm_in_kernel=True,
                scale=None,
                output=None,
            )

    def _rec_pad_to_bucket(self, batch_size: int) -> Optional[int]:
        """Smallest captured bucket >= batch_size, or None if no warmup graphs
        exist or batch_size exceeds the largest captured bucket (→ eager
        fallback)."""
        if not self._rec_capture_bs:
            return None
        i = bisect.bisect_left(self._rec_capture_bs, batch_size)
        if i == len(self._rec_capture_bs):
            return None
        return self._rec_capture_bs[i]

    def capture_recovery_graphs(self, capture_bs):
        """Warmup: capture one FlashInfer recovery graph per batch-size bucket, on
        the side stream, into a single shared mempool. Serving then pads the real
        batch up to a bucket and replays (no live capture → no per-step stall /
        global-mode crash). Must run AFTER a target_verify forward has allocated
        the per-layer stash at its final addresses (else this is a no-op and
        recovery falls back to eager side-stream launches).

        Padded rows (bucket - B) point at reserved SSM slot 0 (never allocated to
        a real request; free_slots starts at 1), so their output is harmless.
        """
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            fi_recovery_kernel,
        )

        use_fi_recovery = fi_recovery_kernel(self.linear_attn_backend) is not None
        if not use_fi_recovery:
            # Triton recovery or non-FI backend: no recovery graphs to capture;
            # recovery runs eager on the side stream.
            return
        stash_per_layer = getattr(self.linear_attn_backend, "_no_cache_stash", {})
        if not stash_per_layer:
            logger.warning(
                "[gdn_recovery] stash not allocated at capture time; "
                "recovery will run eagerly on the side stream (no cuda graph)."
            )
            return

        pool = self.linear_attn_backend.req_to_token_pool
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

        dev = pool.mamba2_layer_cache(next(iter(stash_per_layer))).temporal.device
        if self._rec_state_idx_buf is None:
            self._rec_state_idx_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
            self._rec_acc_steps_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
            # Boundary (interval-checkpoint) pass buffers: ping-pong track output
            # slots and the per-request boundary step. Allocated here so the
            # eager boundary launch reuses address-stable buffers.
            self._rec_track_idx_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
            self._rec_track_steps_buf = torch.empty(
                pool.size, dtype=torch.int32, device=dev
            )
        # Dummy indices → reserved slot 0 during capture (records only; the real
        # per-step indices are copied in before each replay).
        self._rec_state_idx_buf.fill_(0)
        self._rec_acc_steps_buf.fill_(0)
        self._rec_track_idx_buf.fill_(0)
        self._rec_track_steps_buf.fill_(0)

        # Capture the interval-checkpoint boundary pass too when mamba radix
        # tracking is on. Tracking is a server-level setting, so this is decided
        # once here; batches that carry no track indices simply skip the replay.
        capture_boundary = get_server_args().enable_mamba_extra_buffer()

        self._ensure_recovery_stream()

        buckets = sorted({int(b) for b in capture_bs if 0 < int(b) <= pool.size})
        if not buckets:
            return
        try:
            shared_pool = torch.cuda.graph_pool_handle()
            # Capture largest first so smaller graphs reuse the shared pool.
            for B in reversed(buckets):
                # Warm-compile this bucket's kernel + populate the kernel's per-B
                # argument defaults OUTSIDE capture (writes to reserved slot 0, so
                # harmless), so the capture itself is JIT-free and alloc-free.
                with torch.cuda.stream(self._recovery_stream):
                    self._fi_recovery_launch(
                        B, stash_per_layer, pool, gated_delta_rule_mtp
                    )
                self._recovery_stream.synchronize()

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    g,
                    pool=shared_pool,
                    stream=self._recovery_stream,
                    capture_error_mode="thread_local",
                ):
                    self._fi_recovery_launch(
                        B, stash_per_layer, pool, gated_delta_rule_mtp
                    )
                self._rec_graphs[B] = g

                if capture_boundary:
                    # Boundary variant: reads h_0 from the working slots, writes to
                    # the ping-pong track slots. Distinct output indices select the
                    # kernel's split-pool codegen, so this is a separate compiled
                    # variant -- warm-compile it outside capture as well (all index
                    # buffers hold 0 here, so it only touches the discard slot).
                    with torch.cuda.stream(self._recovery_stream):
                        self._fi_recovery_launch(
                            B,
                            stash_per_layer,
                            pool,
                            gated_delta_rule_mtp,
                            init_idx_buf=self._rec_state_idx_buf,
                            out_idx_buf=self._rec_track_idx_buf,
                            acc_steps_buf=self._rec_track_steps_buf,
                        )
                    self._recovery_stream.synchronize()

                    gb = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        gb,
                        pool=shared_pool,
                        stream=self._recovery_stream,
                        capture_error_mode="thread_local",
                    ):
                        self._fi_recovery_launch(
                            B,
                            stash_per_layer,
                            pool,
                            gated_delta_rule_mtp,
                            init_idx_buf=self._rec_state_idx_buf,
                            out_idx_buf=self._rec_track_idx_buf,
                            acc_steps_buf=self._rec_track_steps_buf,
                        )
                    self._rec_boundary_graphs[B] = gb
            self._rec_capture_bs = buckets
            logger.info(
                "[gdn_recovery] captured %d recovery cuda graphs (buckets=%s, "
                "boundary=%s)",
                len(buckets),
                buckets,
                capture_boundary,
            )
        except Exception as e:
            logger.warning(
                "[gdn_recovery] recovery cuda graph capture failed (%s); "
                "falling back to eager side-stream recovery.",
                e,
            )
            self._rec_graphs.clear()
            self._rec_boundary_graphs.clear()
            self._rec_capture_bs = None

    def _no_cache_mtp_recompute(
        self,
        accepted_steps: torch.Tensor,
        state_indices_tensor: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor] = None,
        mamba_steps_to_track: Optional[torch.Tensor] = None,
    ):
        """Recover accepted GDN SSM state for gdn_mtp_cache_mode=none.

        Replays the state-update recurrence over stashed post-conv k/v/a/b and
        writes h_{accepted_step} directly to the request's SSM state slot.

        When mamba radix tracking is active (extra_buffer), a request's accepted
        draft window may cross a track boundary. none-mode caches no intermediate
        state, so it also runs a second FlashInfer boundary pass that folds only
        ``mamba_steps_to_track`` steps from h_0 and writes the resulting boundary
        state to the ping-pong track slot ``mamba_track_indices``. That pass reads
        h_0 from the working slot, so it MUST run before the in-place working
        recovery overwrites h_0 with h_K. Requests that cross no boundary carry
        step == -1 and are redirected to reserved slot 0 / step 0 (harmless).
        """
        # Local imports to avoid a circular dependency at module load time.
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_recover_final_state,
        )

        stash_per_layer: dict = getattr(self.linear_attn_backend, "_no_cache_stash", {})
        if not stash_per_layer:
            # No GDN layer ran in cache_mode=none for this batch.
            return

        pool = self.linear_attn_backend.req_to_token_pool

        # Recovery runs outside the captured graph. Derive per-call sizes from
        # current tensors because the stash spans multiple batch-size captures.
        batch_size = accepted_steps.shape[0]
        draft_token_num = self.linear_attn_backend._no_cache_draft_token_num
        assert draft_token_num is not None, (
            "draft_token_num not cached — forward_extend was never called "
            "in target_verify mode before _mamba_verify_update."
        )
        actual_seq_len = batch_size * draft_token_num
        cache_steps = draft_token_num

        # On SM100+ with a bf16 state pool, recover via the FlashInfer MTP kernel
        # (PR #3502) — the cuda-graph path, reading k/v as strided views of the
        # persistent conv-out buffer. A non-FI / non-state-pool decode kernel uses
        # the Triton recurrence recover kernel with a flat k/v stash instead.
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            fi_recovery_kernel,
        )

        use_fi_recovery = fi_recovery_kernel(self.linear_attn_backend) is not None

        # The Triton recover kernel takes these as kernel arguments, so it needs
        # materialized int32 tensors (and keeps them alive for record_stream
        # below). The FlashInfer path only ever copies them into the stable
        # _rec_* int32 buffers, and Tensor.copy_ narrows on the way in — so the
        # conversion there is two eager dispatches of pure overhead, which a
        # host-latency-bound bs=1 decode pays in wall clock.
        state_idx_i32 = None
        accepted_steps_i32 = None
        if not use_fi_recovery:
            state_idx_i32 = state_indices_tensor.to(torch.int32).contiguous()
            accepted_steps_i32 = accepted_steps.to(torch.int32).contiguous()

        # Interval-checkpoint boundary pass runs on both recovery paths. FI uses
        # native output_state_indices; the Triton recover kernel takes a separate
        # output-index arg (default in-place). Per-step Triton boundary tensors are
        # built below (FI uses the stable _rec_track_* buffers instead).
        do_boundary = mamba_track_indices is not None
        track_out_i32 = None
        track_steps_i32 = None

        # One launch per GDN layer. Factored into a closure so it can run either
        # inline (CUDA graph capture) or on the side stream (eager overlap).
        B_bucket = None
        if use_fi_recovery:
            from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                gated_delta_rule_mtp,
            )

            B = batch_size
            # Stable fixed-address index buffers (normally allocated at warmup in
            # capture_recovery_graphs; allocate here for the no-warmup path).
            if self._rec_state_idx_buf is None:
                pool_size = pool.size
                dev = state_indices_tensor.device
                self._rec_state_idx_buf = torch.empty(
                    pool_size, dtype=torch.int32, device=dev
                )
                self._rec_acc_steps_buf = torch.empty(
                    pool_size, dtype=torch.int32, device=dev
                )
                self._rec_track_idx_buf = torch.empty(
                    pool_size, dtype=torch.int32, device=dev
                )
                self._rec_track_steps_buf = torch.empty(
                    pool_size, dtype=torch.int32, device=dev
                )
            # Smallest warmup-captured bucket >= B (None → eager, no graph).
            B_bucket = self._rec_pad_to_bucket(B)
            # Refresh stable buffers on the main stream before the side-stream
            # read (the join event recorded below orders after these copies).
            # Batched into one grouped _foreach_copy_: the buffers are int32 and
            # copy_ narrows the int64 sources itself, so no .to()/.contiguous()
            # temporaries are needed and the whole refresh is one dispatch.
            # Destinations are distinct buffers over disjoint rows, so the
            # regrouping _grouped_foreach_copy_ does by dtype pair is order-safe.
            from sglang.srt.model_executor.cuda_graph_buffer_registry import (
                _grouped_foreach_copy_,
            )

            refresh_dsts = [self._rec_state_idx_buf[:B], self._rec_acc_steps_buf[:B]]
            refresh_srcs = [state_indices_tensor, accepted_steps]
            crossed = None
            if do_boundary:
                # Boundary pass output slots come from mamba_track_indices, its
                # fold count from mamba_steps_to_track. Requests crossing no
                # boundary carry step == -1 → redirect to reserved slot 0 / step 0
                # (folds 0 steps, writing h_0 to the discard slot).
                #
                # clamp(min=0) is exactly where(steps >= 0, steps, 0) for every
                # integer input, so the fold count needs no explicit mask. The
                # output slot still does — leaving a non-crossing request pointed
                # at its real track slot would clobber a live ping-pong
                # checkpoint with h_0 — but that mask applies in place on the
                # stable buffer below, which avoids torch.where's output alloc.
                crossed = mamba_steps_to_track >= 0
                refresh_dsts += [
                    self._rec_track_idx_buf[:B],
                    self._rec_track_steps_buf[:B],
                ]
                refresh_srcs += [
                    mamba_track_indices,
                    mamba_steps_to_track.clamp(min=0),
                ]
            _grouped_foreach_copy_(refresh_dsts, refresh_srcs)
            if do_boundary:
                self._rec_track_idx_buf[:B].mul_(crossed)
            if B_bucket is not None and B_bucket > B:
                # Pad rows [B:bucket] → reserved slot 0 (never a real request, so
                # their recovery output is discarded harmlessly).
                self._rec_state_idx_buf[B:B_bucket].fill_(0)
                self._rec_acc_steps_buf[B:B_bucket].fill_(0)
                if do_boundary:
                    # The boundary graph is captured at bucket size, so its pad
                    # rows need a destination too → reserved slot 0.
                    self._rec_track_idx_buf[B:B_bucket].fill_(0)
                    self._rec_track_steps_buf[B:B_bucket].fill_(0)
        elif do_boundary:
            # Triton boundary: per-step regular tensors (no stable buffers). The
            # recover kernel already skips rows with a negative output index, so
            # non-crossing requests (step == -1) get output slot -1 (skipped on
            # store) instead of being redirected to a discard slot. h_0 is read
            # from the working slot (state_idx_i32) for all rows.
            track_steps_i32 = mamba_steps_to_track.to(torch.int32).contiguous()
            track_out_i32 = torch.where(
                track_steps_i32 >= 0,
                mamba_track_indices.to(torch.int32),
                torch.full_like(track_steps_i32, -1),
            ).contiguous()

        def _triton_recover_launch(init_indices, out_indices, acc_steps):
            # One recover launch per GDN layer. init_indices supplies h_0; the
            # folded state is written to out_indices (== init_indices for in-place
            # working recovery, the track slot for the boundary pass).
            for layer_id, stash in stash_per_layer.items():
                layer_cache = pool.mamba2_layer_cache(layer_id)
                layer_ssm_states = layer_cache.temporal  # [size+1, HV, V, K]

                conv_dims = stash.get("conv_dims")
                if conv_dims is not None:
                    # k/v are strided views of the persistent conv-out buffer
                    # ([1, actual_seq_len, H, D]); no stash copy.
                    k_recov, v_recov = self._persist_kv(
                        layer_id, conv_dims, batch_size, cache_steps
                    )
                else:
                    k_recov = stash["k"][:, :actual_seq_len]
                    v_recov = stash["v"][:, :actual_seq_len]

                fused_sigmoid_gating_delta_rule_recover_final_state(
                    A_log=stash["A_log"],
                    a=stash["a"][:actual_seq_len],
                    dt_bias=stash["dt_bias"],
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    k=k_recov,
                    v=v_recov,
                    b=stash["b"][:actual_seq_len],
                    initial_state_source=layer_ssm_states,
                    initial_state_indices=init_indices,
                    accepted_steps=acc_steps,
                    cache_steps=cache_steps,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=False,
                    output_state_indices=out_indices,
                )

        def _run_boundary():
            # Interval-checkpoint recovery: fold mamba_steps_to_track steps from h_0
            # (working slot) and write the boundary state to the ping-pong track
            # slot. Must run before _run_recovery() overwrites h_0 with h_K.
            if use_fi_recovery:
                self._fi_recovery_launch(
                    B,
                    stash_per_layer,
                    pool,
                    gated_delta_rule_mtp,
                    init_idx_buf=self._rec_state_idx_buf,
                    out_idx_buf=self._rec_track_idx_buf,
                    acc_steps_buf=self._rec_track_steps_buf,
                )
                return
            _triton_recover_launch(state_idx_i32, track_out_i32, track_steps_i32)

        def _run_recovery():
            if use_fi_recovery:
                # One launch per GDN layer, reading the stable index buffers.
                self._fi_recovery_launch(B, stash_per_layer, pool, gated_delta_rule_mtp)
                return

            # In-place working recovery: out == init == the working SSM slot.
            _triton_recover_launch(state_idx_i32, state_idx_i32, accepted_steps_i32)

        # During CUDA graph capture, recovery must stay on the capture stream
        # (it is normally eager / outside the graph; this is a safety guard).
        if torch.cuda.is_current_stream_capturing():
            if do_boundary:
                _run_boundary()
            _run_recovery()
            return

        # Eager path: overlap recovery on a dedicated side stream so its GPU
        # work hides behind the next step's draft-phase compute (which does not
        # touch the SSM pool). The next target forward joins on _recovery_event
        # (see _wait_recovery_if_pending) before reading / overwriting the SSM
        # pool & stash.
        self._ensure_recovery_stream()
        # Recovery must observe the stash writes and (FI) stable-buffer copies
        # issued on the current stream during this step's verify forward.
        # record() with no argument resolves the current stream in C++, so this
        # pair is wait_stream(current_stream()) without the Python round trip.
        self._recovery_join_event.record()
        self._recovery_stream.wait_event(self._recovery_join_event)

        if use_fi_recovery and B_bucket is not None:
            # Replay the warmup-captured graph for this bucket on the side stream
            # (pure async launch — overlaps draft_extend + next draft). No capture
            # ever happens on the live path. The boundary pass replays its own
            # captured graph (falling back to eager launches if it was not
            # captured) BEFORE the working replay, so it reads h_0 before the
            # working graph overwrites it with h_K.
            with self._recovery_stream_ctx:
                if do_boundary:
                    boundary_graph = self._rec_boundary_graphs.get(B_bucket)
                    if boundary_graph is not None:
                        boundary_graph.replay()
                    else:
                        _run_boundary()
                self._rec_graphs[B_bucket].replay()
        else:
            # No warmup graph (Triton fallback, B beyond the largest captured
            # bucket, or capture disabled/failed): eager recovery on the side
            # stream — the known-good overlap path.
            with self._recovery_stream_ctx:
                if do_boundary:
                    _run_boundary()
                _run_recovery()

        self._recovery_event.record(self._recovery_stream)
        # FlashInfer path reads only the long-lived stable buffers on the side
        # stream, so the per-step state_idx_i32 / accepted_steps_i32 need no
        # record_stream. Triton path reads them directly on the side stream:
        # wait_stream orders but does not extend their lifetime, so pin the
        # blocks until the side-stream recovery is done.
        if not use_fi_recovery:
            state_idx_i32.record_stream(self._recovery_stream)
            accepted_steps_i32.record_stream(self._recovery_stream)
            if do_boundary:
                # Triton boundary reads these per-step tensors on the side stream.
                track_out_i32.record_stream(self._recovery_stream)
                track_steps_i32.record_stream(self._recovery_stream)
        self._recovery_event_pending = True


class ShortConvHybridAttnBackend(HybridLinearAttnBackend):
    """HybridLinearAttnBackend variant for short-conv hybrid models (ZAYA1 CCA,
    LFM2 short conv).

    The linear sidecar is a :class:`ShortConvAttnBackend
    <sglang.srt.layers.attention.linear.short_conv_backend.ShortConvAttnBackend>`
    that owns the per-request conv-state plumbing. The model's conv module
    reaches it via :meth:`conv_state_metadata` (``get_attn_backend()`` returns
    this wrapper) and runs its own conv kernel against the returned handle, so
    the model definition holds no pool access. The sidecar is never reached
    through the full-vs-linear ``forward_decode`` / ``forward_extend`` dispatch.
    """

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        short_conv_backend: MambaAttnBackendBase,
        full_attn_layers: list,
    ):
        # Register short_conv_backend as the linear sidecar so it rides in
        # attn_backend_list and inherits the metadata / cuda-graph fan-out.
        super().__init__(full_attn_backend, short_conv_backend, full_attn_layers)
        self.short_conv_backend = short_conv_backend

    def conv_state_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.short_conv_backend.conv_state_metadata(layer_id, forward_batch)
