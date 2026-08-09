from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    """

    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    extended_lens = prefix_lens + int(draft_token_num)
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    keep_slots.clamp_(min=0, max=int(draft_token_num))
    return keep_slots


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    When draft windowing is disabled, `draft_seq_lens` matches the committed target
    prefix length already materialized in the draft KV cache. When windowing is
    enabled, `draft_seq_lens` is the logical resident length in the draft worker's
    compact req-to-token mapping. In paged mode this may exceed the requested
    window by up to `page_size - 1` so the local page table remains valid. `ctx_lens`
    tracks newly committed target tokens that still need draft KV materialization.
    """

    # Current token to start the next DFlash block (one per request).
    verified_id: torch.Tensor

    # Flattened context features for tokens that need to be appended into the draft cache.
    # Shape: [sum(ctx_lens), K * hidden_size], where K is the number of target-layer
    # hidden-state features concatenated per token (len(dflash_config.target_layer_ids),
    # or default K == draft_num_layers for existing checkpoints).
    target_hidden: torch.Tensor

    # Context lengths per request, used to slice `target_hidden`. Device tensor (int32).
    ctx_lens: torch.Tensor

    # How many committed tokens are visible to the draft worker per request.
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Draft state does not change token accounting.
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        # Rebuild target_hidden for the filtered batch using vectorized indexing.
        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        start = offsets[:-1]
        seg_start = start[new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)

        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

    # Per-request candidate lengths for variable-length verification.
    # When None, all requests use draft_token_num. Shape: [bs] int32.
    candidate_lens: torch.Tensor | None = None

    # True when tokens are packed (variable-length) rather than rectangular.
    _packed: bool = False

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)
        # Detect packed layout: token count equals sum(cl) and width is max(cl).
        # Rect layout uses bs*L tokens with L possibly < max(cl); do not treat as packed.
        if self.candidate_lens is not None:
            max_cl = int(self.candidate_lens.max().item())
            sum_cl = int(self.candidate_lens.sum().item())
            if (
                self.draft_token_num < max_cl
                and self.draft_token.numel() == sum_cl
            ):
                self._packed = True
                self._max_cl = max_cl

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        # Packed layout: alloc bs*ntpb slots but assign only candidate_lens per request.
        if self._packed:
            cl = self.candidate_lens
            end_offset = batch.seq_lens + cl
            if page_size == 1:
                batch.out_cache_loc = alloc_token_slots(
                    batch.tree_cache, len(batch.input_ids)
                )
            else:
                prefix_lens = batch.seq_lens
                prefix_lens_cpu = batch.seq_lens_cpu
                end_offset_cpu = prefix_lens_cpu + cl.cpu()
                last_loc = get_last_loc(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    prefix_lens,
                )
                batch.out_cache_loc = alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    end_offset,
                    end_offset_cpu,
                    last_loc,
                    len(batch.input_ids),
                )
                self.last_loc = last_loc
        else:
            if page_size == 1:
                batch.out_cache_loc = alloc_token_slots(
                    batch.tree_cache, len(batch.input_ids)
                )
                end_offset = batch.seq_lens + self.draft_token_num
            else:
                prefix_lens = batch.seq_lens
                prefix_lens_cpu = batch.seq_lens_cpu
                end_offset = prefix_lens + self.draft_token_num
                end_offset_cpu = prefix_lens_cpu + self.draft_token_num
                last_loc = get_last_loc(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    prefix_lens,
                )
                batch.out_cache_loc = alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    end_offset,
                    end_offset_cpu,
                    last_loc,
                    len(batch.input_ids),
                )
                self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 0:
            raise ValueError(
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )

        # Packed: per-request variable q_len; rectangular: uniform q_len.
        mask_chunks: List[torch.Tensor] = []
        if self._packed:
            cl_cpu = self.candidate_lens.cpu().tolist()
            for prefix_len, q_len_i in zip(batch.seq_lens_cpu.tolist(), cl_cpu):
                prefix_len_i = int(prefix_len)
                q_len_i = int(q_len_i)
                kv_len = prefix_len_i + q_len_i
                q_idx = torch.arange(q_len_i, device=batch.device, dtype=torch.int32).unsqueeze(1)
                k_idx = torch.arange(kv_len, device=batch.device, dtype=torch.int32).unsqueeze(0)
                allow = k_idx <= (prefix_len_i + q_idx)
                mask_chunks.append(allow.flatten())
        else:
            q_len = int(self.draft_token_num)
            q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
            for prefix_len in batch.seq_lens_cpu.tolist():
                prefix_len_i = int(prefix_len)
                kv_len = prefix_len_i + q_len
                k_idx = torch.arange(
                    kv_len, device=batch.device, dtype=torch.int32
                ).unsqueeze(0)
                allow = k_idx <= (prefix_len_i + q_idx)
                mask_chunks.append(allow.flatten())
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        if self._packed:
            cl = self.candidate_lens  # [bs] int32
            qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            qo_indptr[1:] = cl.cumsum(0)
            paged_kernel_lens = paged_kernel_lens + cl
            new_kv_total = int(cl.sum().item())
        else:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            paged_kernel_lens = paged_kernel_lens + self.draft_token_num
            new_kv_total = self.draft_token_num * bs

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + new_kv_total,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            if self._packed:
                cl_long = cl.long()
                mask_numel = int(
                    (paged_kernel_lens.long() * cl_long).sum().item()
                )
            else:
                mask_numel = (
                    paged_kernel_lens_sum * self.draft_token_num
                    + (self.draft_token_num**2) * bs
                )
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def _unpack_to_rect(
        self,
        bs: int,
        device: torch.device,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ):
        """Convert packed [bs*ntpb] layout to rectangular [bs*max_cl] for acceptance."""
        cl = self.candidate_lens
        sum_cl = int(cl.sum().item())
        max_cl = self._max_cl
        arange = torch.arange(max_cl, device=device)
        valid = arange[None, :] < cl[:, None]  # [bs, max_cl]

        # Scatter packed draft tokens → rectangular
        rect = self.draft_token.new_zeros(bs, max_cl)
        rect[valid] = self.draft_token[:sum_cl]
        self.draft_token = rect.reshape(-1)

        # Scatter packed logits → rectangular
        logits = logits_output.next_token_logits
        V = logits.shape[-1]
        rect_l = logits.new_zeros(bs, max_cl, V)
        rect_l[valid] = logits[:sum_cl]
        logits_output.next_token_logits = rect_l.reshape(bs * max_cl, V)

        # Scatter packed hidden states → rectangular
        if logits_output.hidden_states is not None:
            hidden = logits_output.hidden_states
            H = hidden.shape[-1]
            rect_h = hidden.new_zeros(bs, max_cl, H)
            rect_h[valid] = hidden[:sum_cl]
            logits_output.hidden_states = rect_h.reshape(bs * max_cl, H)

        # Scatter out_cache_loc: free padding slots, place real slots in rectangular.
        real_locs = batch.out_cache_loc[:sum_cl]
        padding_locs = batch.out_cache_loc[sum_cl:]
        if padding_locs.numel() > 0:
            batch.token_to_kv_pool_allocator.free(padding_locs)
        rect_c = real_locs.new_zeros(bs, max_cl)
        rect_c[valid] = real_locs
        batch.out_cache_loc = rect_c.reshape(-1)

        # Switch to rectangular mode
        self.draft_token_num = max_cl
        self.num_tokens_per_batch = max_cl
        self._packed = False
        self._valid_mask = valid  # for KV cleanup to skip padding positions

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """DFlash verification for greedy and non-greedy sampling.

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        # Unpack from packed layout [bs*ntpb] to rectangular [bs*max_cl] so all
        # downstream acceptance/cleanup code works unchanged.
        was_packed = self._packed
        if self._packed:
            self._unpack_to_rect(bs, device, batch, logits_output)

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=self.draft_token_num,
            )

        candidates = self.draft_token.view(bs, self.draft_token_num)
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            top_ks = [int(req.sampling_params.top_k) for req in batch.reqs]
            accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=max(max(top_ks), 1) if top_ks else 1,
                uniform_top_k_value=(
                    top_ks[0]
                    if top_ks and all(top_k == top_ks[0] for top_k in top_ks)
                    else None
                ),
            )
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, self.draft_token_num
            )
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )

        # Single D2H transfer: candidates[1:] + accept_len + bonus
        packed = torch.cat(
            [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        max_acc = self.draft_token_num - 1
        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        # Pre-compute per-request candidate len caps if adaptive length is enabled
        candidate_lens_cpu = None
        if self.candidate_lens is not None:
            candidate_lens_cpu = self.candidate_lens.cpu().tolist()

        for i, req in enumerate(batch.reqs):
            acc_len = int(packed[i, max_acc].item())

            # Cap acceptance at candidate_len - 1 when adaptive verification is enabled
            if candidate_lens_cpu is not None:
                max_accept = max(0, int(candidate_lens_cpu[i]) - 1)
                if acc_len > max_accept:
                    # When acc_len > max_accept, position max_accept was accepted, so
                    # candidates[i, max_accept+1] == target_predict[i, max_accept] — use it as bonus.
                    acc_len = max_accept
                    bonus_token = int(packed[i, max_accept].item()) if max_accept < max_acc else int(packed[i, max_acc + 1].item())
                else:
                    bonus_token = int(packed[i, max_acc + 1].item())
            else:
                bonus_token = int(packed[i, max_acc + 1].item())

            proposed = packed[i, :acc_len].tolist() + [bonus_token]

            appended = 0
            for token_id in proposed:
                token_id = int(token_id)
                req.output_ids.append(token_id)
                appended += 1
                req.check_finished()
                if req.finished():
                    break
                if req.grammar is not None:
                    req.grammar.accept_token(token_id)

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                # If no token was appended in this verify step, keep the current token unchanged.
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]
            # Avg Verify Len: packed → per-request cl[i]; rect dynamic → shared L
            # (draft_token_num). Must use was_packed: _unpack_to_rect clears _packed.
            if candidate_lens_cpu is not None:
                if was_packed:
                    req.spec_proposed_tokens += candidate_lens_cpu[i]
                else:
                    req.spec_proposed_tokens += self.draft_token_num
            else:
                req.spec_proposed_tokens += self.draft_token_num

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
        # _valid_mask is set by _unpack_to_rect for packed layout (marks real vs padding).
        valid_mask = getattr(self, "_valid_mask", None)  # [bs, draft_token_num] or None
        if page_size == 1:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            if valid_mask is not None:
                # Only free real (allocated) positions that are not committed.
                batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask & valid_mask])
                batch.out_cache_loc = out_cache_loc[keep_mask & valid_mask]
            else:
                batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
                batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            keep_slots = _compute_paged_keep_slots(
                prefix_lens=batch.seq_lens,
                commit_lens=commit_lens,
                draft_token_num=self.draft_token_num,
                page_size=page_size,
            )
            free_mask = row_offsets >= keep_slots[:, None]
            if valid_mask is not None:
                free_mask = free_mask & valid_mask
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])

            keep_mask = row_offsets < commit_lens[:, None]
            if valid_mask is not None:
                keep_mask = keep_mask & valid_mask
            batch.out_cache_loc = out_cache_loc[keep_mask]

        # Update req-level KV cache accounting.
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token pool mapping for newly committed tokens.
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Update batch seq lens.
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        # Keep seq_lens_sum in sync; flashinfer indices updaters rely on this for buffer sizing.
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, self.draft_token_num, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
