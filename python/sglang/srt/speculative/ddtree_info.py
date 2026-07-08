"""DDTreeVerifyInput: verify-stage state for DDTree full-tree spec decode.

Holds the compiled tree (flat draft tokens, positions, LCRS topology, tree
attention mask) and implements verify():
- runs EAGLE's verify_tree_greedy kernel to find, per request, the longest
  root->leaf path whose tokens match the target's greedy predictions;
- commits the accepted tokens, evicts the rest of the tree's KV (keeping the
  scattered accepted slots), and reindexes req_to_token;
- returns the accepted tokens + per-node hidden states for the next draft step.

The Mamba recurrent-state commit for the accepted path is done by the worker
(DDTreeWorker._update_target_mamba_state_tree), not here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.common import (
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


@dataclass
class DDTreeDraftInput(SpecInput):
    """Per-batch DDTree draft state for the non-overlap decode loop."""

    # Current token to start the next draft block (one per request).
    verified_id: torch.Tensor

    # Flattened target hidden features waiting to be materialized into the draft KV
    # cache. Shape: [sum(ctx_lens), K * hidden_size].
    target_hidden: torch.Tensor

    # Newly committed context lengths per request.
    ctx_lens: torch.Tensor

    # Number of committed tokens visible to the draft worker per request.
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_DRAFT)

    def get_spec_adjust_token_coefficient(self):
        return (1, 1)

    def prepare_for_decode(self, batch):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        batch.forward_mode = ForwardMode.DECODE
        batch.input_embeds = None
        batch.seq_lens_sum = None

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

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

    def merge_batch(self, spec_info: "DDTreeDraftInput"):
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
class DDTreeVerifyInput(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    tree_budget: int
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    child_maps: List[Dict[int, Dict[int, int]]] = field(default_factory=list)
    actual_tree_sizes: Optional[torch.Tensor] = None

    custom_mask: Optional[torch.Tensor] = None
    topk: int = 1

    # EAGLE-format LCRS tree topology (set by the worker for full-tree mode).
    # Shapes [bs, draft_token_num], int64, -1 = none. retrieve_parent_token is
    # derived inside the mamba kernel and must NOT be supplied.
    retrive_index: Optional[torch.Tensor] = None
    retrive_next_token: Optional[torch.Tensor] = None
    retrive_next_sibling: Optional[torch.Tensor] = None
    spec_steps: int = 0
    draft_probs: Optional[torch.Tensor] = None

    # Populated by verify(): flat batch-global accepted node slots (EAGLE
    # contract, used by the worker to compute mamba accepted_steps).
    accepted_indices: List[List[int]] = field(default_factory=list)
    accept_index_flat: Optional[torch.Tensor] = None  # 1D int, accepted global slots
    accept_length_per_req: Optional[torch.Tensor] = None  # [bs] int, accepted drafts
    next_tokens: Optional[torch.Tensor] = None

    # When True, the tree is a pure linear chain (no branching siblings).
    # In this mode, cascade attention is unnecessary and a standard causal
    # mask suffices, matching DFLASH's verify pattern exactly.
    tree_is_spine: bool = False

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_VERIFY)

    def get_spec_adjust_token_coefficient(self):
        return (self.draft_token_num, self.draft_token_num)

    @property
    def retrieve_index(self) -> Optional[torch.Tensor]:
        return self.retrive_index

    @property
    def retrieve_next_token(self) -> Optional[torch.Tensor]:
        return self.retrive_next_token

    @property
    def retrieve_next_sibling(self) -> Optional[torch.Tensor]:
        return self.retrive_next_sibling

    @property
    def max_tree_depth(self) -> int:
        return (
            self.spec_steps if self.spec_steps > 0 else self.draft_token_num - 1
        ) + 1

    @property
    def tree_topk(self) -> int:
        return self.topk

    def prepare_for_verify(self, batch, page_size):
        bs = len(batch.reqs)
        q_len = self.draft_token_num

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + q_len
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset_cpu = [pl + q_len for pl in prefix_lens_cpu.tolist()]
            from sglang.srt.mem_cache.common import (
                alloc_paged_token_slots_extend,
            )

            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens_cpu.tolist(),
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            end_offset = torch.tensor(
                end_offset_cpu, dtype=prefix_lens.dtype, device=prefix_lens.device
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        from sglang.srt.layers.attention.utils import (
            create_flashinfer_kv_indices_triton,
        )

        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
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

    def verify(self, *, batch, logits_output, page_size, model_runner=None):
        bs = len(batch.reqs)
        device = batch.device

        D = self.draft_token_num
        target_predict_for_probe = torch.argmax(
            logits_output.next_token_logits, dim=-1
        ).reshape(bs, D)

        # --- 1) Acceptance via EAGLE's tree sampling/verify path ---
        # Greedy requests still use verify_tree_greedy. Non-greedy requests use
        # target-only tree sampling, so temperature/top-p/top-k/penalty/logit-bias
        # semantics are honored by the target verifier instead of silently falling
        # back to greedy decoding.
        from sglang.srt.speculative.eagle_utils import eagle_sample

        predict, accept_length, accept_index = eagle_sample(
            verify_input=self,
            batch=batch,
            logits_output=logits_output,
        )

        # accept_index[b]: [root_global_slot, accepted child global slots..., -1...]
        # accept_length[b]: number of accepted *draft* tokens (excludes root).
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()

        commit_lens = []  # tokens committed to output per req (drafts + bonus)
        num_correct_drafts_per_req = []
        next_tokens_list = []
        accepted_tokens_padded = []
        self.accepted_indices = []  # per-req LOCAL accepted node indices (for KV)
        accepted_flat = []  # batch-global COMMITTED slots in accept order (EAGLE)

        for i, req in enumerate(batch.reqs):
            row = accept_index_cpu[i]
            base = i * D

            # Commit predict[slot] for each accepted node (root..last). Stop early
            # if the request finishes; only COMMITTED slots are kept in the flat
            # list so KV keep-count == commit-count (no leak).
            # accept_length+1 candidate slots, mirrors DFLASH's drafts+bonus.
            local_accepted = []
            local_tokens = []
            appended = 0
            for slot in row:
                if slot == -1:
                    break
                token_id = int(predict_cpu[slot])
                local_tokens.append(token_id)
                local_accepted.append(slot - base)
                accepted_flat.append(slot)
                appended += 1

            self.accepted_indices.append(local_accepted)

            if local_tokens:
                new_verified_token = int(local_tokens[-1])
            elif req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DDTREE verify cannot determine current token: both "
                    "output_ids and origin_input_ids are empty."
                )
            next_tokens_list.append(new_verified_token)
            accepted_tokens_padded.extend(
                local_tokens + [new_verified_token] * max(0, D - len(local_tokens))
            )

            commit_lens.append(appended)
            num_correct_drafts_per_req.append(max(0, appended - 1))

        self.next_tokens = torch.tensor(
            next_tokens_list, dtype=torch.long, device=device
        )
        self.accepted_tokens = torch.tensor(
            accepted_tokens_padded, dtype=torch.long, device=device
        )
        # Store EAGLE-format outputs for the worker's mamba state commit.
        # accept_index_flat now contains exactly the committed slots (post-finish
        # truncation), and accept_length_per_req is recomputed to match.
        self.accept_index_flat = torch.tensor(
            accepted_flat, dtype=torch.long, device=device
        )
        self.accept_length_per_req = torch.tensor(
            num_correct_drafts_per_req, dtype=torch.long, device=device
        )

        commit_lens_tensor = torch.tensor(commit_lens, dtype=torch.long, device=device)

        # --- KV cache management (EAGLE-style: keep the scattered accepted
        # slots, free the rest, reindex req_to_token). No per-layer data
        # movement (works on hybrid Mamba models; Mamba state is committed
        # separately by the worker). Only page_size==1 is supported for now. ---
        if model_runner is not None:
            assert page_size == 1, "DDTREE full-tree currently supports page_size==1"
            # accept_index_flat holds, per req in accept order, the global slots
            # (b*D + local_idx) of every accepted node. These are the KV slots to
            # keep; everything else in the bs*D verify block is freed.
            all_slots = batch.out_cache_loc  # [bs*D]
            keep_slots = all_slots[self.accept_index_flat]  # scattered -> contiguous

            evict_mask = torch.ones(all_slots.numel(), dtype=torch.bool, device=device)
            evict_mask[self.accept_index_flat] = False
            batch.token_to_kv_pool_allocator.free(all_slots[evict_mask])
            batch.out_cache_loc = keep_slots

            # Update req-level KV cache accounting.
            for req, commit_len in zip(batch.reqs, commit_lens, strict=True):
                req.kv_allocated_len = req.kv_committed_len + commit_len

            # Update req_to_token pool mapping for newly committed tokens.
            end_offset = batch.seq_lens + commit_lens_tensor.to(batch.seq_lens.dtype)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                end_offset,
                batch.out_cache_loc,
                bs,
            )

            # Update batch seq lens.
            batch.seq_lens.add_(commit_lens_tensor.to(batch.seq_lens.dtype))
            batch.seq_lens_cpu.add_(
                torch.tensor(
                    [int(c) for c in commit_lens], dtype=batch.seq_lens_cpu.dtype
                )
            )
            batch.seq_lens_sum += sum(commit_lens)
        else:
            # Fallback path
            batch.seq_lens += commit_lens_tensor

        # Gather hidden states for the accepted (scattered) tree nodes, in the
        # same accept order as accept_index_flat (= concatenation per req).
        hidden = logits_output.hidden_states
        if hidden is not None:
            next_target_hidden = hidden[self.accept_index_flat]
        else:
            next_target_hidden = None

        num_correct_drafts_cpu = num_correct_drafts_per_req

        # --- Per-depth target-token RANK probe (env-gated) ---
        # Along the target's correct path, at each depth find what RANK the
        # target's next token has in the DRAFT's full distribution for that
        # position (top1/2/4/8/16/32 or miss). Directly answers "would more
        # budget/topk per position help" — if target is often top2-8 (in dist
        # but ranked low), wider trees help; if it's top1-or-miss, they don't.
        _draft_pos_topk = getattr(self, "_draft_pos_topk", None)
        if _draft_pos_topk is not None:
            cls = DDTreeVerifyInput
            seq0 = int(batch.seq_lens[0].item()) if len(batch.seq_lens) else 0
            if seq0 >= 64:
                THR = [1, 2, 4, 8, 16, 32]
                # bucket[d] = [hits@top1, @top2, @top4, @top8, @top16, @top32, reached]
                buck = getattr(cls, "_rank_buck", None)
                if buck is None:
                    buck = [[0] * (len(THR) + 1) for _ in range(D)]
                tp_cpu = target_predict_for_probe.tolist()
                # draft_pos_topk: [bs, block_size-1, 32]; position p (depth p+1)
                dpt = _draft_pos_topk.tolist()
                npos = _draft_pos_topk.shape[1]
                for b in range(bs):
                    cmap = self.child_maps[b] if b < len(self.child_maps) else {}
                    cur = 0
                    depth = 0
                    while depth < npos:
                        want = tp_cpu[b][cur]
                        ranked = dpt[b][
                            depth
                        ]  # draft top-32 token ids at this position
                        buck[depth][len(THR)] += 1  # reached
                        try:
                            r = ranked.index(want)  # 0-based rank
                            for ti, t in enumerate(THR):
                                if r < t:
                                    buck[depth][ti] += 1
                        except ValueError:
                            pass  # not even in top-32
                        # follow accepted path (target token must be a child to continue)
                        children = cmap.get(cur, {})
                        if want in children:
                            cur = children[want]
                            depth += 1
                        else:
                            break
                cls._rank_buck = buck
                cls._rank_step = getattr(cls, "_rank_step", 0) + 1
                if cls._rank_step % 50 == 0:
                    import logging as _lg

                    lines = [
                        "[DDTREE rank-probe] target-token rank in draft dist (cumulative):"
                    ]
                    lines.append("depth reached  t1    t2    t4    t8    t16   t32")
                    for d in range(npos):
                        reached = buck[d][len(THR)]
                        if reached == 0:
                            break
                        rates = "  ".join(
                            f"{buck[d][ti] / reached:.2f}" for ti in range(len(THR))
                        )
                        lines.append(f"d{d:<4} {reached:<7} {rates}")
                    _lg.getLogger(__name__).info("\n".join(lines))

        return (
            self.next_tokens,
            commit_lens_tensor,
            next_target_hidden,
            num_correct_drafts_cpu,
        )
