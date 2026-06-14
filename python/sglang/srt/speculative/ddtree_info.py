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

    accepted_indices: List[List[int]] = field(default_factory=list)
    next_tokens: Optional[torch.Tensor] = None

    # When True, the tree is a pure linear chain (no branching siblings).
    # In this mode, cascade attention is unnecessary and a standard causal
    # mask suffices, matching DFLASH's verify pattern exactly.
    tree_is_spine: bool = False

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_VERIFY)

    def get_spec_adjust_token_coefficient(self):
        return (self.draft_token_num, self.draft_token_num)

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
        from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

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
        from sglang.srt.speculative.ddtree_utils import (
            follow_verified_tree,
        )
        from sglang.srt.speculative.dflash_utils import (
            is_dflash_sampling_verify_available,
        )

        bs = len(batch.reqs)
        device = batch.device

        sampling_info = batch.sampling_info
        use_sampling = (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        )

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)

        # --- 1) Acceptance ---
        if self.tree_is_spine:
            if use_sampling:
                # Chain-based sampling verification via sgl_kernel.
                # candidates must include ALL N tokens (bonus + drafts).
                from sglang.srt.speculative.dflash_utils import (
                    compute_dflash_sampling_correct_drafts_and_bonus,
                )
                candidates = self.draft_token.view(bs, self.draft_token_num)
                correct_len, bonus = (
                    compute_dflash_sampling_correct_drafts_and_bonus(
                        candidates=candidates,
                        next_token_logits=logits_output.next_token_logits,
                        sampling_info=sampling_info,
                    )
                )
                # correct_len = number of accepted candidates (includes bonus at pos 0).
                # commit_len = correct_len + 1 (includes the final bonus token).
                commit_lens = (correct_len + 1).tolist()
                num_correct_drafts_per_req = [max(0, cl - 1) for cl in commit_lens]
                self.accepted_indices = [
                    list(range(cl)) for cl in (correct_len + 1).tolist()
                ]
                self.next_tokens = bonus
            else:
                # Spine greedy: chain comparison on GPU.
                candidates = target_predict[:, :-1]
                targets = target_predict[:, 1:]
                matches = (candidates == targets).to(torch.int32)
                correct = matches.cumprod(dim=1).sum(dim=1)
                commit_lens = (correct + 1).tolist()
                num_correct_drafts_per_req = [max(0, cl - 1) for cl in commit_lens]
                self.accepted_indices = [list(range(cl)) for cl in commit_lens]
                next_tokens_list = [
                    int(target_predict[b_i, cl - 1]) if cl > 0
                    else int(target_predict[b_i, 0])
                    for b_i, cl in enumerate(commit_lens)
                ]
                self.next_tokens = torch.tensor(
                    next_tokens_list, dtype=torch.long, device=device
                )
        else:
            # Full tree path: greedy-only for now (tree sampling requires
            # sgl_kernel tree topology which is expensive to build per-step).
            self.accepted_indices, self.next_tokens = follow_verified_tree(
                self.child_maps, target_predict
            )
            commit_lens = []
            num_correct_drafts_per_req = []
            for i, req in enumerate(batch.reqs):
                accepted = self.accepted_indices[i]
                appended = 0
                for idx in accepted[1:]:
                    token_id = int(self.draft_token[i * self.draft_token_num + idx])
                    req.output_ids.append(token_id)
                    appended += 1
                    req.update_finish_state()
                    if req.finished():
                        break
                if not req.finished():
                    bonus = int(self.next_tokens[i].item())
                    req.output_ids.append(bonus)
                    appended += 1
                    req.update_finish_state()
                commit_lens.append(appended)
                num_correct_drafts_per_req.append(max(0, appended - 1))
                req.spec_verify_ct += 1
                req.spec_num_correct_drafts += max(0, appended - 1)

        # --- 2) Commit tokens to output ---
        if self.tree_is_spine:
            for i, req in enumerate(batch.reqs):
                for idx in range(1, commit_lens[i]):
                    token_id = int(self.draft_token[i * self.draft_token_num + idx])
                    req.output_ids.append(token_id)
                    req.update_finish_state()
                    if req.finished():
                        break
                else:
                    bonus = int(self.next_tokens[i].item())
                    req.output_ids.append(bonus)
                    req.update_finish_state()
                req.spec_verify_ct += 1
                req.spec_num_correct_drafts += num_correct_drafts_per_req[i]

        commit_lens_tensor = torch.tensor(
            commit_lens, dtype=torch.long, device=device
        )

        # --- KV cache compaction (skip entirely if all paths are contiguous) ---
        if model_runner is not None:
            # Fast check: if every accepted path is contiguous from 0, no compaction needed.
            need_compaction = False
            for accepted in self.accepted_indices:
                if accepted != list(range(len(accepted))):
                    need_compaction = True
                    break

            if need_compaction:
                model = model_runner.model
                model_layers = getattr(model, "model", model)
                model_layers = getattr(model_layers, "layers", None)
                if model_layers is None:
                    model_layers = []

                from sglang.srt.speculative.ddtree_utils import compact_ddtree_kv_cache
                token_to_kv_pool = model_runner.token_to_kv_pool
                past_lengths = batch.seq_lens.clone()

                for layer in model_layers:
                    attn_layer = layer.self_attn.attn
                    compact_ddtree_kv_cache(
                        token_to_kv_pool,
                        attn_layer,
                        batch.out_cache_loc.view(bs, self.draft_token_num),
                        self.accepted_indices,
                        past_lengths,
                        self.actual_tree_sizes,
                    )

            # Free uncommitted KV cache slots.
            if page_size == 1:
                out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
                keep_mask = (
                    torch.arange(self.draft_token_num, device=device)[None, :]
                    < commit_lens_tensor[:, None]
                )
                batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
                batch.out_cache_loc = out_cache_loc[keep_mask]

            # Update req-level KV cache accounting.
            for req, commit_len in zip(batch.reqs, commit_lens, strict=True):
                req.kv_committed_len += commit_len
                req.kv_allocated_len = req.kv_committed_len

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

        hidden = logits_output.hidden_states
        if hidden is not None:
            hidden = hidden.view(bs, self.draft_token_num, -1)
            segments = []
            for i, n in enumerate(commit_lens):
                segments.append(hidden[i, :n, :])
            next_target_hidden = torch.cat(segments, dim=0)
        else:
            next_target_hidden = None

        num_correct_drafts_cpu = num_correct_drafts_per_req

        return self.next_tokens, commit_lens_tensor, next_target_hidden, num_correct_drafts_cpu
