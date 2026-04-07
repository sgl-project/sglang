from __future__ import annotations

from collections import OrderedDict
import hashlib
import dataclasses
import logging
import os
from typing import Optional

import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.utils import get_bool_env_var
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)

logger = logging.getLogger(__name__)


class SSDWorkerV1(EAGLEWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self._ssd_next_verify_map = OrderedDict()
        self._ssd_last_tree_parent_list = None
        self._ssd_last_tree_top_scores_index = None
        self._ssd_last_tree_draft_tokens = None
        self._ssd_step_cache_hits_sum = 0
        self._ssd_step_cache_hits_count = 0
        self._ssd_step_subtree_attempts = 0
        self._ssd_step_subtree_fallbacks = 0
        self._ssd_debug_left = int(os.getenv("SGLANG_SSD_V1_DEBUG_LIMIT", "20"))
        self._ssd_root_cov_left = int(os.getenv("SGLANG_SSD_V1_ROOT_COV_LIMIT", "50"))
        self._ssd_cache_max_entries = int(
            os.getenv("SGLANG_SSD_V1_CACHE_MAX_ENTRIES", "8192")
        )
        self._ssd_last_candidate_tokens = None
        self._ssd_last_candidate_scores = None
        self._ssd_last_candidate_miss_indices = None
        self._ssd_last_candidate_miss_tokens = None
        self._ssd_last_candidate_miss_scores = None
    @staticmethod
    def _get_rid_hash_tensor(reqs, device: torch.device) -> torch.Tensor:
        vals = []
        for r in reqs:
            v = getattr(r, "_ssd_rid_hash", None)
            if v is None:
                h = hashlib.blake2b(str(r.rid).encode("utf-8"), digest_size=8).digest()
                v = int.from_bytes(h, byteorder="big", signed=False)
                if v >= (1 << 63):
                    v -= 1 << 64
                setattr(r, "_ssd_rid_hash", v)
            vals.append(v)
        return torch.tensor(vals, device=device, dtype=torch.int64)

    @staticmethod
    def _ssd_build_next_verify_input_by_subtree(
        parent_list: torch.Tensor,
        top_scores_index: torch.Tensor,
        draft_tokens: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        topk: int,
        next_verified_id: torch.Tensor,
    ):
        device = top_scores_index.device
        bs = int(accept_lens.numel())
        n = int(draft_tokens.shape[1]) + 1
        if bs <= 0 or n <= 1:
            return None
        if not isinstance(next_verified_id, torch.Tensor) or next_verified_id.numel() != bs:
            return None
        parent_len = int(parent_list.shape[1])
        if n - 1 > parent_len - 1:
            return None
        accept_lens_i64 = accept_lens.to(dtype=torch.int64)
        base = torch.arange(bs, device=accept_lens_i64.device, dtype=torch.int64) * n
        last_pos = (accept_lens_i64 - 1).clamp(min=0, max=int(accept_index.shape[1]) - 1)
        idx = (
            accept_index.to(dtype=torch.int64)
            .gather(1, last_pos.view(-1, 1))
            .squeeze(1)
        )
        zero_mask = accept_lens_i64 <= 0
        if zero_mask.any().item():
            idx = idx.clone()
            idx[zero_mask] = base[zero_mask]
        root_pos = (idx - base).clamp(min=0, max=n - 1).to("cpu", dtype=torch.int64).tolist()

        parent_list_cpu = parent_list.to("cpu", dtype=torch.int64)
        top_scores_index_cpu = top_scores_index.to("cpu", dtype=torch.int64)
        draft_tokens_cpu = draft_tokens.to("cpu", dtype=torch.int64)
        next_verified_id_cpu = next_verified_id.to("cpu", dtype=torch.int64)
        new_parent_list = torch.full_like(parent_list_cpu, -1)
        new_top_scores = torch.empty_like(top_scores_index_cpu)
        new_draft_tokens = torch.empty_like(draft_tokens_cpu)
        reuse_mask = torch.ones((bs,), dtype=torch.bool)
        max_tb_idx = parent_len - 1
        relax_fill = get_bool_env_var(
            "SGLANG_SSD_TRUE_TREE_CACHE_SUBTREE_RELAXED_FILL", "true"
        )
        debug = os.getenv("SGLANG_SSD_V1_DEBUG", "false").lower() in ("1", "true")
        fail_accept = 0
        fail_len = 0
        fail_dup = 0
        fail_root = 0
        fail_order = 0
        fail_map = 0
        match_accept = 0
        found_any = 0

        for b in range(bs):
            selected = top_scores_index_cpu[b].tolist()
            accepted_pos = int(root_pos[b]) - 1
            if accepted_pos >= n - 1:
                reuse_mask[b] = False
                fail_accept += 1
                continue
            if len(selected) != n - 1:
                reuse_mask[b] = False
                fail_len += 1
                continue
            token_to_pos = {}
            has_dup = False
            for i, t in enumerate(selected):
                if t in token_to_pos:
                    has_dup = True
                    break
                token_to_pos[t] = i
            if has_dup:
                fail_dup += 1
                parent_pos = [-1] * (n - 1)
                children = [[] for _ in range(n - 1)]
                root_children = list(range(n - 1))
            else:
                parent_pos = [-1] * (n - 1)
                for i, token_idx in enumerate(selected):
                    parent_tb_idx = token_idx // topk
                    if parent_tb_idx > 0:
                        parent_token_idx = int(parent_list_cpu[b, parent_tb_idx].item())
                        parent_pos[i] = token_to_pos.get(parent_token_idx, -1)

                children = [[] for _ in range(n - 1)]
                root_children = []
                for i, p in enumerate(parent_pos):
                    if p >= 0:
                        children[p].append(i)
                    else:
                        root_children.append(i)

            root_sel = None
            if accepted_pos >= 0:
                if int(draft_tokens_cpu[b, accepted_pos].item()) == int(
                    next_verified_id_cpu[b].item()
                ):
                    root_sel = accepted_pos
                    match_accept += 1
            else:
                if int(next_verified_id_cpu[b].item()) in draft_tokens_cpu[b].tolist():
                    found_any += 1
                q = list(root_children)
                seen_match = set()
                while q:
                    u = q.pop(0)
                    if u in seen_match:
                        continue
                    seen_match.add(u)
                    if int(draft_tokens_cpu[b, u].item()) == int(
                        next_verified_id_cpu[b].item()
                    ):
                        root_sel = u
                        break
                    q.extend(children[u])
            if root_sel is None and accepted_pos >= 0:
                if int(next_verified_id_cpu[b].item()) in draft_tokens_cpu[b].tolist():
                    found_any += 1
                q = list(children[accepted_pos])
                seen_match = set()
                while q:
                    u = q.pop(0)
                    if u in seen_match:
                        continue
                    seen_match.add(u)
                    if int(draft_tokens_cpu[b, u].item()) == int(
                        next_verified_id_cpu[b].item()
                    ):
                        root_sel = u
                        break
                    q.extend(children[u])
            if root_sel is None:
                target = int(next_verified_id_cpu[b].item())
                for pos, tok in enumerate(draft_tokens_cpu[b].tolist()):
                    if int(tok) == target:
                        root_sel = pos
                        break
            if root_sel is None:
                reuse_mask[b] = False
                fail_root += 1
                continue

            q = list(children[root_sel])
            order = []
            seen = set()
            while q and len(order) < n - 1:
                u = q.pop(0)
                if u in seen:
                    continue
                seen.add(u)
                order.append(u)
                q.extend(children[u])
            relax_used = False
            if len(order) < n - 1:
                if not relax_fill:
                    reuse_mask[b] = False
                    continue
                relax_used = True
                for i in range(n - 1):
                    if i in seen:
                        continue
                    order.append(i)
                    if len(order) >= n - 1:
                        break
            if len(order) < n - 1:
                reuse_mask[b] = False
                fail_order += 1
                continue

            token_idx_list = [0] * (n - 1)
            draft_token_list = [0] * (n - 1)
            valid = True
            if relax_used:
                for new_idx, pos in enumerate(order):
                    tb_idx = new_idx + 1
                    if tb_idx > max_tb_idx:
                        valid = False
                        break
                    parent_tb_idx = (tb_idx - 1) // topk
                    child_slot = (tb_idx - 1) % topk
                    token_idx = parent_tb_idx * topk + child_slot
                    token_idx_list[new_idx] = token_idx
                    draft_token_list[new_idx] = int(draft_tokens_cpu[b, pos].item())
                    new_parent_list[b, tb_idx] = token_idx
            else:
                pos_to_tb = {}
                for new_idx, pos in enumerate(order):
                    tb_idx = new_idx + 1
                    if tb_idx > max_tb_idx:
                        valid = False
                        break
                    pos_to_tb[pos] = tb_idx
                if valid:
                    child_counts = {}
                    for new_idx, pos in enumerate(order):
                        p = parent_pos[pos]
                        if p == root_sel:
                            parent_tb_idx = 0
                        else:
                            parent_tb_idx = pos_to_tb.get(p, None)
                            if parent_tb_idx is None:
                                valid = False
                                break
                        count = child_counts.get(parent_tb_idx, 0)
                        if count >= topk:
                            valid = False
                            break
                        token_idx = parent_tb_idx * topk + count
                        child_counts[parent_tb_idx] = count + 1
                        token_idx_list[new_idx] = token_idx
                        draft_token_list[new_idx] = int(draft_tokens_cpu[b, pos].item())
                        tb_idx = new_idx + 1
                        new_parent_list[b, tb_idx] = token_idx
            if not valid:
                if relax_fill:
                    valid2 = True
                    for new_idx, pos in enumerate(order):
                        tb_idx = new_idx + 1
                        if tb_idx > max_tb_idx:
                            valid2 = False
                            break
                        parent_tb_idx = (tb_idx - 1) // topk
                        child_slot = (tb_idx - 1) % topk
                        token_idx = parent_tb_idx * topk + child_slot
                        token_idx_list[new_idx] = token_idx
                        draft_token_list[new_idx] = int(draft_tokens_cpu[b, pos].item())
                        new_parent_list[b, tb_idx] = token_idx
                    valid = valid2
            if not valid:
                reuse_mask[b] = False
                fail_map += 1
                continue
            new_top_scores[b] = torch.tensor(
                token_idx_list, dtype=top_scores_index_cpu.dtype
            )
            new_draft_tokens[b] = torch.tensor(
                draft_token_list, dtype=draft_tokens_cpu.dtype
            )
        if debug and bs > 0:
            reuse_true = int(reuse_mask.to(torch.int32).sum().item())
            logger.info(
                "[SSD_V1] reuse_true=%d/%d match_accept=%d found_any=%d "
                "fail_accept=%d fail_len=%d fail_dup=%d fail_root=%d fail_order=%d fail_map=%d",
                reuse_true,
                bs,
                match_accept,
                found_any,
                fail_accept,
                fail_len,
                fail_dup,
                fail_root,
                fail_order,
                fail_map,
            )

        return (
            new_parent_list.to(device=device, dtype=parent_list.dtype),
            new_top_scores.to(device=device, dtype=top_scores_index.dtype),
            new_draft_tokens.to(device=device, dtype=draft_tokens.dtype),
            reuse_mask.to(device=device),
        )

    def draft(self, batch):
        if batch.forward_mode.is_idle():
            return super().draft(batch)

        self._draft_preprocess_decode(batch)

        bs = batch.batch_size()
        self._ssd_step_cache_hits_sum = 0
        self._ssd_step_cache_hits_count = bs
        subtree_enabled = (
            get_bool_env_var("SGLANG_SSD_SUBTREE_REUSE", "false") and self.topk > 1
        )
        self._ssd_step_subtree_attempts = bs if subtree_enabled else 0
        self._ssd_step_subtree_fallbacks = bs if subtree_enabled else 0

        spec_info = batch.spec_info
        if (
            subtree_enabled
            and self._ssd_next_verify_map is not None
            and isinstance(spec_info, EagleDraftInput)
            and isinstance(getattr(spec_info, "verified_id", None), torch.Tensor)
        ):
            seq_lens_i64 = batch.seq_lens.to(dtype=torch.int64)
            verified_ids_i64 = spec_info.verified_id.to(dtype=torch.int64)
            rid_hash = self._get_rid_hash_tensor(batch.reqs, seq_lens_i64.device)

            hits = []
            hit_rows = []
            misses = []
            map_before = len(self._ssd_next_verify_map)
            for i in range(bs):
                k = (
                    int(rid_hash[i].item()),
                    int(seq_lens_i64[i].item()),
                    int(verified_ids_i64[i].item()),
                )
                v = self._ssd_next_verify_map.pop(k, None)
                if v is None:
                    misses.append(i)
                else:
                    hits.append(i)
                    hit_rows.append(v)
            if (
                os.getenv("SGLANG_SSD_V1_DEBUG", "false").lower() in ("1", "true")
                and self._ssd_debug_left > 0
            ):
                self._ssd_debug_left -= 1
                backend_type = (
                    self.server_args.speculative_draft_attention_backend
                    or self.server_args.decode_attention_backend
                    or self.server_args.attention_backend
                )
                logger.info(
                    "[SSD_V1] lookup bs=%d map_before=%d map_after=%d hits=%d misses=%d page_size=%d backend=%s sample_key=%s",
                    bs,
                    map_before,
                    len(self._ssd_next_verify_map),
                    len(hits),
                    len(misses),
                    int(self.page_size),
                    str(backend_type),
                    (
                        (
                            int(rid_hash[0].item()),
                            int(seq_lens_i64[0].item()),
                            int(verified_ids_i64[0].item()),
                        )
                        if bs > 0
                        else None
                    ),
                )

            hit_count = len(hits)
            if hit_count > 0:
                miss_count = bs - hit_count
                self._ssd_step_subtree_fallbacks = miss_count
                self._ssd_step_cache_hits_sum = hit_count
                backend_type = (
                    self.server_args.speculative_draft_attention_backend
                    or self.server_args.decode_attention_backend
                    or self.server_args.attention_backend
                )
                allow_partial_paged = get_bool_env_var(
                    "SGLANG_SSD_V1_ALLOW_PARTIAL_PAGED", "false"
                )
                if (
                    miss_count > 0
                    and self.page_size > 1
                    and backend_type == "flashinfer"
                    and not allow_partial_paged
                ):
                    self._ssd_step_subtree_fallbacks = bs
                    self._ssd_step_cache_hits_sum = 0
                else:
                    device = batch.seq_lens.device
                    parent_row, top_row, draft_row = hit_rows[0][:3]
                    parent_list = torch.full(
                        (bs, parent_row.numel()),
                        -1,
                        device=device,
                        dtype=parent_row.dtype,
                    )
                    top_scores_index = torch.empty(
                        (bs, top_row.numel()),
                        device=device,
                        dtype=top_row.dtype,
                    )
                    draft_tokens = torch.empty(
                        (bs, draft_row.numel()),
                        device=device,
                        dtype=draft_row.dtype,
                    )
                    cand_tokens = None
                    cand_scores = None
                    if len(hit_rows[0]) >= 5:
                        cand0 = hit_rows[0][3]
                        score0 = hit_rows[0][4]
                        if isinstance(cand0, torch.Tensor):
                            cand_tokens = torch.empty(
                                (bs, cand0.numel()),
                                device=device,
                                dtype=cand0.dtype,
                            )
                        if isinstance(score0, torch.Tensor):
                            cand_scores = torch.empty(
                                (bs, score0.numel()),
                                device=device,
                                dtype=score0.dtype,
                            )
                    hit_indices = torch.tensor(hits, device=device, dtype=torch.int64)
                    parent_hit = torch.stack([x[0] for x in hit_rows], dim=0)
                    top_hit = torch.stack([x[1] for x in hit_rows], dim=0)
                    draft_hit = torch.stack([x[2] for x in hit_rows], dim=0)
                    parent_list.index_copy_(0, hit_indices, parent_hit)
                    top_scores_index.index_copy_(0, hit_indices, top_hit)
                    draft_tokens.index_copy_(0, hit_indices, draft_hit)
                    if isinstance(cand_tokens, torch.Tensor):
                        cand_hit = torch.stack(
                            [
                                x[3]
                                if len(x) >= 5 and isinstance(x[3], torch.Tensor)
                                else torch.full_like(hit_rows[0][3], -1)
                                for x in hit_rows
                            ],
                            dim=0,
                        )
                        cand_tokens.index_copy_(0, hit_indices, cand_hit)
                    if isinstance(cand_scores, torch.Tensor):
                        score_hit = torch.stack(
                            [
                                x[4]
                                if len(x) >= 5 and isinstance(x[4], torch.Tensor)
                                else torch.full_like(hit_rows[0][4], -1e9)
                                for x in hit_rows
                            ],
                            dim=0,
                        )
                        cand_scores.index_copy_(0, hit_indices, score_hit)

                    if miss_count > 0:
                        model_worker_batch = batch.get_model_worker_batch()
                        per_req = self.topk * self.speculative_num_steps
                        can_slice = (
                            isinstance(model_worker_batch.out_cache_loc, torch.Tensor)
                            and model_worker_batch.out_cache_loc.dim() == 1
                            and model_worker_batch.out_cache_loc.numel() == bs * per_req
                        )
                        if not can_slice:
                            self._ssd_step_subtree_fallbacks = bs
                            self._ssd_step_cache_hits_sum = 0
                        else:
                            miss_indices = torch.tensor(
                                misses, device=device, dtype=torch.int64
                            )
                            miss_indices_cpu = misses
                            miss_spec = dataclasses.replace(model_worker_batch.spec_info)
                            for name in [
                                "topk_p",
                                "topk_index",
                                "hidden_states",
                                "verified_id",
                            ]:
                                v = getattr(miss_spec, name, None)
                                if isinstance(v, torch.Tensor) and v.shape[0] == bs:
                                    setattr(miss_spec, name, v[miss_indices])
                            miss_out_cache_loc = (
                                model_worker_batch.out_cache_loc.view(bs, per_req)[
                                    miss_indices
                                ]
                                .reshape(-1)
                                .contiguous()
                            )
                            miss_mwb = dataclasses.replace(
                                model_worker_batch,
                                input_ids=(
                                    model_worker_batch.input_ids[miss_indices]
                                    if isinstance(model_worker_batch.input_ids, torch.Tensor)
                                    and model_worker_batch.input_ids.shape[0] == bs
                                    else model_worker_batch.input_ids
                                ),
                                req_pool_indices=model_worker_batch.req_pool_indices[
                                    miss_indices
                                ],
                                seq_lens=model_worker_batch.seq_lens[miss_indices],
                                seq_lens_cpu=(
                                    model_worker_batch.seq_lens_cpu[miss_indices_cpu]
                                    if isinstance(model_worker_batch.seq_lens_cpu, torch.Tensor)
                                    and model_worker_batch.seq_lens_cpu.shape[0] == bs
                                    else model_worker_batch.seq_lens_cpu
                                ),
                                seq_lens_sum=int(
                                    model_worker_batch.seq_lens[miss_indices].sum().item()
                                ),
                                out_cache_loc=miss_out_cache_loc,
                                spec_info=miss_spec,
                                reqs=[batch.reqs[i] for i in miss_indices_cpu],
                            )
                            miss_spec.positions = miss_mwb.seq_lens.repeat_interleave(
                                self.topk, dim=0
                            )
                            miss_fb = ForwardBatch.init_new(
                                miss_mwb, self.draft_model_runner
                            )
                            miss_fb.can_run_dp_cuda_graph = False
                            if (
                                not miss_fb.forward_mode.is_idle()
                                and self.speculative_num_steps > 1
                            ):
                                self.draft_attn_backend.init_forward_metadata(miss_fb)
                            parent_list_m, top_scores_index_m, draft_tokens_m = (
                                self.draft_forward(miss_fb)
                            )
                            self._ssd_last_candidate_miss_indices = miss_indices_cpu
                            self._ssd_last_candidate_miss_tokens = getattr(
                                miss_fb, "_ssd_candidate_tokens", None
                            )
                            self._ssd_last_candidate_miss_scores = getattr(
                                miss_fb, "_ssd_candidate_scores", None
                            )
                            parent_list.index_copy_(0, miss_indices, parent_list_m)
                            top_scores_index.index_copy_(
                                0, miss_indices, top_scores_index_m
                            )
                            draft_tokens.index_copy_(0, miss_indices, draft_tokens_m)
                            if isinstance(cand_tokens, torch.Tensor) and isinstance(
                                self._ssd_last_candidate_miss_tokens, torch.Tensor
                            ):
                                cand_tokens.index_copy_(
                                    0, miss_indices, self._ssd_last_candidate_miss_tokens
                                )
                            if isinstance(cand_scores, torch.Tensor) and isinstance(
                                self._ssd_last_candidate_miss_scores, torch.Tensor
                            ):
                                cand_scores.index_copy_(
                                    0, miss_indices, self._ssd_last_candidate_miss_scores
                                )

                    if self._ssd_step_cache_hits_sum > 0:
                        self._ssd_last_tree_parent_list = parent_list
                        self._ssd_last_tree_top_scores_index = top_scores_index
                        self._ssd_last_tree_draft_tokens = draft_tokens
                        self._ssd_last_candidate_tokens = cand_tokens
                        self._ssd_last_candidate_scores = cand_scores

                        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
                        spec_info.num_tokens_per_req = self.topk
                        spec_info.num_tokens_for_logprob_per_req = self.topk
                        (
                            tree_mask,
                            position,
                            retrive_index,
                            retrive_next_token,
                            retrive_next_sibling,
                            draft_tokens_flat,
                        ) = build_tree_kernel_efficient(
                            spec_info.verified_id,
                            parent_list,
                            top_scores_index,
                            draft_tokens,
                            batch.seq_lens,
                            batch.seq_lens_sum,
                            self.topk,
                            self.speculative_num_steps,
                            self.speculative_num_draft_tokens,
                        )
                        return EagleVerifyInput(
                            draft_token=draft_tokens_flat,
                            custom_mask=tree_mask,
                            positions=position,
                            retrive_index=retrive_index,
                            retrive_next_token=retrive_next_token,
                            retrive_next_sibling=retrive_next_sibling,
                            retrive_cum_len=None,
                            spec_steps=self.speculative_num_steps,
                            topk=self.topk,
                            draft_token_num=self.speculative_num_draft_tokens,
                            capture_hidden_mode=CaptureHiddenMode.FULL,
                            seq_lens_sum=int(batch.seq_lens_sum),
                            seq_lens_cpu=batch.seq_lens_cpu,
                        )

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.can_run_dp_cuda_graph = False
        if not forward_batch.forward_mode.is_idle() and self.speculative_num_steps > 1:
            self.draft_attn_backend.init_forward_metadata(forward_batch)

        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)
        self._ssd_last_tree_parent_list = parent_list
        self._ssd_last_tree_top_scores_index = top_scores_index
        self._ssd_last_tree_draft_tokens = draft_tokens
        self._ssd_last_candidate_tokens = getattr(
            forward_batch, "_ssd_candidate_tokens", None
        )
        self._ssd_last_candidate_scores = getattr(
            forward_batch, "_ssd_candidate_scores", None
        )
        self._ssd_last_candidate_miss_indices = None
        self._ssd_last_candidate_miss_tokens = None
        self._ssd_last_candidate_miss_scores = None

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens_flat,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens_flat,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def forward_batch_generation(self, batch):
        from sglang.srt.managers.utils import GenerationBatchResult

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(batch)

        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            spec_info = self.draft(batch)
        logits_output, verify_output, _, can_run_cuda_graph = self.verify(batch, spec_info)

        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            if self.server_args.enable_dp_attention or batch.spec_info.verified_id.shape[0] > 0:
                self.forward_draft_extend_after_decode(batch)

        parent_list = self._ssd_last_tree_parent_list
        top_scores_index = self._ssd_last_tree_top_scores_index
        draft_tokens = self._ssd_last_tree_draft_tokens
        if (
            get_bool_env_var("SGLANG_SSD_SUBTREE_REUSE", "false")
            and self.topk > 1
            and isinstance(parent_list, torch.Tensor)
            and isinstance(top_scores_index, torch.Tensor)
            and isinstance(draft_tokens, torch.Tensor)
            and isinstance(batch.spec_info, EagleDraftInput)
            and isinstance(getattr(verify_output, "accepted_indices", None), torch.Tensor)
            and isinstance(getattr(verify_output, "accept_length_per_req_cpu", None), list)
        ):
            if (int(self.speculative_num_draft_tokens) - 1) > (
                int(self.topk) * (int(self.speculative_num_steps) - 1)
            ):
                if (
                    os.getenv("SGLANG_SSD_V1_DEBUG", "false").lower() in ("1", "true")
                    and self._ssd_debug_left > 0
                ):
                    self._ssd_debug_left -= 1
                    logger.info(
                        "[SSD_V1] pack_skip invalid_config draft_tokens=%d topk=%d steps=%d",
                        int(self.speculative_num_draft_tokens),
                        int(self.topk),
                        int(self.speculative_num_steps),
                    )
                res = GenerationBatchResult(
                    logits_output=logits_output,
                    next_token_ids=verify_output.verified_id,
                    num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                    accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                    can_run_cuda_graph=can_run_cuda_graph,
                    ssd_cache_hits_sum=int(self._ssd_step_cache_hits_sum),
                    ssd_cache_hits_count=int(self._ssd_step_cache_hits_count),
                    ssd_subtree_reuse_attempts=int(self._ssd_step_subtree_attempts),
                    ssd_subtree_reuse_fallbacks=int(self._ssd_step_subtree_fallbacks),
                )
                self._ssd_step_cache_hits_sum = 0
                self._ssd_step_cache_hits_count = 0
                self._ssd_step_subtree_attempts = 0
                self._ssd_step_subtree_fallbacks = 0
                return res
            bs = batch.batch_size()
            next_draft_input = (
                verify_output.draft_input
                if isinstance(getattr(verify_output, "draft_input", None), EagleDraftInput)
                else None
            )
            if (
                next_draft_input is None
                or not isinstance(getattr(next_draft_input, "verified_id", None), torch.Tensor)
                or next_draft_input.verified_id.numel() == 0
                or not isinstance(
                    getattr(next_draft_input, "req_pool_indices_for_draft_extend", None),
                    torch.Tensor,
                )
            ):
                next_draft_input = None

            accept_lens_inclusive = torch.tensor(
                [int(x) + 1 for x in verify_output.accept_length_per_req_cpu],
                device=verify_output.accepted_indices.device,
                dtype=torch.int32,
            )
            accepted_indices = verify_output.accepted_indices.to(dtype=torch.int32)
            cols = int(self.speculative_num_steps) + 1
            accept_index_mat = torch.full(
                (bs, cols), -1, dtype=torch.int32, device=accepted_indices.device
            )
            pt = 0
            for i, n in enumerate(accept_lens_inclusive.to("cpu", dtype=torch.int64).tolist()):
                if n > 0:
                    accept_index_mat[i, : min(n, cols)] = accepted_indices[pt : pt + min(n, cols)]
                pt += n
            if next_draft_input is not None:
                base_req_pool = batch.req_pool_indices.to("cpu", dtype=torch.int64).tolist()
                base_map = {v: i for i, v in enumerate(base_req_pool)}
                unfinished_cpu = [
                    base_map.get(int(v), -1)
                    for v in next_draft_input.req_pool_indices_for_draft_extend.to(
                        "cpu", dtype=torch.int64
                    ).tolist()
                ]
                unfinished_cpu = [i for i in unfinished_cpu if i >= 0]
                if len(unfinished_cpu) > 0:
                    unfinished_idx = torch.tensor(
                        unfinished_cpu,
                        device=parent_list.device,
                        dtype=torch.int64,
                    )
                    accept_lens_inclusive = torch.tensor(
                        [int(x) + 1 for x in next_draft_input.accept_length_cpu],
                        device=accept_lens_inclusive.device,
                        dtype=torch.int32,
                    )
                    possible_reuse = int((accept_lens_inclusive > 1).to(torch.int32).sum().item())
                    next_verified_per_req = None
                    if isinstance(getattr(next_draft_input, "verified_id", None), torch.Tensor):
                        v = next_draft_input.verified_id
                        need = int(accept_lens_inclusive.to(torch.int64).sum().item())
                        if v.numel() == need:
                            last = (
                                accept_lens_inclusive.to(dtype=torch.int64)
                                .cumsum(dim=0)
                                .sub(1)
                            )
                            next_verified_per_req = v.index_select(0, last).to(torch.int32)
                    accept_index_mat = accept_index_mat[unfinished_idx]
                    parent_list_u = parent_list[unfinished_idx]
                    top_scores_index_u = top_scores_index[unfinished_idx]
                    draft_tokens_u = draft_tokens[unfinished_idx]
                    cand_tokens_full = self._ssd_last_candidate_tokens
                    cand_scores_full = self._ssd_last_candidate_scores
                    cand_tokens_u = (
                        cand_tokens_full[unfinished_idx]
                        if isinstance(cand_tokens_full, torch.Tensor)
                        and cand_tokens_full.shape[0] == bs
                        else None
                    )
                    cand_scores_u = (
                        cand_scores_full[unfinished_idx]
                        if isinstance(cand_scores_full, torch.Tensor)
                        and cand_scores_full.shape[0] == bs
                        else None
                    )
                    if cand_tokens_u is None and isinstance(
                        getattr(self, "_ssd_last_candidate_miss_indices", None), list
                    ):
                        miss_base = self._ssd_last_candidate_miss_indices
                        miss_tokens = self._ssd_last_candidate_miss_tokens
                        miss_scores = self._ssd_last_candidate_miss_scores
                        if isinstance(miss_tokens, torch.Tensor) and miss_tokens.shape[0] == len(
                            miss_base
                        ):
                            pos_map = {int(v): j for j, v in enumerate(miss_base)}
                            row_ids = [
                                int(pos_map.get(int(v), -1)) for v in unfinished_cpu
                            ]
                            have = [i for i, r in enumerate(row_ids) if r >= 0]
                            if len(have) > 0:
                                cand_tokens_u = torch.empty(
                                    (len(unfinished_cpu), int(miss_tokens.shape[1])),
                                    device=miss_tokens.device,
                                    dtype=miss_tokens.dtype,
                                )
                                cand_tokens_u.fill_(-1)
                                take = torch.tensor(
                                    [row_ids[i] for i in have],
                                    device=miss_tokens.device,
                                    dtype=torch.int64,
                                )
                                cand_tokens_u.index_copy_(
                                    0, torch.tensor(have, device=miss_tokens.device, dtype=torch.int64), miss_tokens.index_select(0, take)
                                )
                                if isinstance(miss_scores, torch.Tensor) and miss_scores.shape[0] == len(
                                    miss_base
                                ):
                                    cand_scores_u = torch.empty(
                                        (len(unfinished_cpu), int(miss_scores.shape[1])),
                                        device=miss_scores.device,
                                        dtype=miss_scores.dtype,
                                    )
                                    cand_scores_u.fill_(-1e9)
                                    cand_scores_u.index_copy_(
                                        0, torch.tensor(have, device=miss_scores.device, dtype=torch.int64), miss_scores.index_select(0, take)
                                    )

                    if isinstance(next_verified_per_req, torch.Tensor):
                        if isinstance(cand_tokens_u, torch.Tensor):
                            top_cpu = top_scores_index_u.to("cpu", dtype=torch.int64)
                            draft_cpu = draft_tokens_u.to("cpu", dtype=torch.int64)
                            cand_tok_cpu = cand_tokens_u.to("cpu", dtype=torch.int64)
                            cand_score_cpu = (
                                cand_scores_u.to("cpu", dtype=torch.float32)
                                if isinstance(cand_scores_u, torch.Tensor)
                                else None
                            )
                            root_cpu = next_verified_per_req.to("cpu", dtype=torch.int64).tolist()
                            root_in_pool = 0
                            root_in_selected = 0
                            injected = 0
                            for r in range(len(unfinished_cpu)):
                                tgt = int(root_cpu[r])
                                if tgt in draft_cpu[r].tolist():
                                    root_in_selected += 1
                                    continue
                                if (cand_tok_cpu[r] == tgt).any().item():
                                    root_in_pool += 1
                                    idx_full = int((cand_tok_cpu[r] == tgt).nonzero(as_tuple=False)[0].item())
                                    if cand_score_cpu is not None:
                                        sel_scores = cand_score_cpu[r].gather(0, top_cpu[r].clamp(min=0))
                                        evict = int(sel_scores.argmin().item())
                                    else:
                                        evict = int(top_cpu.shape[1]) - 1
                                    top_cpu[r, evict] = idx_full
                                    draft_cpu[r, evict] = tgt
                                    injected += 1
                            top_scores_index_u = top_cpu.to(device=top_scores_index_u.device, dtype=top_scores_index_u.dtype)
                            draft_tokens_u = draft_cpu.to(device=draft_tokens_u.device, dtype=draft_tokens_u.dtype)
                            if (
                                os.getenv("SGLANG_SSD_V1_ROOT_COV", "false").lower()
                                in ("1", "true")
                                and self._ssd_root_cov_left > 0
                            ):
                                self._ssd_root_cov_left -= 1
                                logger.info(
                                    "[SSD_V1] root_cov selected=%d pool=%d injected=%d bs=%d",
                                    root_in_selected,
                                    root_in_pool,
                                    injected,
                                    int(len(unfinished_cpu)),
                                )
                    pack = self._ssd_build_next_verify_input_by_subtree(
                        parent_list_u,
                        top_scores_index_u,
                        draft_tokens_u,
                        accept_lens_inclusive,
                        accept_index_mat,
                        self.topk,
                        (
                            next_verified_per_req
                            if isinstance(next_verified_per_req, torch.Tensor)
                            else next_draft_input.verified_id.to(torch.int32)
                        ),
                    )
                    if pack is not None:
                        parent_pack, top_pack, draft_pack, reuse_mask = pack
                        rid_hash = self._get_rid_hash_tensor(batch.reqs, parent_list.device)[
                            unfinished_idx
                        ]
                        seq_lens_next_i64 = next_draft_input.seq_lens_for_draft_extend.to(
                            torch.int64
                        )
                        verified_ids_i64 = (
                            next_verified_per_req.to(torch.int64)
                            if isinstance(next_verified_per_req, torch.Tensor)
                            else next_draft_input.verified_id.to(torch.int64)
                        )
                        m = {}
                        for i in range(int(unfinished_idx.numel())):
                            if (
                                isinstance(reuse_mask, torch.Tensor)
                                and reuse_mask.numel() == int(unfinished_idx.numel())
                                and not bool(reuse_mask[i].item())
                            ):
                                continue
                            k = (
                                int(rid_hash[i].item()),
                                int(seq_lens_next_i64[i].item()),
                                int(verified_ids_i64[i].item()),
                            )
                            m[k] = (
                                parent_pack[i].detach(),
                                top_pack[i].detach(),
                                draft_pack[i].detach(),
                                (
                                    cand_tokens_u[i].detach()
                                    if isinstance(cand_tokens_u, torch.Tensor)
                                    else None
                                ),
                                (
                                    cand_scores_u[i].detach()
                                    if isinstance(cand_scores_u, torch.Tensor)
                                    else None
                                ),
                            )
                        if self._ssd_next_verify_map is None:
                            self._ssd_next_verify_map = OrderedDict()
                        self._ssd_next_verify_map.update(m)
                        if isinstance(self._ssd_next_verify_map, OrderedDict):
                            for k in m.keys():
                                self._ssd_next_verify_map.move_to_end(k)
                            while (
                                self._ssd_cache_max_entries > 0
                                and len(self._ssd_next_verify_map)
                                > self._ssd_cache_max_entries
                            ):
                                self._ssd_next_verify_map.popitem(last=False)
                        if (
                            os.getenv("SGLANG_SSD_V1_DEBUG", "false").lower()
                            in ("1", "true")
                            and self._ssd_debug_left > 0
                        ):
                            self._ssd_debug_left -= 1
                            reuse_true = (
                                int(reuse_mask.to(torch.int32).sum().item())
                                if isinstance(reuse_mask, torch.Tensor)
                                and reuse_mask.numel() == int(unfinished_idx.numel())
                                else -1
                            )
                            logger.info(
                                "[SSD_V1] pack bs=%d possible=%d reuse_true=%d map_add=%d map_total=%d unfinished=%d",
                                bs,
                                possible_reuse,
                                reuse_true,
                                len(m),
                                (
                                    len(self._ssd_next_verify_map)
                                    if self._ssd_next_verify_map is not None
                                    else 0
                                ),
                                int(unfinished_idx.numel()),
                            )
            else:
                if (
                    os.getenv("SGLANG_SSD_V1_DEBUG", "false").lower() in ("1", "true")
                    and self._ssd_debug_left > 0
                ):
                    self._ssd_debug_left -= 1
                    logger.info("[SSD_V1] pack=None bs=%d", bs)

        res = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            ssd_cache_hits_sum=int(self._ssd_step_cache_hits_sum),
            ssd_cache_hits_count=int(self._ssd_step_cache_hits_count),
            ssd_subtree_reuse_attempts=int(self._ssd_step_subtree_attempts),
            ssd_subtree_reuse_fallbacks=int(self._ssd_step_subtree_fallbacks),
        )
        self._ssd_step_cache_hits_sum = 0
        self._ssd_step_cache_hits_count = 0
        self._ssd_step_subtree_attempts = 0
        self._ssd_step_subtree_fallbacks = 0
        return res
