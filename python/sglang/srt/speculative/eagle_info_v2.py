from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_alloc_reserve_per_decode,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
    prepare_mamba_track_for_verify,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_draft_cache_locs_contiguous as assign_draft_cache_locs_contiguous,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs_func as assign_extend_cache_locs_func,
)
from sglang.srt.speculative.triton_ops.eagle import (
    fill_bonus_tokens as fill_bonus_tokens,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_nan,
    maybe_detect_oob,
    sanitize_nan_logits,
)
from sglang.srt.utils.common import is_cuda, is_hip, is_musa, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import (
        EagleDraftExtendInput,
        EagleDraftInput,
        EagleVerifyInput,
    )

if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:
    """Copy the prefix partial-tail page into each branch's first-page holes (page>1 + topk>1).

    The draft-decode expand pass reads each branch's own draft page by block id
    (cache_loc // page_size), so branch b>=1's hole slots [0, last_page) must hold the
    real prefix tail (branch 0's first page already is it). Mirrors V1 #7725.
    """
    if topk <= 1:
        return
    bs = rows.shape[0]
    page_off = torch.arange(page_size, device=rows.device, dtype=torch.int64)
    branches = torch.arange(1, topk, device=rows.device, dtype=torch.int64).view(
        1, topk - 1, 1
    )
    # Source: the prefix tail page [prefix_base, prefix_base + page_size), one per branch.
    src_pos = (prefix_base.view(bs, 1, 1) + page_off.view(1, 1, page_size)).expand(
        bs, topk - 1, page_size
    )
    # Target: branch b's first page [prefix_base + b*num_new_pages*page, + page_size).
    tgt_pos = (
        prefix_base.view(bs, 1, 1)
        + branches * (num_new_pages.view(bs, 1, 1) * page_size)
        + page_off.view(1, 1, page_size)
    )
    # Only [0, last_page) holds real prefix KV; [last_page, page_size) are the branch's
    # own draft slots and must not be overwritten.
    vmask = (page_off.view(1, 1, page_size) < last_page.view(bs, 1, 1)).expand(
        bs, topk - 1, page_size
    )
    src_slots = torch.gather(rows, 1, src_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    tgt_slots = torch.gather(rows, 1, tgt_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    if src_slots.numel() > 0:
        token_to_kv_pool.move_kv_cache(tgt_slots, src_slots)


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Accumulate penalty
        # This is a relaxed version of penalties for speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        double_alloc = get_alloc_reserve_per_decode()

        cur_kv_lens = [0] * bs
        nxt_kv_lens = [0] * bs
        num_needed_tokens = 0
        for i, r in enumerate(batch.reqs):
            cur = r.kv_allocated_len
            # max(cur, ...) clamps so adaptive downswitch (smaller alloc_len_per_decode)
            # cannot make nxt < cur and corrupt allocator state. kv_committed_len lags
            # batch.seq_lens by ~1 verify in overlap mode, so we react to adaptive
            # switches one batch later than a seq_lens-based baseline; the 2*alloc
            # over-allocation buffer absorbs that lag.
            nxt = max(cur, r.kv_committed_len + double_alloc)
            cur_kv_lens[i] = cur
            nxt_kv_lens[i] = nxt
            num_needed_tokens += nxt - cur
            r.kv_allocated_len = nxt
            r.decode_batch_idx += 1
            # Pre-claim bonus slot here (like normal decode); resolve subtracts 1.
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

        # Fail fast if the page>1 + topk>1 draft over-allocation
        # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
        # would OOB and free would leak KV. The row is widened to hold it in _init_pools
        # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.
        from sglang.srt.server_args import get_global_server_args

        if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
            max_alloc_len = int(nxt_kv_lens_cpu.max())
            row_width = batch.req_to_token_pool.req_to_token.shape[1]
            assert max_alloc_len <= row_width, (
                f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
                f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
                f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
            )

        # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
        # barrier has chained to the prev forward -> host stalls a full forward.
        cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens_device,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens_device,
                cur_kv_lens_cpu,
                nxt_kv_lens_device,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_device,
            nxt_kv_lens_device,
            out_cache_loc,
            bs,
        )

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # Assign cache locations (draft-write targets).
            page_size = batch.token_to_kv_pool_allocator.page_size
            if page_size == 1 or topk == 1:
                batch.out_cache_loc = torch.empty(
                    (bs * topk * num_steps,),
                    dtype=torch.int64,
                    device=batch.device,
                )
                # FIXME(lsyin): align with the default code path
                assign_draft_cache_locs_contiguous[(bs,)](
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
            else:
                # page_size > 1 + topk > 1: per-branch page-aligned draft pages.
                # Reduce out_cache_loc from the page-aligned tree region down to the
                # dense draft slots (skip each branch's duplicated prefix-tail slots
                # and trailing padding), matching generate_draft_decode_kv_indices'
                # paged read formula: prefix_base + t*num_new_pages*page + last_page + s.
                # base is batch.seq_lens (== KV-ready committed prefix at draft time;
                # the bonus is the tree root written by verify, not part of [0:seq_lens]).
                rows = req_to_token_pool.req_to_token[batch.req_pool_indices.long()]
                seq_lens = batch.seq_lens.to(torch.int64)
                last_page = seq_lens % page_size
                prefix_base = seq_lens - last_page
                num_new_pages = (last_page + num_steps + page_size - 1) // page_size
                topk_ids = torch.arange(
                    topk, device=rows.device, dtype=torch.int64
                ).view(1, topk)
                starts = (
                    prefix_base.view(bs, 1)
                    + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                    + last_page.view(bs, 1)
                )
                steps = torch.arange(
                    num_steps, device=rows.device, dtype=torch.int64
                ).view(1, 1, num_steps)
                pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
                batch.out_cache_loc = (
                    torch.gather(rows, 1, pos).reshape(-1).contiguous()
                )

                # Each branch's page-aligned region starts with `last_page` hole slots
                # overlapping the prefix tail page; duplicate the real prefix-tail KV
                # into them so whole-page reads stay coherent (see helper docstring).
                duplicate_prefix_tail_to_draft_branches(
                    draft_model_runner.token_to_kv_pool,
                    rows,
                    prefix_base,
                    last_page,
                    num_new_pages,
                    topk,
                    page_size,
                )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph


class EagleDraftExtendInputV2Mixin:
    def prepare_for_extend_to_fill_draft_kvcache(
        self: EagleDraftExtendInput,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        bs = len(batch.seq_lens)
        extend_num_tokens = bs * num_draft_tokens
        # When seq_lens_cpu is absent, stay on GPU-only path -- no .tolist()/.cpu().
        gpu_only = batch.seq_lens_cpu is None

        batch.spec_info = self
        batch.input_ids = predict
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_extend_to_fill_draft_kvcache input_ids",
        )
        # init_new requires both list or both Tensor;
        # gpu_only emits device tensors to skip H2D.
        if gpu_only:
            batch.prefix_lens = batch.seq_lens.to(torch.int32)
            batch.extend_lens = torch.full(
                (bs,), num_draft_tokens, dtype=torch.int32, device=batch.seq_lens.device
            )
        else:
            batch.prefix_lens = batch.seq_lens_cpu.tolist()
            batch.extend_lens = [num_draft_tokens] * bs
        batch.extend_num_tokens = extend_num_tokens
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        # Forward sees post-write length (draft extend writes num_draft_tokens
        # slots); mutation stays on forward_batch to preserve SB.seq_lens.
        forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
        if not gpu_only:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
        else:
            # Supply CPU mirror (extend_seq_lens are all num_draft_tokens) so
            # backend max() reads from list without a per-iter D2H sync.
            forward_batch.extend_seq_lens_cpu = [num_draft_tokens] * bs
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
            # Planned pre-pad; do NOT opt into post-pad re-plan. DSA's indexer
            # cannot rebuild its deep_gemm schedule_meta on a DP-padded batch
            # (the `_batch_size == batch_size` assertion, see #27091); the
            # marked pre-pad metadata is used as-is, matching the proven
            # skip_attn_backend_init=True behavior.
            # On NPU with --disable-cuda-graph, block_table shape won't match
            # after prepare_mlp_sync_batch padding; defer re-init to
            # forward_extend (post-pad) instead.
            if not _is_npu or can_cuda_graph:
                forward_batch.mark_forward_metadata_ready()
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    @property
    def max_tree_depth(self: EagleVerifyInput) -> int:
        """Longest root-to-leaf chain of the verify tree, incl. the root;
        bounds the accept_index row width. EAGLE trees are depth-bounded by
        the draft loop. Algorithms with other tree shapes override this."""
        return self.spec_steps + 1

    @property
    def tree_topk(self: EagleVerifyInput) -> int:
        """Branching factor passed to the tree-verify kernels; -1 means an
        irregular tree (no fixed per-level branching)."""
        return self.topk

    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            maybe_detect_oob(
                batch.input_ids,
                0,
                batch.model_config.vocab_size,
                "v2 prepare_for_verify input_ids",
            )
            device = batch.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

            prepare_mamba_track_for_verify(batch)

            # TBO's split_spec_info reads these; no-verify-sync leaves both None.
            self.seq_lens_cpu = batch.seq_lens_cpu
            self.seq_lens_sum = (
                int(batch.seq_lens_cpu.sum())
                if batch.seq_lens_cpu is not None
                else None
            )

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        capture_mode = (
            CaptureHiddenMode.NULL
            if target_worker.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.capture_hidden_mode = capture_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.replay_prepare(
                verify_forward_batch
            )
            verify_forward_batch.mark_forward_metadata_ready()
        # Non-cuda-graph: defer init to forward_extend, which runs after
        # `_forward_raw -> prepare_mlp_sync_batch` pads the batch. Initing
        # here would use pre-pad shapes and trip DSv4 indexer shape match.

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        device = batch.device
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=device)
            num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
            accept_index = torch.empty(0, dtype=torch.int32, device=device)
            return predict, num_correct_drafts, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits

        sanitize_nan_logits(next_token_logits, "verify: target model logits")

        # Apply penalty
        # This is a relaxed version of penalties for speculative decoding.
        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.draft_token_num, dim=0
                )
            )
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.draft_token_num, dim=0
                )
            )

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.max_tree_depth), -1, dtype=torch.int32, device=device
        )
        num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.tree_topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)

        if SIMULATE_ACC_LEN > 0:
            # Do simulation. The helper builds (and returns) a replacement
            # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
            # to keep the simulated width identical to the real one.
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_correct_drafts=num_correct_drafts,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.max_tree_depth - 1,
            )

        # `num_correct_drafts` stays drafts-only inside this function; the returned
        # tensor includes the trailing/bonus token via out-of-place +1 so the
        # name no longer flips semantics mid-function (naming doc C2).
        return predict, num_correct_drafts + 1, accept_index
