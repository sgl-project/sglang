from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_draft_cache_locs_contiguous,
)
from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
    get_draft_recurrent_hidden_state_spec,
)
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    move_accept_tokens_to_target_kvcache,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()

if _is_cpu:
    from sgl_kernel import assign_draft_cache_locs_contiguous_cpu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.base_spec_worker import (
        BaseSpecWorker,
        EagleDraftWorkerBase,
    )
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


class EagleWorkerContext(msgspec.Struct, frozen=True, kw_only=True):
    """Stable collaborators + capability flags for the eagle worker-step
    functions in this module.

    A derived, frozen VIEW of the owning worker's state, not a second source
    of truth: the worker's ``self.*`` stays authoritative, and the context is
    rebuilt unconditionally at the end of ``alloc_memory_pool``.

    Membership: a worker builds this iff it runs the eagle pipeline (draft
    tree proposal -> ``EagleVerifyInput`` verify -> draft extend). Other spec
    families (ngram / dflash / dspark) never build it; they define their own
    context type if they ever need one -- this one is never subclassed or
    extended with Optional fields.

    Rules (do not relax without a design round):
    - Fields are identity-stable handles or true constants; values rewritten
      at runtime (adaptive ``topk`` / ``num_steps`` / ``num_draft_tokens``)
      stay per-call arguments.
    - No methods besides ``build``, no callable-hook fields; behavior lives
      in module functions. Variation lanes: prove a no-op gate and merge it,
      compose at the worker wrapper, or write a separate pipeline function.
    - Capability flags are named for what the worker IS (not which code block
      to skip) and must carry a removal path.
    """

    draft_worker: EagleDraftWorkerBase
    target_worker: TpModelWorker
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    # get_plan_stream's pair: a device Stream or None, and its enter context.
    plan_stream: Any
    plan_stream_ctx: contextlib.AbstractContextManager
    device: str
    # Capability flags (see class docstring for the admission rules).
    # Marks verify forward-metadata ready pre-pad unconditionally (multi-layer
    # eagle behavior); False relies on eagle_prepare_for_verify marking it only
    # when the cuda-graph load_batch path ran. Removal: metadata-ready adopt
    # fix after GPU validation.
    preplans_verify_metadata: bool
    # Compacts the accepted tree path to the front of each per-req block for
    # topk > 1 (single-layer eagle behavior). Removal: adopt for multi-layer
    # after GPU validation.
    compacts_accept_path: bool
    # False only for STANDALONE drafting, which skips hidden states end-to-end.
    captures_hidden_states: bool

    @classmethod
    def build(cls, worker: BaseSpecWorker) -> EagleWorkerContext:
        """Snapshot the worker's collaborators into a frozen view; call at
        the end of ``alloc_memory_pool`` (asserted below)."""
        assert worker.req_to_token_pool is not None, (
            "EagleWorkerContext.build before pool allocation; "
            "build it at the end of alloc_memory_pool"
        )
        return cls(
            draft_worker=worker.draft_worker,
            target_worker=worker.target_worker,
            req_to_token_pool=worker.req_to_token_pool,
            token_to_kv_pool_allocator=worker.token_to_kv_pool_allocator,
            plan_stream=worker.plan_stream,
            plan_stream_ctx=worker.plan_stream_ctx,
            device=worker.device,
            preplans_verify_metadata=worker._preplans_verify_metadata,
            compacts_accept_path=worker._compacts_accept_path,
            captures_hidden_states=not worker.speculative_algorithm.is_standalone(),
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


def prepare_for_draft_extend(
    draft_extend_input: EagleDraftExtendInput,
    batch: ScheduleBatch,
    predict: torch.Tensor,
    num_draft_tokens: int,
    draft_model_runner: Any,
    cuda_graph_runner: Any,
    *,
    return_hidden_states_before_norm: bool,
):
    bs = len(batch.seq_lens)
    extend_num_tokens = bs * num_draft_tokens
    # When seq_lens_cpu is absent, stay on GPU-only path -- no .tolist()/.cpu().
    gpu_only = batch.seq_lens_cpu is None

    batch.spec_info = draft_extend_input
    # Do NOT cast predict dtype here. The caller (e.g., _draft_extend_for_decode)
    # may run this under a plan stream; casting inside the plan stream creates a
    # cross-stream dependency that can lead to data races and break MTP acceptance.
    # The caller should cast to int64 before entering the plan stream context.
    batch.input_ids = predict
    maybe_detect_oob(
        batch.input_ids,
        0,
        batch.model_config.vocab_size,
        "v2 prepare_for_draft_extend input_ids",
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
    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=return_hidden_states_before_norm,
    )
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
    can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
        forward_batch
    )
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
        if not is_npu() or can_cuda_graph:
            forward_batch.mark_forward_metadata_ready()
    return forward_batch


def prepare_for_draft(
    draft_input: EagleDraftInput,
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
            if _is_cpu:
                assign_draft_cache_locs_contiguous_cpu(
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
            else:
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
            topk_ids = torch.arange(topk, device=rows.device, dtype=torch.int64).view(
                1, topk
            )
            starts = (
                prefix_base.view(bs, 1)
                + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                + last_page.view(bs, 1)
            )
            steps = torch.arange(num_steps, device=rows.device, dtype=torch.int64).view(
                1, 1, num_steps
            )
            pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
            batch.out_cache_loc = torch.gather(rows, 1, pos).reshape(-1).contiguous()

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
    # Actual width of the next draft-decode forward: topk tokens per req.
    draft_input.num_tokens_per_req = topk
    draft_input.num_tokens_for_logprob_per_req = topk
    capture_mode = (
        CaptureHiddenMode.NULL
        if draft_model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.LAST
    )
    draft_input.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=False,
    )
    can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
        forward_batch
    )
    return forward_batch, can_cuda_graph


def build_eagle_verify_input(
    batch: ScheduleBatch,
    draft_input: EagleDraftInput,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    *,
    target_worker: TpModelWorker,
    topk: int,
    num_steps: int,
    num_draft_tokens: int,
    tree_mask_mode: TreeMaskMode,
    device: str,
) -> EagleVerifyInput:
    """Shared draft() tail: idle input, tree-mask build, EagleVerifyInput assembly.

    ``draft_probs`` is the caller's source of draft distributions (single-layer
    eagle: this round's draft_forward output; multi-layer eagle: the ones the
    draft input carried).
    """
    if batch.forward_mode.is_idle():
        return EagleVerifyInput.create_idle_input(
            topk,
            num_steps,
            num_draft_tokens,
            device,
        )

    # Build tree mask
    # Directly write to cuda graph buffers for verify attn
    tree_mask_buf, position_buf = (
        target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
    )

    # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
    # tree mask; over-size is safe. Skip per-iter .sum().item() D2H via UB.
    seq_lens_sum = batch.seq_lens_sum
    if seq_lens_sum is None:
        if tree_mask_buf is None:
            max_context_len = target_worker.model_runner.attn_backend.max_context_len
            seq_lens_sum = batch.seq_lens.shape[0] * max_context_len
        else:
            # tree_mask_buf preallocated -> kernel ignores seq_lens_sum.
            seq_lens_sum = 0

    (
        tree_mask,
        position,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        draft_input.bonus_tokens,
        parent_list,
        top_scores_index,
        draft_tokens,
        batch.seq_lens,
        seq_lens_sum,
        topk,
        num_steps,
        num_draft_tokens,
        tree_mask_mode,
        tree_mask_buf,
        position_buf,
    )

    return EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=tree_mask,
        positions=position,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=None,
        spec_steps=num_steps,
        topk=topk,
        draft_token_num=num_draft_tokens,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        draft_probs=draft_probs,
    )


def _finalize_accept_tree_path(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    logits_output: Any,
    bs: int,
    *,
    token_to_kv_pool_allocator: Any,
    num_draft_tokens: int,
) -> torch.Tensor:
    """Tree drafting (topk > 1): move the accepted path -- KV slots, predict,
    hidden_states -- to the contiguous front of each per-req block, which the
    downstream chain-layout code (draft-extend select_index, committed-KV reads)
    assumes. Returns compacted predict; mutates logits_output.hidden_states
    (moved only when present)."""
    move_accept_tokens_to_target_kvcache(
        batch, accept_index, accept_lens - 1, token_to_kv_pool_allocator
    )
    predict = _compact_accept_to_front(
        predict, accept_index, bs, num_draft_tokens=num_draft_tokens
    )
    if logits_output.hidden_states is not None:
        logits_output.hidden_states = _compact_accept_to_front(
            logits_output.hidden_states,
            accept_index,
            bs,
            num_draft_tokens=num_draft_tokens,
        )
    return predict


def _compact_accept_to_front(
    x: torch.Tensor,
    accept_index: torch.Tensor,
    bs: int,
    *,
    num_draft_tokens: int,
) -> torch.Tensor:
    """Gather the accepted tree path to the front of each per-req block.

    ``x`` is node-indexed over the whole tree (``[bs * num_draft_tokens, ...]``),
    ``accept_index`` is ``[bs, spec_steps + 1]`` global node indices (-1 padded).
    Padded entries clamp to node 0 but land past accept_lens (never read);
    trailing unaccepted slots stay and are freed as overshoot.
    """
    nd = num_draft_tokens
    s1 = accept_index.shape[1]  # spec_steps + 1
    safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
    gathered = x[safe]
    out = x.clone()
    out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
    return out


def run_eagle_verify(
    batch: ScheduleBatch,
    ctx: EagleWorkerContext,
    *,
    topk: int,
    num_steps: int,
    num_draft_tokens: int,
) -> GenerationBatchResult:
    """Shared verify step: target-verify forward, sampling, acceptance bookkeeping.

    The single-layer eagle verify body is the source of truth (superset); the
    two ``ctx`` capability flags encode the multi-layer worker's
    preserved-verbatim differences (see EagleWorkerContext's field comments).
    """
    target_worker = ctx.target_worker
    req_to_token_pool = ctx.req_to_token_pool
    token_to_kv_pool_allocator = ctx.token_to_kv_pool_allocator
    plan_stream = ctx.plan_stream
    plan_stream_ctx = ctx.plan_stream_ctx
    device = ctx.device

    fwd_stream = torch.get_device_module(device).current_stream()
    verify_input: EagleVerifyInput = batch.spec_info
    record_stream_for_v2_verify(batch, verify_input, fwd_stream)

    bs = len(batch.seq_lens)

    # Batch 1: Target verify
    # Prepare for target verify in a separate stream
    with plan_stream_ctx:
        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            req_to_token_pool,
            batch,
            target_worker,
        )

    # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
    record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

    # Correct some buffers due to the overlap plan
    if plan_stream:
        torch.get_device_module(device).current_stream().wait_stream(plan_stream)
        if (
            _is_npu
            and target_worker.model_runner.model_config.model_is_mrope
            and batch.spec_info is not None
            and getattr(batch.spec_info, "positions", None) is not None
            and not batch.forward_mode.is_idle()
        ):
            # mrope_position depends on draft output in default stream and is computed in plan stream,
            # causing errors. Compute it here for correct values.
            verify_forward_batch.compute_spec_mrope_positions(
                target_worker.model_runner, batch
            )

        # Some values such as custom_mask and position depend on the output of draft,
        # so the previous plan step used the wrong values. Here, we need to run the related
        # computation again to update them to the correct values.
        target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
            verify_input,
            (
                target_worker.model_runner.decode_cuda_graph_runner.bs
                if can_run_cuda_graph
                else None
            ),
        )

    # Prepare grammar data on CPU if needed
    if batch.has_grammar:
        retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
        retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
        draft_tokens_cpu = verify_input.draft_token.view(
            verify_input.retrieve_next_token.shape
        ).cpu()

    if ctx.preplans_verify_metadata:
        # Multi-layer eagle preserved-verbatim behavior: metadata init is
        # skipped here unconditionally, although eagle_prepare_for_verify
        # only plans when cuda-graph load_batch ran. Single-layer eagle
        # re-inits the non-graph path instead (post-pad); multi-layer has
        # not adopted that fix. On NPU with --disable-cuda-graph, non-graph
        # verify needs metadata init in forward_extend (post-pad); only
        # mark ready for the cuda-graph path.
        if not _is_npu or can_run_cuda_graph:
            verify_forward_batch.mark_forward_metadata_ready()

    # Run target verify batch in the main compute stream (GPU compute).
    # Metadata init is skipped iff cuda-graph already ran load_batch —
    # eagle_prepare_for_verify marked the batch in exactly that case; the
    # non-cuda-graph path stays unmarked and gets forward_extend's init
    # (post-pad).
    forward_batch_output = target_worker.forward_batch_generation(
        batch=None,
        forward_batch=verify_forward_batch,
        is_verify=True,
    )
    logits_output = forward_batch_output.logits_output

    # Generate vocab mask for constrained decoding
    vocab_mask = None
    if batch.has_grammar:
        # Generate the logit mask for structured output.
        vocab_mask = generate_token_bitmask(
            batch.reqs,
            verify_input,
            retrieve_next_token_cpu,
            retrieve_next_sibling_cpu,
            draft_tokens_cpu,
            batch.sampling_info.vocab_size,
        )

        if vocab_mask is not None:
            assert verify_input.grammar is not None
            vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
            # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
            # and will be applied to produce wrong results
            batch.sampling_info.vocab_mask = None

    # Sample
    maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
    maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
    (
        predict,
        accept_lens,
        accept_index,
    ) = eagle_sample(verify_input, batch, logits_output, vocab_mask)
    new_seq_lens = batch.seq_lens + accept_lens
    clear_unaccepted_c128 = getattr(
        token_to_kv_pool_allocator.get_kvcache(),
        "clear_unaccepted_c128_draft_states",
        None,
    )
    if clear_unaccepted_c128 is not None and not batch.forward_mode.is_idle():
        clear_unaccepted_c128(
            batch.req_pool_indices,
            batch.seq_lens,
            accept_lens,
            num_draft_tokens,
        )

    # Update mamba state for hybrid GDN models after verification
    commit_mamba_states_after_verify(
        target_worker,
        batch,
        accept_lens,
        accept_index,
        num_draft_tokens,
    )

    if not batch.forward_mode.is_idle():
        accept_tokens = predict[accept_index]
        bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
        # stride = accept_tokens per-req width = accept_index.shape[1]
        # (spec_steps + 1); NOT num_draft_tokens, wrong for topk > 1 trees.
        fill_bonus_tokens_func(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_index.shape[1],
            bs,
        )
    else:
        bonus_tokens = torch.empty((0,), device=device, dtype=torch.int32)

    if batch.return_logprob and not batch.forward_mode.is_idle():
        compute_spec_v2_logprobs(batch, logits_output, predict, accept_index, num_steps)

    if ctx.compacts_accept_path and not batch.forward_mode.is_idle() and topk > 1:
        # topk == 1 needs nothing here: the accepted path is already the front
        # chain, so the whole compaction is an identity transform.
        predict = _finalize_accept_tree_path(
            batch,
            accept_index,
            accept_lens,
            predict,
            logits_output,
            bs,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            num_draft_tokens=num_draft_tokens,
        )

    next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

    # verify_forward_batch transitively holds verify-time GPU tensors
    # (draft_token / out_cache_loc / ...) that must outlive the imminent
    # batch.input_ids rebind in prepare_for_draft_extend.
    # Scheduler pins it in batch_record_buf for the 2-iter window.
    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=predict,
        can_run_cuda_graph=can_run_cuda_graph,
        speculative_num_draft_tokens=num_draft_tokens,
        next_draft_input=next_draft_input,
        accept_lens=accept_lens,
        new_seq_lens=new_seq_lens,
        routed_experts_output=forward_batch_output.routed_experts_output,
        indexer_topk_output=forward_batch_output.indexer_topk_output,
        extra_keep_alive_refs=[verify_forward_batch],
    )


def ensure_idle_draft_input(
    batch: ScheduleBatch, ctx: EagleWorkerContext, idle_topk: int
) -> None:
    """Fill ``batch.spec_info`` with an idle ``EagleDraftInput`` if the batch
    carries none (idle forward on this rank)."""
    if batch.spec_info is not None:
        return
    capture_mode = (
        CaptureHiddenMode.LAST if ctx.captures_hidden_states else CaptureHiddenMode.NULL
    )
    hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
        ctx.draft_worker.draft_runner
    )
    batch.spec_info = EagleDraftInput.create_idle_input(
        device=ctx.device,
        hidden_size=hidden_size,
        dtype=hidden_dtype,
        topk=idle_topk,
        capture_hidden_mode=capture_mode,
        vocab_size=ctx.target_worker.model_config.vocab_size,
    )


def eagle_forward_generation(
    batch: ScheduleBatch,
    on_publish: Any,
    ctx: EagleWorkerContext,
    *,
    topk: int,
    num_steps: int,
    num_draft_tokens: int,
    idle_topk: int,
) -> GenerationBatchResult:
    """Shared forward step: target prefill + draft prefill, or draft -> verify
    -> draft extend.

    The single-layer eagle body is the source of truth. Worker-specific parts
    stay in the worker wrappers: single-layer's num_steps == 0 trivial-verify
    path and the adaptive activate hook run before delegating here; draft-side
    context wrapping comes from ``draft_worker.draft_stage_ctx``. ``idle_topk``
    is the idle draft input's topk_p/topk_index width (single-layer: topk;
    multi-layer: topk * num_steps).
    """
    draft_worker = ctx.draft_worker
    target_worker = ctx.target_worker

    if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        # Target prefill
        target_capture_mode = (
            CaptureHiddenMode.FULL
            if ctx.captures_hidden_states
            else CaptureHiddenMode.NULL
        )
        batch_output = target_worker.forward_batch_generation(
            batch, capture_hidden_mode=target_capture_mode
        )

        # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
        # Extend processed L prompt tokens; next verify iter expects same L.
        batch_output.new_seq_lens = batch.seq_lens
        # Publish before draft_extend so the fence is at target-end.
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        # Draft prefill
        with draft_worker.draft_stage_ctx("draft_extend"):
            batch_output.next_draft_input = draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
                batch_output.logits_output.mm_input_embeds,
            )
            return batch_output
    else:
        ensure_idle_draft_input(batch, ctx, idle_topk)
        with draft_worker.draft_stage_ctx("draft"):
            verify_input: EagleVerifyInput = draft_worker.draft(batch)
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        batch_output = run_eagle_verify(
            batch,
            ctx,
            topk=topk,
            num_steps=num_steps,
            num_draft_tokens=num_draft_tokens,
        )
        # Publish before draft_extend so the fence is at verify-end.
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        with draft_worker.draft_stage_ctx("draft_extend"):
            draft_worker._draft_extend_for_decode(batch, batch_output)

        return batch_output
