from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import assign_draft_cache_locs_contiguous_cpu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import (
        EagleDraftExtendInput,
        EagleDraftInput,
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


class EagleDraftWorkerBase(ABC):
    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_attention_backends(self):
        """Subclasses wrap this with their context managers (draft_tp_context,
        speculative_moe_backend_context, etc.) rather than reimplementing it."""
        self.draft_worker.init_attention_backends()
        self.init_attention_backend()

    def init_cuda_graphs(self):
        """Capture draft graphs (decode disabled on the draft TpModelWorker)."""
        self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        self._capture_cuda_graphs()

    def prepare_for_draft_extend(
        self,
        draft_extend_input: EagleDraftExtendInput,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
        *,
        return_hidden_states_before_norm: bool,
    ):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.utils.async_probe import maybe_detect_oob
        from sglang.srt.utils.common import is_npu

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
        self,
        draft_input: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_draft_cache_locs_contiguous,
        )
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

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


class BaseSpecWorker(ABC):
    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase | TpModelWorker]:
        # dflash / dspark drive the draft model through a plain TpModelWorker;
        # ngram has no draft worker at all (returns None via its override).
        return self._draft_worker

    @property
    def war_fastpath_runner(self):
        # The runner that runs the step's LAST shared-buffer-reading phase --
        # it owns the read-done event the scheduler's WAR barrier waits on.
        # Default is the target runner; override if the last phase runs
        # elsewhere (eagle's draft_extend runs on the draft runner).
        return self.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Attn backends touched by spec_v2 forward; OR-ed by decide_needs_cpu_seq_lens.
        Default returns target only; subclasses extend with draft backends."""
        return (self.target_worker.model_runner.attn_backend,)

    def clear_cache_pool(self):
        """Default no-op: the allocator and kv cache pool are shared with the
        target worker and cleared by the scheduler."""
        # TODO: move this method to BaseTpWorker and call through self.model_runner
        pass

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if self.draft_worker is not None:
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        pass

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        """Hook called by the batch-result processor when a request finishes.

        Default no-op. DSpark overrides this to settle / censor its
        block-accept estimator state for the finished request.
        """
        pass

    def activate_step_by_batch(self, batch_size: int) -> None:
        """Activate the optimal adaptive step for the current batch size.

        Default no-op. Adaptive-aware workers override this to switch
        the runtime state before each draft round.
        """
        pass
