"""ExternalWorker: speculative decoding driven by an external drafter.

The draft model runs in a separate process that SGLang does not own.
SGLang receives draft token IDs over a transport interface and runs
chain verification on the target model using the existing mask-driven
verify path.
"""

import logging
import weakref
from array import array
from typing import Optional

import numpy as np
import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func,
)
from sglang.kernels.ops.speculative.reconstruct_tree import (
    reconstruct_indices_from_tree_mask_func,
)
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import (
    get_device,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.external_components.base_transport import (
    ExternalDraftTransport,
)
from sglang.srt.speculative.external_components.contract import (
    DraftContext,
    PrefillContext,
)
from sglang.srt.speculative.external_components.zmq_transport import ZmqTransport
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    alloc_verify_tree_buffers,
    build_grammar_vocab_mask,
    commit_verify,
    derive_tree_links,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)

logger = logging.getLogger(__name__)


class _ChainVerifyInput(NgramVerifyInput):
    @property
    def tree_topk(self) -> int:
        return 1


def _drafter_prompt(req: Req) -> array:
    # Multimodal padding replaces media tokens with cache ids, so the drafter gets
    # the unpadded prompt; a text prompt may have been truncated at admission.
    if req.multimodal_inputs is None:
        return req.origin_input_ids
    return req.origin_input_ids_unpadded


def _finishes_without_drafting(req: Req) -> bool:
    # A requested finish (e.g. an abort at admission) commits on the next result.
    return req.finished() or req.to_finish is not None


class ExternalWorker(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self._tp_rank = get_parallel().tp_rank
        self.page_size = get_schedule().page_size
        self.device = get_device().device

        self.num_draft_tokens: int = get_spec().speculative_num_draft_tokens
        self._num_wire_draft_tokens: int = self.num_draft_tokens - 1

        self._transport: Optional[ExternalDraftTransport] = None
        # tp rank 0 is the sole leader: _handle_external rejects pp_size > 1,
        # dp_size > 1 and DP attention, so only this rank talks to the drafter.
        if self._tp_rank == 0:
            self._transport = ZmqTransport(
                endpoint=get_spec().speculative_external_endpoint,
            )
            self._transport.open()
            self._transport.configure_draft_engine(
                num_draft_tokens=self._num_wire_draft_tokens,
            )

        self._rid_to_wire_id: dict[str, int] = {}
        self._next_wire_id: int = 0
        # rid -> req.retraction_count when its context was sent to the drafter.
        self._prefilled_retraction_counts: dict[str, int] = {}
        # Prefilled at the drafter but in no decode batch yet; a request that
        # is retracted before its first decode step is cancelled from here.
        # Weak, so it is still freed.
        self._prefill_only_reqs: dict[str, weakref.ref[Req]] = {}
        self._prev_decode_rids: set[str] = set()

        self._chain_mask_np = np.tril(
            np.ones((self.num_draft_tokens, self.num_draft_tokens), dtype=np.bool_)
        )

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        return None

    def alloc_memory_pool(self, **kwargs):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        (
            self.draft_tokens_batch,
            self.tree_mask_batch,
            self.retrieve_indexes_batch,
            self.retrieve_next_token_batch,
            self.retrieve_next_sibling_batch,
            self.positions_batch,
        ) = alloc_verify_tree_buffers(
            max_batch_size=self.max_batch_size,
            draft_token_num=self.num_draft_tokens,
            device=self.device,
        )

    def clear_cache_pool(self):
        if self._transport is not None:
            wire_ids = list(self._rid_to_wire_id.values())
            if wire_ids:
                self._transport.cancel_inference(request_ids=wire_ids)
        # Keep _next_wire_id: wire ids are never reused, so a late reply for
        # a cancelled request cannot be mistaken for a new one.
        self._rid_to_wire_id.clear()
        self._prefilled_retraction_counts.clear()
        self._prefill_only_reqs.clear()
        self._prev_decode_rids.clear()

    # -- Main entry point ----------------------------------------------------

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)

        if batch.forward_mode.is_decode():
            return self._forward_decode(batch, on_publish=on_publish)

        return self._forward_extend(batch, on_publish=on_publish)

    # -- Extend (prefill) ----------------------------------------------------

    def _forward_extend(
        self, batch: ScheduleBatch, *, on_publish=None
    ) -> GenerationBatchResult:
        bs = len(batch.reqs)
        batch_result = self.target_worker.forward_batch_generation(batch)
        logits_output = batch_result.logits_output
        predict = batch_result.next_token_ids
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "PREFILL prefill_tokens=%s bonus_tokens=%s",
                [req.origin_input_ids[-1] for req in batch.reqs],
                predict.tolist(),
            )

        if self._tp_rank == 0:
            self._send_prefill_contexts(batch)

        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
        new_seq_lens = batch.seq_lens.clone()
        accept_tokens = torch.zeros(
            bs, self.num_draft_tokens, dtype=torch.int32, device=self.device
        )
        accept_tokens[:, 0] = predict
        accept_tokens = accept_tokens.flatten()

        if on_publish is not None:
            on_publish(new_seq_lens)

        next_draft_input = _ChainVerifyInput(
            draft_token_num=self.num_draft_tokens,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.num_draft_tokens,
        )

    def _send_prefill_contexts(self, batch: ScheduleBatch) -> None:
        cancel_wire_ids = self._pop_finished_prefill_only_reqs()
        prefill_contexts = []
        for req in batch.reqs:
            if _finishes_without_drafting(req):
                continue
            sent_retraction_count = self._prefilled_retraction_counts.get(req.rid)
            if sent_retraction_count == req.retraction_count:
                continue
            if sent_retraction_count is not None:
                # Retracted with no decode step in between, so the drafter
                # still holds the context from before the retraction.
                stale_wire_id = self._rid_to_wire_id.pop(req.rid, None)
                if stale_wire_id is not None:
                    cancel_wire_ids.append(stale_wire_id)
            wire_id = self._get_wire_id(req.rid)
            # A retracted request is re-prefilled with the outputs it kept,
            # so they are part of the context the target holds in KV.
            prefill_contexts.append(
                PrefillContext(
                    request_id=wire_id,
                    prompt_token_ids=[*_drafter_prompt(req), *req.output_ids],
                )
            )
            self._prefilled_retraction_counts[req.rid] = req.retraction_count
            self._prefill_only_reqs[req.rid] = weakref.ref(req)
        if cancel_wire_ids:
            self._transport.cancel_inference(request_ids=cancel_wire_ids)
        if prefill_contexts:
            self._transport.prefill_cache(prefill_contexts=prefill_contexts)

    # -- Decode (draft + verify) ---------------------------------------------

    def _forward_decode(
        self, batch: ScheduleBatch, *, on_publish=None
    ) -> GenerationBatchResult:
        bs = len(batch.reqs)
        self._prepare_verify_input(batch)

        verify_input: NgramVerifyInput = batch.spec_info

        batch_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        grammar_mask = None
        if batch.has_grammar:
            mask_np = np.broadcast_to(
                self._chain_mask_np, (bs, self.num_draft_tokens, self.num_draft_tokens)
            ).reshape(-1)
            retrieve_next_token_cpu, retrieve_next_sibling_cpu = derive_tree_links(
                mask_np, bs, self.num_draft_tokens
            )
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=GrammarTree.from_host(
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    verify_input.draft_token.view(bs, -1).cpu().to(torch.int64),
                ),
                sampling_info=batch.sampling_info,
                device=verify_input.retrieve_next_token.device,
                barrier=None,
            )

        predict, accept_lens, accept_index = eagle_sample(
            verify_input,
            batch,
            logits_output,
            grammar_mask,
            watermark_state=self.model_runner.watermark_state,
        )
        accept_tokens, new_seq_lens = commit_verify(
            batch=batch,
            logits_output=logits_output,
            predict=predict,
            accept_lens=accept_lens,
            accept_index=accept_index,
            watermark_state=self.model_runner.watermark_state,
            target_worker=self.target_worker,
            draft_token_num=self.num_draft_tokens,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VERIFY accept_lens=%s (max=%d, requested=%d)",
                accept_lens.tolist(),
                self.num_draft_tokens,
                self._num_wire_draft_tokens,
            )

        if on_publish is not None:
            on_publish(new_seq_lens)

        self._cancel_departed_requests(batch)

        batch.forward_mode = ForwardMode.DECODE

        next_draft_input = _ChainVerifyInput(
            draft_token_num=self.num_draft_tokens,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.num_draft_tokens,
        )

    # -- Verify input construction -------------------------------------------

    def _prepare_verify_input(self, batch: ScheduleBatch) -> None:
        bs = len(batch.reqs)
        vn = self.num_draft_tokens

        seq_lens, bonus_tokens = self._get_draft_positions(batch)
        drafts = self._fetch_draft_tokens(batch, seq_lens, bonus_tokens)
        if logger.isEnabledFor(logging.DEBUG):
            for req, req_drafts in zip(batch.reqs, drafts):
                logger.debug("DRAFT rid=%s drafts=%s", req.rid, req_drafts.tolist())
        verify_tokens_np = np.empty((bs, vn), dtype=np.int64)
        verify_tokens_np[:, 0] = bonus_tokens
        verify_tokens_np[:, 1:] = drafts
        verify_tokens_np = verify_tokens_np.reshape(-1)

        draft_tokens = self.draft_tokens_batch[bs]
        draft_tokens.copy_(
            torch.from_numpy(verify_tokens_np).to(
                device=self.device, dtype=torch.int64, non_blocking=True
            ),
            non_blocking=True,
        )

        chain_mask_np = np.broadcast_to(self._chain_mask_np, (bs, vn, vn)).reshape(-1)
        tree_mask = self.tree_mask_batch[bs]
        tree_mask.copy_(
            torch.from_numpy(chain_mask_np).to(self.device, non_blocking=True),
            non_blocking=True,
        )

        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]

        reconstruct_indices_from_tree_mask_func(
            tree_mask,
            batch.seq_lens,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            bs,
            vn,
        )

        full_mask_parts: list[torch.Tensor] = []
        mask_reshaped = chain_mask_np.reshape(bs, vn, vn)
        for i in range(bs):
            seq_len = batch.seq_lens_cpu[i]
            prefix = torch.ones((vn, seq_len), dtype=torch.bool, device=self.device)
            suffix = torch.from_numpy(mask_reshaped[i]).to(self.device)
            full_mask_parts.append(torch.cat([prefix, suffix], dim=1).flatten())
        full_mask = torch.cat(full_mask_parts, dim=0)

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + vn,
            batch_size=bs,
            draft_token_num=vn,
            device=self.device,
        )

        prepare_mamba_track_for_verify(batch)

        batch.spec_info = _ChainVerifyInput(
            draft_token=draft_tokens,
            custom_mask=full_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=vn,
        )

    def _get_draft_positions(self, batch: ScheduleBatch) -> tuple[list[int], list[int]]:
        # The previous round's bonus token is in req.output_ids only once its result
        # is processed; this launch is counted, so a count above one means it is
        # not yet. Every request's prefill relays an accept_len of 1 before its decode.
        pending = [req.inflight_steps > 1 for req in batch.reqs]
        if any(pending):
            stride = self.num_draft_tokens
            prev_tokens = batch.spec_info.accept_tokens.cpu().tolist()
            prev_accept_lens = batch.spec_info.accept_lens.cpu().tolist()

        seq_lens = []
        bonus_tokens = []
        for i, (req, seq_len) in enumerate(
            zip(batch.reqs, batch.seq_lens_cpu.tolist())
        ):
            # Multimodal padding is not part of the drafter's context.
            num_padding = len(req.origin_input_ids) - len(_drafter_prompt(req))
            seq_lens.append(int(seq_len) - num_padding)
            if pending[i]:
                assert prev_accept_lens[i] > 0, f"{req.rid=} has no relayed bonus"
                bonus_token = prev_tokens[i * stride + prev_accept_lens[i] - 1]
            else:
                bonus_token = req.output_ids[-1]
            bonus_tokens.append(int(bonus_token))
        return seq_lens, bonus_tokens

    def _fetch_draft_tokens(
        self,
        batch: ScheduleBatch,
        seq_lens: list[int],
        bonus_tokens: list[int],
    ) -> np.ndarray:
        # A request can still be in the batch after it finished (results are
        # processed after the batch is formed); it gets placeholder drafts.
        live = [
            i for i, req in enumerate(batch.reqs) if not _finishes_without_drafting(req)
        ]
        live_drafts = None
        if self._tp_rank == 0:
            wire_ids = [self._get_wire_id(batch.reqs[i].rid) for i in live]
            self._transport.request_drafts(
                draft_contexts=[
                    DraftContext(
                        request_id=wire_id,
                        seq_len=seq_lens[i],
                        bonus_token_id=bonus_tokens[i],
                    )
                    for i, wire_id in zip(live, wire_ids)
                ]
            )
            result = self._transport.receive_drafts(
                request_ids=wire_ids,
                num_draft_tokens=self._num_wire_draft_tokens,
            )
            live_drafts = [result[wire_id] for wire_id in wire_ids]
        return self._broadcast_and_finalize(batch, live, live_drafts)

    def _broadcast_and_finalize(
        self,
        batch: ScheduleBatch,
        live: list[int],
        live_drafts: Optional[list[Optional[list[int]]]],
    ) -> np.ndarray:
        # Requests without drafts get zeros; those the drafter failed are aborted.
        n = self._num_wire_draft_tokens
        bs = len(batch.reqs)
        drafts = np.zeros((bs, n), dtype=np.int64)
        failed_indices = []
        if self._tp_rank == 0:
            for i, req_drafts in zip(live, live_drafts):
                if req_drafts is None:
                    failed_indices.append(i)
                else:
                    drafts[i] = req_drafts

        tp_group = get_parallel().tp_group
        if tp_group.world_size > 1:
            buf = self.draft_tokens_batch[bs][: bs * n]
            if self._tp_rank == 0:
                buf.copy_(torch.from_numpy(drafts.reshape(-1)).to(self.device))
            tp_group.broadcast(buf, src=0)
            drafts = buf.cpu().numpy().reshape(bs, n)

            fail_len = torch.tensor(
                [len(failed_indices)], dtype=torch.int64, device=self.device
            )
            tp_group.broadcast(fail_len, src=0)
            if fail_len.item() > 0:
                fail_buf = torch.tensor(
                    failed_indices, dtype=torch.int64, device=self.device
                )
                if self._tp_rank != 0:
                    fail_buf = torch.empty(
                        int(fail_len.item()), dtype=torch.int64, device=self.device
                    )
                tp_group.broadcast(fail_buf, src=0)
                failed_indices = fail_buf.cpu().tolist()

        for i in failed_indices:
            batch.reqs[i].request_finish(
                FINISH_ABORT("External drafter failed to produce draft tokens")
            )

        return drafts

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        # Called when the request finishes, in prefill or in decode.
        wire_id = self._rid_to_wire_id.pop(rid, None)
        self._prefilled_retraction_counts.pop(rid, None)
        self._prefill_only_reqs.pop(rid, None)
        if wire_id is not None and self._tp_rank == 0:
            self._transport.cancel_inference(request_ids=[wire_id])

    # -- Request tracking ----------------------------------------------------

    def _get_wire_id(self, rid: str) -> int:
        cached = self._rid_to_wire_id.get(rid)
        if cached is not None:
            return cached
        wire_id = self._next_wire_id
        self._next_wire_id += 1
        self._rid_to_wire_id[rid] = wire_id
        return wire_id

    def _cancel_departed_requests(self, batch: ScheduleBatch) -> None:
        cur_rids = {req.rid for req in batch.reqs}
        for rid in cur_rids:
            self._prefill_only_reqs.pop(rid, None)
        cancel_wire_ids = self._pop_finished_prefill_only_reqs()
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                wire_id = self._rid_to_wire_id.pop(req.rid, None)
                if wire_id is not None:
                    cancel_wire_ids.append(wire_id)
                self._prefilled_retraction_counts.pop(req.rid, None)

        departed_rids = self._prev_decode_rids - cur_rids
        for rid in departed_rids:
            wire_id = self._rid_to_wire_id.pop(rid, None)
            if wire_id is not None:
                cancel_wire_ids.append(wire_id)
            self._prefilled_retraction_counts.pop(rid, None)

        if cancel_wire_ids and self._tp_rank == 0:
            self._transport.cancel_inference(request_ids=cancel_wire_ids)

        self._prev_decode_rids = cur_rids

    def _pop_finished_prefill_only_reqs(self) -> list[int]:
        # Wire ids of requests that finished or were retracted before decode.
        wire_ids = []
        for rid, req_ref in list(self._prefill_only_reqs.items()):
            req = req_ref()
            if req is None or req.finished() or req.is_retracted:
                del self._prefill_only_reqs[rid]
                self._prefilled_retraction_counts.pop(rid, None)
                wire_id = self._rid_to_wire_id.pop(rid, None)
                if wire_id is not None:
                    wire_ids.append(wire_id)
        return wire_ids
