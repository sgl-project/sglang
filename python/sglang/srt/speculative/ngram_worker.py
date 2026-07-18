import logging
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func as assign_extend_cache_locs_func,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan

_is_cpu = is_cpu()

logger = logging.getLogger(__name__)


USE_FULL_MASK = True


class NGRAMWorker(BaseSpecWorker):
    def alloc_memory_pool(self, **kwargs):
        # The target memory pool does not exist yet when __init__ runs.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.enable_overlap = not server_args.disable_overlap_schedule
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = ps.tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        # req_to_token_pool / token_to_kv_pool_allocator are set in
        # alloc_memory_pool(), after the target pools are allocated.
        self.device = server_args.device

        self.adaptive_controller = None
        # rids of the last decode batch; used to erase corpus match state for
        # requests that left the batch (see forward_batch_generation).
        self._prev_decode_rids: set = set()

        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        if server_args.speculative_ngram_external_corpus_path is not None:
            from sglang.srt.speculative.cpp_ngram.external_corpus import (
                iter_external_corpus_chunks,
            )

            corpus_path = server_args.speculative_ngram_external_corpus_path
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            loaded = self.add_external_corpus(corpus_path, chunks)
            self.commit_corpus_load(corpus_path, loaded)
            logger.info(
                "Loaded external ngram corpus '%s' (%d tokens).",
                corpus_path,
                loaded,
            )

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        # NGRAM has no draft model; drafts come from the CPU-side corpus.
        return None

    def clear_cache_pool(self):
        self.ngram_corpus.reset()
        self._prev_decode_rids = set()

    def update_weights_from_tensor(self, recv_req):
        # NGRAM has no draft weights of its own — the n-gram corpus is a CPU
        # lookup structure built from request token streams — and its
        # `model_runner` is shared with the target worker. The scheduler
        # mixin dispatches via `self.draft_worker or self.tp_worker`, so
        # without this method any caller of `update_weights_from_tensor`
        # under `--speculative-algorithm NGRAM` raises AttributeError.
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        return self.ngram_corpus.load_external_corpus_named(corpus_id, token_chunks)

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        self.ngram_corpus.commit_external_corpus_load(corpus_id, loaded_token_count)

    def remove_external_corpus(self, corpus_id: str) -> None:
        self.ngram_corpus.remove_external_corpus(corpus_id)

    def list_external_corpora(self) -> dict[str, int]:
        return self.ngram_corpus.list_external_corpora()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrieve_next_token_batch = []
        self.retrieve_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrieve_next_token_batch.append(self.retrieve_next_token[:bs, :])
            self.retrieve_next_sibling_batch.append(self.retrieve_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        # Signature must match BaseSpecWorker.on_verify_complete_cpu; the
        # result processor calls it with batch_size as a keyword argument.
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = len(batch.reqs)
        stride = self.draft_token_num

        prev_token_ids, prev_accept_lens = (
            batch.spec_info.accept_tokens,
            batch.spec_info.accept_lens,
        )
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        # Worker-level staging: written here at draft prep, consumed by
        # _update_ngram_corpus after verify within the same forward call.
        self.prev_token_ids = prev_token_ids.tolist()
        self.prev_accept_lens = prev_accept_lens.tolist()

        self.ngram_corpus.synchronize()
        req_ids = []
        batch_tokens = []
        total_lens = []
        assert len(batch.reqs) == len(self.prev_accept_lens)
        # Overlap mode processes results one iteration behind, so the last
        # round's accepted tokens are not yet in req.output_ids and must be
        # spliced in from spec_info. Sync mode and grammar batches process
        # results before the next draft prep, so output_ids is already
        # complete and splicing would duplicate the tail.
        use_prev_tokens = self.enable_overlap and not batch.has_grammar
        i = 0
        for req in batch.reqs:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            check_token = self._efficient_concat_last_n(
                list(req.origin_input_ids),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            req_ids.append(req.rid)
            batch_tokens.append(check_token)
            i += 1
            total_lens.append(
                len(req.origin_input_ids) + len(req.output_ids) + len(prev_tokens)
            )
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        # Decode-only: extend goes through the plain target forward, and an
        # IDLE batch must keep its forward_mode instead of being rewritten to
        # TARGET_VERIFY below (relevant once DP attention support lands).
        if not batch.forward_mode.is_decode():
            return

        bs = len(batch.reqs)

        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # generate positions and some indices using tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrieve_index,  # mutable
            retrieve_next_token,  # mutable
            retrieve_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK and not _is_cpu:
            tree_mask = []
            mask = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
            # TODO(siyuan): the for loop here leads to significant overhead in large batch size. Can be written into a kernel.
            for i in range(bs):
                seq_len = batch.seq_lens_cpu[i]
                req_mask = torch.ones(
                    (self.draft_token_num, seq_len), device=self.device
                )
                req_mask = torch.cat(
                    (
                        req_mask,
                        torch.from_numpy(mask[i]).to(
                            device=self.device, non_blocking=True
                        ),
                    ),
                    dim=1,
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + self.draft_token_num,
            batch_size=bs,
            draft_token_num=self.draft_token_num,
            device=self.device,
        )

        prepare_mamba_track_for_verify(batch)

        batch.spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=self.draft_token_num,
        )

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        batch_tokens = []
        i, stride = 0, self.draft_token_num
        # Same splice condition as _prepare_draft_tokens: only overlap mode
        # has accepted tokens missing from req.output_ids.
        use_prev_tokens = self.enable_overlap and not batch.has_grammar
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            put_ids = self._efficient_concat_last_n(
                list(req.origin_input_ids),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(put_ids)
            i += 1
        self.ngram_corpus.batch_put(batch_tokens)

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        verify_input: NgramVerifyInput = batch.spec_info
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        if batch.forward_mode.is_target_verify():
            # Prepare grammar data on CPU if needed
            if batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrieve_next_token.shape
                ).cpu()

            batch_result = self.target_worker.forward_batch_generation(
                batch, is_verify=True
            )

            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                # Generate the logit mask for structured output.
                # Overlap the CPU operations for bitmask generation with the forward pass.
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
                    # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                    # and will be applied to produce wrong results
                    batch.sampling_info.vocab_mask = None

            # Sample
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, "verify: target model logits"
            )
            (
                predict,
                accept_lens,
                accept_index,
            ) = eagle_sample(verify_input, batch, logits_output, vocab_mask)
            new_seq_lens = batch.seq_lens + accept_lens
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.draft_token_num,
            )
            accept_tokens = predict[accept_index].flatten()
            next_token_ids = accept_tokens

            # The KV mover expects drafts-only counts. NGRAM's
            # accept_lens includes the bonus token, matching scheduler output.
            num_correct_drafts_per_req = accept_lens - 1
            move_accept_tokens_to_target_kvcache(
                batch,
                accept_index,
                num_correct_drafts_per_req,
                self.token_to_kv_pool_allocator,
            )
            if batch.return_logprob:
                # The last arg is the accept_index row width minus 1. NGRAM's
                # accept_index is (bs, draft_token_num) -- the tree depth is not
                # bounded by spec_steps like EAGLE's (bs, spec_steps + 1).
                compute_spec_v2_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index,
                    self.draft_token_num - 1,
                )

            if on_publish is not None:
                on_publish(new_seq_lens)

            self._update_ngram_corpus(batch)
            # Erase match state of requests that left the decode batch.
            # req.finished() is unusable here: under overlap it flips at result
            # processing, one iteration after the request left the batch.
            # The last batch's entries persist while idle (bounded, small).
            cur_rids = {req.rid for req in batch.reqs}
            departed_rids = self._prev_decode_rids - cur_rids
            if departed_rids:
                self.ngram_corpus.erase_match_state(list(departed_rids))
            self._prev_decode_rids = cur_rids
            batch.forward_mode = ForwardMode.DECODE

        else:
            batch_result = self.target_worker.forward_batch_generation(batch)
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            new_seq_lens = batch.seq_lens.clone()

            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict

            if on_publish is not None:
                on_publish(new_seq_lens)

        # Construct the next draft input
        next_draft_input = NgramVerifyInput(
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            # Consumed by the non-overlap V2 scheduler branch to advance
            # batch.seq_lens after the isolation restore; overlap mode relays
            # it via on_publish instead.
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )
