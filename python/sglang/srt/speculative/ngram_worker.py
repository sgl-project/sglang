import logging
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs_func,
    move_accepted_tokens_to_target_kvcache,
)
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import generate_token_bitmask, maybe_detect_nan

logger = logging.getLogger(__name__)


USE_FULL_MASK = True


class NGRAMWorker:
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
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        # Set constant
        NgramVerifyInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self._init_preallocated_tensors()

        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
        )

        # Precomputed draft cache: maps (req_idx, node_j) -> (drafts, mask)
        # populated by precompute_draft_tokens, consumed by _prepare_for_speculative_decoding
        self._precomputed_cache = None
        self._precomputed_batch_reqs = None  # list of req objects for identity check
        self._precomputed_draft_tokens_np = (
            None  # the current batch's draft tokens (numpy)
        )
        self._precomputed_tree_mask_np = None  # the current batch's tree mask (numpy)

    def clear_cache_pool(self):
        self.ngram_corpus.reset()

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
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
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
        self.retrive_next_token_batch = []
        self.retrive_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _try_use_precomputed_drafts(
        self, batch: ModelWorkerBatch
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Try to use precomputed draft tokens from the previous batch's
        precompute_draft_tokens call. Returns (drafts, mask) if ALL requests
        hit, or None if any miss (with partial-hit regeneration possible).

        Match condition: since batch_get returns drafts[0] = last context token
        = draft[node_j], checking first_draft_token == bonus_token is equivalent
        to finding nodes where prev_draft[node_j] == bonus_token. This avoids
        iterating over all precomputed paths.
        """
        if self._precomputed_cache is None:
            return None

        bs = len(batch.reqs)
        d = self.draft_token_num

        prev_token_ids, prev_accept_lens = (
            batch.spec_info.verified_tokens,
            batch.spec_info.accept_lens,
        )
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        prev_token_ids_list = prev_token_ids.tolist()
        prev_accept_lens_list = prev_accept_lens.tolist()

        # Check that batch composition matches (same requests in same order)
        if (
            self._precomputed_batch_reqs is None
            or len(self._precomputed_batch_reqs) != bs
        ):
            self._precomputed_cache = None
            return None

        # Direct lookup: draft[node_j] == bonus_token, no need to iterate paths
        prev_draft_2d = self._precomputed_draft_tokens_np.reshape(bs, d)
        result_drafts = np.empty(bs * d, dtype=np.int64)
        result_masks = np.empty(bs * d * d, dtype=np.int64)
        hit_flags = [False] * bs
        hit_count = 0

        for i in range(bs):
            if batch.reqs[i] is not self._precomputed_batch_reqs[i]:
                self._precomputed_cache = None
                return None
            accept_len = prev_accept_lens_list[i]  # includes bonus token
            bonus_token = prev_token_ids_list[i * d + accept_len - 1]

            matching_nodes = np.where(prev_draft_2d[i] == bonus_token)[0]
            for node_j in matching_nodes:
                key = (i, int(node_j))
                if key in self._precomputed_cache:
                    drafts_for_path, mask_for_path = self._precomputed_cache[key]
                    result_drafts[i * d : (i + 1) * d] = drafts_for_path
                    result_masks[i * d * d : (i + 1) * d * d] = mask_for_path.flatten()
                    hit_flags[i] = True
                    hit_count += 1
                    break

        # Store prev token info (needed by precompute_draft_tokens of next batch)
        self.prev_token_ids = prev_token_ids_list
        self.prev_accept_lens = prev_accept_lens_list

        if hit_count == bs:
            logger.debug(f"Precomputed draft cache HIT for all {bs} requests")
            return result_drafts, result_masks
        elif hit_count == 0:
            logger.debug(
                f"Precomputed draft cache MISS for all {bs} requests, "
                "falling back to fresh generation"
            )
            return None
        else:
            # Partial hit: regenerate only the missed requests and merge
            logger.debug(
                f"Precomputed draft cache partial HIT: {hit_count}/{bs} requests"
            )
            return self._regenerate_missing_drafts(
                batch,
                result_drafts,
                result_masks,
                hit_flags,
                prev_token_ids_list,
                prev_accept_lens_list,
            )

    def _regenerate_missing_drafts(
        self,
        batch: ModelWorkerBatch,
        result_drafts: np.ndarray,
        result_masks: np.ndarray,
        hit_flags: list,
        prev_token_ids_list: list,
        prev_accept_lens_list: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Regenerate drafts only for requests that missed the precomputed cache,
        then merge with precomputed hits."""
        bs = len(batch.reqs)
        d = self.draft_token_num
        stride = d

        self.ngram_corpus.synchronize()

        # Collect lookup tokens only for missed requests
        miss_indices = []
        miss_tokens = []
        for i in range(bs):
            if not hit_flags[i]:
                # This request missed the precomputed cache
                miss_indices.append(i)
                if not batch.has_grammar:
                    prev_tokens = prev_token_ids_list[
                        i * stride : i * stride + prev_accept_lens_list[i]
                    ]
                else:
                    prev_tokens = []
                check_token = self._efficient_concat_last_n(
                    batch.reqs[i].origin_input_ids,
                    batch.reqs[i].output_ids + prev_tokens,
                    self.max_trie_depth,
                )
                miss_tokens.append(check_token)

        if not miss_tokens:
            return result_drafts, result_masks

        # Batch lookup only for missed requests
        miss_drafts, miss_masks = self.ngram_corpus.batch_get(miss_tokens)
        miss_drafts_2d = miss_drafts.reshape(len(miss_indices), d)
        miss_masks_3d = miss_masks.reshape(len(miss_indices), d, d)

        # Merge into results
        for j, i in enumerate(miss_indices):
            result_drafts[i * d : (i + 1) * d] = miss_drafts_2d[j]
            result_masks[i * d * d : (i + 1) * d * d] = miss_masks_3d[j].flatten()

        return result_drafts, result_masks

    def _prepare_draft_tokens(
        self, batch: ModelWorkerBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = len(batch.reqs)
        stride = self.draft_token_num

        prev_token_ids, prev_accept_lens = (
            batch.spec_info.verified_tokens,
            batch.spec_info.accept_lens,
        )
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        self.prev_token_ids = prev_token_ids.tolist()
        self.prev_accept_lens = prev_accept_lens.tolist()

        self.ngram_corpus.synchronize()
        batch_tokens = []
        assert len(batch.reqs) == len(self.prev_accept_lens)
        i = 0
        for req in batch.reqs:
            # grammar doesn't support overlap and output_ids will be complete.
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if not batch.has_grammar
                else []
            )
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids,
                req.output_ids + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(check_token)
            i += 1
        req_drafts, mask = self.ngram_corpus.batch_get(batch_tokens)
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_extend():
            self._precomputed_cache = None
            return

        bs = len(batch.reqs)

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        # Wait for the previous batch's verify to finish (we need verified_tokens)
        if batch.spec_info.verify_done is not None:
            batch.spec_info.verify_done.synchronize()

        # Try to use precomputed drafts from the previous precompute_draft_tokens call
        precomputed_result = self._try_use_precomputed_drafts(batch)
        if precomputed_result is not None:
            req_drafts, mask = precomputed_result
        else:
            req_drafts, mask = self._prepare_draft_tokens(batch)

        # Save for precompute_draft_tokens and _try_use_precomputed_drafts
        self._precomputed_draft_tokens_np = req_drafts.copy()
        self._precomputed_tree_mask_np = mask.copy()

        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # generate positions and some indices using tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrive_index,  # mutable
            retrive_next_token,  # mutable
            retrive_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
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

        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
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
        batch.spec_info = NgramVerifyInput(
            server_args=self.server_args,
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=self.draft_token_num,
        )

    def _update_ngram_corpus(self, batch: ModelWorkerBatch):
        batch_tokens = []
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_trie_depth
            )
            batch_tokens.append(put_ids)
        self.ngram_corpus.batch_put(batch_tokens)

    def precompute_draft_tokens(self, batch: ModelWorkerBatch):
        """Precompute draft tokens for the next batch by enumerating all possible
        verify outcomes of the current batch.

        Timing context (overlap scheduling):
        - At precompute time: req.output_ids includes up to batch_{K-2}'s tokens
        - At next batch's _prepare_draft_tokens: req.output_ids includes batch_{K-1}'s
        - So we include prev_tokens (batch_{K-1}'s verified tokens) in the context.

        The approach:
        1. From the current batch's tree mask, extract all valid paths (root to each node)
        2. For each (request, path), construct the ngram check context:
           context = last_n(output_ids + prev_tokens_from_current_batch + path_draft_tokens)
        3. Batch all checks into one batch_get call
        4. Store results indexed by (req_idx, node_j)
        5. At next _prepare_for_speculative_decoding, check if
           draft[node_j] == bonus_token via direct numpy lookup
        """
        if batch.forward_mode.is_extend():
            self._precomputed_cache = None
            return

        bs = len(batch.reqs)
        d = self.draft_token_num

        cur_draft_tokens = self._precomputed_draft_tokens_np  # shape (bs * d,)
        cur_tree_mask = self._precomputed_tree_mask_np  # shape (bs * d * d,)

        if cur_draft_tokens is None or cur_tree_mask is None:
            self._precomputed_cache = None
            return

        if not hasattr(self, "prev_token_ids") or self.prev_token_ids is None:
            self._precomputed_cache = None
            return

        cur_draft_tokens_2d = cur_draft_tokens.reshape(bs, d)
        cur_tree_mask_3d = cur_tree_mask.reshape(bs, d, d)

        self.ngram_corpus.synchronize()

        all_check_tokens = []
        req_path_info = []  # req_path_info[req_idx] = [(node_idx, check_idx), ...]

        stride = d
        for req_idx in range(bs):
            req = batch.reqs[req_idx]
            req_draft = cur_draft_tokens_2d[req_idx]  # shape (d,)
            req_mask = cur_tree_mask_3d[req_idx]  # shape (d, d)
            req_paths = []

            # batch_{K-1}'s verified tokens (will be added to output_ids by scheduler)
            if not batch.has_grammar:
                prev_tokens = self.prev_token_ids[
                    req_idx * stride : req_idx * stride + self.prev_accept_lens[req_idx]
                ]
            else:
                prev_tokens = []

            # Simulate what output_ids will look like at next batch's draft prep
            base_output = req.output_ids + prev_tokens

            for node_j in range(d):
                # Skip invalid/padding nodes
                if req_mask[node_j, node_j] == 0:
                    continue

                # Extract the path: columns where mask[node_j][c] == 1
                path_cols = np.nonzero(req_mask[node_j])[0]
                path_draft_tokens = req_draft[path_cols].tolist()

                if not batch.has_grammar:
                    check_token = self._efficient_concat_last_n(
                        req.origin_input_ids,
                        base_output + path_draft_tokens,
                        self.max_trie_depth,
                    )
                else:
                    check_token = self._efficient_concat_last_n(
                        req.origin_input_ids,
                        base_output,
                        self.max_trie_depth,
                    )

                check_idx = len(all_check_tokens)
                all_check_tokens.append(check_token)
                req_paths.append((node_j, check_idx))

            req_path_info.append(req_paths)

        if not all_check_tokens:
            self._precomputed_cache = None
            return

        # Single batched ngram lookup for all (request, path) combinations
        all_drafts, all_masks = self.ngram_corpus.batch_get(all_check_tokens)

        num_checks = len(all_check_tokens)
        all_drafts_2d = all_drafts.reshape(num_checks, d)
        all_masks_3d = all_masks.reshape(num_checks, d, d)

        # Build the precomputed cache: (req_idx, node_j) -> (drafts, mask)
        precomputed = {}
        for req_idx in range(bs):
            for node_j, check_idx in req_path_info[req_idx]:
                precomputed[(req_idx, node_j)] = (
                    all_drafts_2d[check_idx],
                    all_masks_3d[check_idx],
                )

        self._precomputed_cache = precomputed
        self._precomputed_batch_reqs = list(batch.reqs)

        logger.debug(
            f"Precomputed {len(precomputed)} draft combos from "
            f"{num_checks} contexts for {bs} requests"
        )

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(model_worker_batch.seq_lens)
        self._prepare_for_speculative_decoding(model_worker_batch)
        verify_input: NgramVerifyInput = model_worker_batch.spec_info
        accept_length = torch.tensor([1] * bs, dtype=torch.int32, device=self.device)

        if model_worker_batch.forward_mode.is_target_verify():
            # Prepare grammar data on CPU if needed
            if model_worker_batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrive_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrive_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrive_next_token.shape
                ).cpu()

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )

            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            # Generate vocab mask for constrained decoding
            vocab_mask = None
            if model_worker_batch.has_grammar:
                # Generate the logit mask for structured output.
                # Overlap the CPU operations for bitmask generation with the forward pass.
                vocab_mask = generate_token_bitmask(
                    model_worker_batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    model_worker_batch.sampling_info.vocab_size,
                )

                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrive_next_token.device)
                    # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                    # and will be applied to produce wrong results
                    model_worker_batch.sampling_info.vocab_mask = None

            # Sample
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            (
                predict,
                accept_length,
                accept_index,
            ) = verify_input.sample(model_worker_batch, logits_output, vocab_mask)
            new_seq_lens = model_worker_batch.seq_lens + accept_length
            verified_tokens = predict[accept_index].flatten()

            # copy kvcache will not use the new_seq_lens
            move_accepted_tokens_to_target_kvcache(
                model_worker_batch,
                accept_index,
                accept_length,
                self.token_to_kv_pool_allocator,
                self.draft_token_num,
            )
            # TODO logprobs for spec v2
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

            self._update_ngram_corpus(model_worker_batch)
            model_worker_batch.forward_mode = ForwardMode.DECODE

            # Precompute draft tokens for the NEXT batch while GPU is running verify
            self.precompute_draft_tokens(model_worker_batch)

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            new_seq_lens = model_worker_batch.seq_lens.clone()

            verified_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            verified_tokens[:, 0] = predict
            verified_tokens = verified_tokens.flatten()
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

        # Construct the next draft input
        next_draft_input = NgramVerifyInput(
            server_args=self.server_args,
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            verified_tokens=verified_tokens,
            accept_lens=accept_length,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_length,
            next_draft_input=next_draft_input,
        )
