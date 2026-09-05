from __future__ import annotations

import logging
import os
import time

import torch

from sglang.srt.distributed import get_attn_tp_group, get_tp_group
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker, _derive_tree_links
from sglang.srt.speculative.racer.draft_provider import RacerDraftProvider
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    record_stream_for_v2_verify,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan

logger = logging.getLogger(__name__)


class RACERWorker(NGRAMWorker):
    """RACER on top of SGLang's irregular-tree TARGET_VERIFY path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.racer_topk = int(os.getenv("SGLANG_RACER_TOPK", "9"))
        self.racer_ngram = int(os.getenv("SGLANG_RACER_NGRAM", "10"))
        self.racer_max_nodes = int(os.getenv("SGLANG_RACER_MAX_NODES", "10000"))
        self.max_trie_depth = int(os.getenv("SGLANG_RACER_HISTORY_WINDOW", "4096"))
        self.racer_prompt_warmup = os.getenv(
            "SGLANG_RACER_PROMPT_WARMUP", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.racer_prompt_warmup_chunk = max(
            1, int(os.getenv("SGLANG_RACER_PROMPT_WARMUP_CHUNK", "64"))
        )

        self.racer_stats_enabled = os.getenv("SGLANG_RACER_STATS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.racer_stats_interval = max(
            1, int(os.getenv("SGLANG_RACER_STATS_INTERVAL", "100"))
        )

        self.ngram_corpus = RacerDraftProvider(
            draft_token_num=self.draft_token_num,
            ngram=self.racer_ngram,
            topk=self.racer_topk,
            max_nodes=self.racer_max_nodes,
            stats_enabled=self.racer_stats_enabled,
        )

        self._stats_rounds = 0
        self._stats_requests = 0
        self._stats_borders = 0
        self._stats_retrieval_selected = 0
        self._stats_tokenbin_budget = 0
        self._stats_tokenbin_paths = 0
        self._stats_nodes_before_padding = 0
        self._stats_padding_nodes = 0
        self._stats_padding_rounds = 0
        self._stats_proposal_ms = 0.0
        self._stats_copy_topk_ms = 0.0
        self._stats_copy_d2h_ms = 0.0
        self._stats_copy_update_ms = 0.0
        self._stats_accept_sum = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )

    def _reset_racer_stats(self) -> None:
        self._stats_rounds = 0
        self._stats_requests = 0
        self._stats_borders = 0
        self._stats_retrieval_selected = 0
        self._stats_tokenbin_budget = 0
        self._stats_tokenbin_paths = 0
        self._stats_nodes_before_padding = 0
        self._stats_padding_nodes = 0
        self._stats_padding_rounds = 0
        self._stats_proposal_ms = 0.0
        self._stats_copy_topk_ms = 0.0
        self._stats_copy_d2h_ms = 0.0
        self._stats_copy_update_ms = 0.0
        self._stats_accept_sum.zero_()

    def _collect_proposal_stats(self) -> None:
        if not self.racer_stats_enabled:
            return
        for stats in self.ngram_corpus.consume_last_batch_stats():
            self._stats_requests += 1
            self._stats_borders += int(stats["borders"])
            self._stats_retrieval_selected += int(stats["retrieval_selected"])
            self._stats_tokenbin_budget += int(stats["tokenbin_budget"])
            self._stats_tokenbin_paths += int(stats["tokenbin_paths"])
            self._stats_nodes_before_padding += int(stats["nodes_before_padding"])
            padding_nodes = int(stats["padding_nodes"])
            self._stats_padding_nodes += padding_nodes
            self._stats_padding_rounds += int(padding_nodes > 0)
            self._stats_proposal_ms += float(stats["proposal_ms"])

    def _maybe_report_racer_stats(self) -> None:
        if (
            not self.racer_stats_enabled
            or self._stats_rounds < self.racer_stats_interval
        ):
            return

        requests = max(1, self._stats_requests)
        rounds = max(1, self._stats_rounds)
        avg_accept_len = float(self._stats_accept_sum.item()) / requests
        copy_topk_ms = self._stats_copy_topk_ms / rounds
        copy_d2h_ms = self._stats_copy_d2h_ms / rounds
        copy_update_ms = self._stats_copy_update_ms / rounds
        logger.info(
            "[RACER_STATS] rounds=%d reqs=%d K=%d avg_accept_len=%.3f "
            "avg_borders=%.2f avg_retrieval_selected=%.2f "
            "avg_tokenbin_budget=%.2f avg_tokenbin_paths=%.2f "
            "avg_nodes_before_padding=%.2f padding_rounds=%d/%d(%.1f%%) "
            "avg_padding_nodes=%.2f proposal_ms/req=%.3f "
            "copy_topk_gpu_ms/round=%.3f copy_d2h_ms/round=%.3f "
            "copy_update_cpu_ms/round=%.3f copy_total_ms/round=%.3f",
            self._stats_rounds,
            self._stats_requests,
            self.draft_token_num,
            avg_accept_len,
            self._stats_borders / requests,
            self._stats_retrieval_selected / requests,
            self._stats_tokenbin_budget / requests,
            self._stats_tokenbin_paths / requests,
            self._stats_nodes_before_padding / requests,
            self._stats_padding_rounds,
            requests,
            100.0 * self._stats_padding_rounds / requests,
            self._stats_padding_nodes / requests,
            self._stats_proposal_ms / requests,
            copy_topk_ms,
            copy_d2h_ms,
            copy_update_ms,
            copy_topk_ms + copy_d2h_ms + copy_update_ms,
        )
        self._reset_racer_stats()

    def _update_copy_logits(self, batch, logits_output) -> tuple[float, float, float]:
        """Refresh copy-logit rows from every verified tree node.

        Paper correspondence: Sec. 3.1 defines copy-logit reuse, while Appendix
        F's "w/ Rejected Logits" setting keeps logits from rejected candidates
        as well as accepted ones. TARGET_VERIFY returns logits for all ``B * C``
        draft nodes, so this update intentionally consumes the complete tree.

        Softmax is not needed here because TopK(softmax(z)) == TopK(z).

        Timing note: the old wall-clock timer started immediately after the
        target forward returned, so the subsequent ``.cpu()`` could charge
        outstanding TARGET_VERIFY kernels to RACER. CUDA events now isolate the
        top-k kernel. When stats are enabled we synchronize that end event before
        timing D2H and the CPU automaton update separately.
        """

        start_event = end_event = None
        if self.racer_stats_enabled:
            device_module = torch.get_device_module(self.device)
            start_event = device_module.Event(enable_timing=True)
            end_event = device_module.Event(enable_timing=True)
            start_event.record()

        bs = len(batch.reqs)
        k = self.draft_token_num
        logits = logits_output.next_token_logits[: bs * k]
        topk = min(self.racer_topk, logits.shape[-1])
        topk_ids = torch.topk(logits, k=topk, dim=-1).indices

        if end_event is not None:
            end_event.record()
            end_event.synchronize()
            topk_gpu_ms = start_event.elapsed_time(end_event)
            d2h_start = time.perf_counter()
        else:
            topk_gpu_ms = 0.0
            d2h_start = 0.0

        draft_tokens = batch.spec_info.draft_token[: bs * k]
        draft_tokens_cpu = draft_tokens.detach().cpu().numpy()
        topk_ids_cpu = topk_ids.detach().cpu().numpy()

        if self.racer_stats_enabled:
            d2h_ms = (time.perf_counter() - d2h_start) * 1000.0
            update_start = time.perf_counter()
        else:
            d2h_ms = 0.0
            update_start = 0.0

        self.ngram_corpus.update_logits(
            [req.rid for req in batch.reqs],
            draft_tokens_cpu,
            topk_ids_cpu,
        )

        if self.racer_stats_enabled:
            update_ms = (time.perf_counter() - update_start) * 1000.0
            return topk_gpu_ms, d2h_ms, update_ms
        return 0.0, 0.0, 0.0

    @staticmethod
    def _build_local_vocab_ids(lm_head, device: torch.device) -> torch.Tensor:
        """Map one LM-head shard's column indices to global token ids.

        VocabParallelEmbedding lays out each shard as base-vocab entries,
        base padding, added-vocab entries, then added-vocab padding. Padding
        columns must never enter a local top-k even though their zero-filled
        weights can otherwise beat negative real logits.
        """

        base_lm_head = getattr(lm_head, "base_layer", lm_head)
        shard = getattr(base_lm_head, "shard_indices", None)
        local_width = getattr(base_lm_head, "num_embeddings_per_partition", None)
        if shard is None or local_width is None:
            raise RuntimeError(
                "RACER TP prompt warm-up requires VocabParallelEmbedding shard metadata."
            )

        local_vocab_ids = torch.full(
            (int(local_width),), -1, dtype=torch.int32, device=device
        )

        num_org = int(shard.num_org_elements)
        if num_org:
            local_vocab_ids[:num_org] = torch.arange(
                int(shard.org_vocab_start_index),
                int(shard.org_vocab_end_index),
                dtype=torch.int32,
                device=device,
            )

        added_offset = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        if num_added:
            local_vocab_ids[added_offset : added_offset + num_added] = torch.arange(
                int(shard.added_vocab_start_index),
                int(shard.added_vocab_end_index),
                dtype=torch.int32,
                device=device,
            )

        return local_vocab_ids

    @staticmethod
    def _gather_prompt_topk_candidates(
        logits_processor,
        local_values: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-gather only each vocabulary shard's local top-k candidates."""

        group = (
            get_attn_tp_group()
            if logits_processor.use_attn_tp_group
            else get_tp_group()
        )
        gathered_values = group.all_gather(local_values, dim=-1)
        gathered_ids = group.all_gather(local_ids, dim=-1)
        return gathered_values, gathered_ids

    def _warm_prompt_tokenbin(self, batch, hidden_states: torch.Tensor) -> None:
        """Seed RACER's copy-logit adjacency from prompt positions.

        Paper correspondence: Sec. 3.1 reuses the latest next-token logit
        distribution associated with a repeated vocabulary token. Prompt-time
        logits therefore provide valid initial rows for the same top-k adjacency
        used by Logits Tree expansion.

        SGLang adaptation: EXTEND already computed the prompt hidden states, so
        this path applies only the LM head in small chunks instead of rerunning
        the transformer. Under TP, each rank first computes top-k on its own
        vocabulary shard, then all-gathers only ``(value, global_token_id)``
        candidates and performs a final top-k over ``TP * k`` candidates:

            hidden_states      : [Tc, H]
            local logits       : [Tc, V / TP]
            local candidates   : [Tc, k]
            gathered candidates: [Tc, TP * k]
            global top-k ids   : [Tc, k]

        Any global top-k token must be in its shard's local top-k, so this is
        exactly equivalent to gathering full logits before top-k (up to
        irrelevant tie ordering) while reducing communication from O(Tc * V)
        to O(Tc * TP * k). Padding columns are masked before local top-k.
        Softmax is omitted because it preserves the logits ranking.
        """

        if not self.racer_prompt_warmup:
            return
        if hidden_states is None or hidden_states.ndim != 2:
            logger.warning(
                "RACER prompt warm-up expected rank-2 hidden states, got %s; skipping.",
                None if hidden_states is None else tuple(hidden_states.shape),
            )
            return

        # ScheduleBatch stores the current EXTEND chunk lengths in extend_lens.
        # extend_seq_lens_cpu belongs to ForwardBatch and is not available here.
        prompt_lens = [int(x) for x in batch.extend_lens]
        total = sum(prompt_lens)
        if total <= 0:
            return
        if hidden_states.shape[0] != total:
            logger.warning(
                "RACER prompt warm-up hidden/token mismatch: hidden_rows=%d total_extend=%d; "
                "skipping this extend batch.",
                hidden_states.shape[0],
                total,
            )
            return

        prompt_tokens = batch.input_ids[:total]
        model = self.model_runner.model
        logits_processor = model.logits_processor
        lm_head = model.lm_head
        vocab_size = self.model_runner.model_config.vocab_size
        topk = min(self.racer_topk, vocab_size)

        base_lm_head = getattr(lm_head, "base_layer", lm_head)
        lm_head_tp_size = int(getattr(base_lm_head, "tp_size", 1))
        use_sharded_topk = (
            logits_processor.do_tensor_parallel_all_gather and lm_head_tp_size > 1
        )
        local_vocab_ids = (
            self._build_local_vocab_ids(lm_head, hidden_states.device)
            if use_sharded_topk
            else None
        )

        topk_chunks: list[torch.Tensor] = []
        timing_events = []
        num_chunks = 0
        device_module = (
            torch.get_device_module(self.device) if self.racer_stats_enabled else None
        )

        for start in range(0, total, self.racer_prompt_warmup_chunk):
            end = min(total, start + self.racer_prompt_warmup_chunk)

            if device_module is not None:
                events = [device_module.Event(enable_timing=True) for _ in range(5)]
                events[0].record()
            else:
                events = None

            local_logits = logits_processor._compute_lm_head(
                hidden_states[start:end], lm_head
            )
            if events is not None:
                events[1].record()

            if use_sharded_topk:
                if local_logits.shape[-1] != local_vocab_ids.numel():
                    raise RuntimeError(
                        "RACER TP prompt warm-up LM-head shard width mismatch: "
                        f"logits={local_logits.shape[-1]} "
                        f"mapping={local_vocab_ids.numel()}."
                    )

                # The LM-head loader zero-fills padding rows. Mask them in-place
                # so padding cannot enter a local top-k when real logits are negative.
                local_logits.masked_fill_(local_vocab_ids.lt(0).unsqueeze(0), -torch.inf)
                local_k = min(topk, local_logits.shape[-1])
                local_values, local_indices = torch.topk(
                    local_logits, k=local_k, dim=-1
                )
                local_ids = local_vocab_ids[local_indices]
                if events is not None:
                    events[2].record()

                candidate_values, candidate_ids = self._gather_prompt_topk_candidates(
                    logits_processor, local_values, local_ids
                )
                if events is not None:
                    events[3].record()

                _, global_indices = torch.topk(candidate_values, k=topk, dim=-1)
                ids = torch.gather(candidate_ids, -1, global_indices)
                if events is not None:
                    events[4].record()

                del (
                    local_values,
                    local_indices,
                    local_ids,
                    candidate_values,
                    candidate_ids,
                    global_indices,
                )
            else:
                # TP1 or a replicated LM head already has global-vocabulary columns.
                logits = local_logits[:, :vocab_size]
                ids = torch.topk(logits, k=topk, dim=-1).indices
                if events is not None:
                    events[2].record()
                    events[3].record()
                    events[4].record()

            topk_chunks.append(ids)
            if events is not None:
                timing_events.append(events)
            num_chunks += 1
            del local_logits

        topk_ids_gpu = torch.cat(topk_chunks, dim=0)

        if self.racer_stats_enabled:
            # The CPU copy below must wait for the same work anyway. Splitting
            # the wait here makes D2H timing uncontaminated by LM-head/top-k/NCCL.
            timing_events[-1][-1].synchronize()
            lm_head_ms = sum(e[0].elapsed_time(e[1]) for e in timing_events)
            local_topk_ms = sum(e[1].elapsed_time(e[2]) for e in timing_events)
            gather_ms = sum(e[2].elapsed_time(e[3]) for e in timing_events)
            merge_topk_ms = sum(e[3].elapsed_time(e[4]) for e in timing_events)
            gpu_ms = lm_head_ms + local_topk_ms + gather_ms + merge_topk_ms
            d2h_start = time.perf_counter()
        else:
            lm_head_ms = local_topk_ms = gather_ms = merge_topk_ms = gpu_ms = 0.0
            d2h_start = 0.0

        topk_ids = topk_ids_gpu.detach().cpu()
        prompt_tokens_cpu = prompt_tokens.detach().cpu().numpy()

        if self.racer_stats_enabled:
            d2h_ms = (time.perf_counter() - d2h_start) * 1000.0
            seed_start = time.perf_counter()
        else:
            d2h_ms = 0.0
            seed_start = 0.0

        self.ngram_corpus.seed_prompt_logits(
            [req.rid for req in batch.reqs],
            prompt_tokens_cpu,
            topk_ids.numpy(),
            prompt_lens,
        )

        if self.racer_stats_enabled:
            seed_cpu_ms = (time.perf_counter() - seed_start) * 1000.0
            logger.info(
                "[RACER_PROMPT_WARMUP] tp=%d rows=%d chunks=%d topk=%d "
                "lm_head_gpu_ms=%.3f local_topk_gpu_ms=%.3f gather_gpu_ms=%.3f "
                "merge_topk_gpu_ms=%.3f gpu_ms=%.3f d2h_ms=%.3f seed_cpu_ms=%.3f",
                self.model_runner.server_args.tp_size,
                total,
                num_chunks,
                topk,
                lm_head_ms,
                local_topk_ms,
                gather_ms,
                merge_topk_ms,
                gpu_ms,
                d2h_ms,
                seed_cpu_ms,
            )

    def forward_batch_generation(self, batch, on_publish=None) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
        self._collect_proposal_stats()

        verify_input: NgramVerifyInput = batch.spec_info
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        if batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(batch, is_verify=True)
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            copy_topk_ms, copy_d2h_ms, copy_update_ms = self._update_copy_logits(
                batch, logits_output
            )
            self._stats_copy_topk_ms += copy_topk_ms
            self._stats_copy_d2h_ms += copy_d2h_ms
            self._stats_copy_update_ms += copy_update_ms

            verify_input = batch.spec_info
            grammar_mask = None
            if batch.has_grammar:
                mask, req_drafts = self.grammar_tree_host
                retrieve_next_token_cpu, retrieve_next_sibling_cpu = _derive_tree_links(
                    mask, bs, self.draft_token_num
                )
                grammar_mask = build_grammar_vocab_mask(
                    reqs=batch.reqs,
                    tree=GrammarTree.from_host(
                        retrieve_next_token_cpu,
                        retrieve_next_sibling_cpu,
                        torch.from_numpy(req_drafts).to(torch.int64).view(bs, -1),
                    ),
                    sampling_info=batch.sampling_info,
                    device=verify_input.retrieve_next_token.device,
                    barrier=None,
                )

            maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
            maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
            predict, accept_lens, accept_index = eagle_sample(
                verify_input, batch, logits_output, grammar_mask
            )
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

            num_correct_drafts_per_req = accept_lens - 1
            move_accept_tokens_to_target_kvcache(
                batch,
                accept_index,
                num_correct_drafts_per_req,
                self.token_to_kv_pool_allocator,
            )
            if batch.return_logprob:
                compute_spec_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index=accept_index,
                )
            if on_publish is not None:
                on_publish(new_seq_lens)

            self._update_ngram_corpus(batch)
            cur_rids = {req.rid for req in batch.reqs}
            departed_rids = self._prev_decode_rids - cur_rids
            if departed_rids:
                self.ngram_corpus.erase_match_state(list(departed_rids))
            self._prev_decode_rids = cur_rids
            batch.forward_mode = ForwardMode.DECODE

            if self.racer_stats_enabled:
                self._stats_rounds += 1
                self._stats_accept_sum.add_(accept_lens.sum())
                self._maybe_report_racer_stats()
        else:
            capture_mode = (
                CaptureHiddenMode.FULL
                if self.racer_prompt_warmup and batch.forward_mode.is_extend()
                else None
            )
            batch_result = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=capture_mode
            )
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            if capture_mode is not None:
                self._warm_prompt_tokenbin(batch, logits_output.hidden_states)
            new_seq_lens = batch.seq_lens.clone()
            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict
            if on_publish is not None:
                on_publish(new_seq_lens)

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
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )
