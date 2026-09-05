from __future__ import annotations

import logging
import os
import time

import torch

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
        self._stats_copy_logits_ms = 0.0
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
        self._stats_copy_logits_ms = 0.0
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
        avg_accept_len = float(self._stats_accept_sum.item()) / requests
        logger.info(
            "[RACER_STATS] rounds=%d reqs=%d K=%d avg_accept_len=%.3f "
            "avg_borders=%.2f avg_retrieval_selected=%.2f "
            "avg_tokenbin_budget=%.2f avg_tokenbin_paths=%.2f "
            "avg_nodes_before_padding=%.2f padding_rounds=%d/%d(%.1f%%) "
            "avg_padding_nodes=%.2f proposal_ms/req=%.3f copy_logits_ms/round=%.3f",
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
            self._stats_copy_logits_ms / max(1, self._stats_rounds),
        )
        self._reset_racer_stats()

    def _update_copy_logits(self, batch, logits_output) -> float:
        """Refresh copy-logit rows from every verified tree node.

        Paper correspondence: Sec. 3.1 defines copy-logit reuse, while Appendix
        F's "w/ Rejected Logits" setting keeps logits from rejected candidates
        as well as accepted ones. TARGET_VERIFY returns logits for all ``B * C``
        draft nodes, so this update intentionally consumes the complete tree.

        Softmax is not needed here because TopK(softmax(z)) == TopK(z).
        """

        start = time.perf_counter() if self.racer_stats_enabled else 0.0
        bs = len(batch.reqs)
        k = self.draft_token_num
        logits = logits_output.next_token_logits[: bs * k]
        topk = min(self.racer_topk, logits.shape[-1])
        topk_ids = torch.topk(logits, k=topk, dim=-1).indices
        draft_tokens = batch.spec_info.draft_token[: bs * k]
        self.ngram_corpus.update_logits(
            [req.rid for req in batch.reqs],
            draft_tokens.detach().cpu().numpy(),
            topk_ids.detach().cpu().numpy(),
        )
        if self.racer_stats_enabled:
            return (time.perf_counter() - start) * 1000.0
        return 0.0

    @staticmethod
    def _gather_prompt_logits(logits_processor, local_logits: torch.Tensor) -> torch.Tensor:
        """Gather vocabulary-sharded prompt logits on every TP rank.

        ``_compute_lm_head`` returns only the local vocabulary shard when the LM
        head is tensor-parallel. RACER needs global token ids for its TokenBin,
        so prompt warm-up must mirror SGLang's normal logits gather before
        applying top-k. We deliberately use the full-logits gather here as the
        correctness-first implementation; prompt chunks keep the temporary
        [Tc, V] tensor bounded.

        For standard TP this is the same ``_logits_gatherer`` used by
        ``LogitsProcessor._get_logits``. DP-attention LM-head configurations use
        their dedicated attention-TP gather path. Every TP rank performs the
        gather so every RACER request-local TokenBin remains identical.
        """

        if not logits_processor.do_tensor_parallel_all_gather:
            return local_logits
        if logits_processor.use_attn_tp_group:
            return logits_processor._gather_attn_tp_logits(local_logits)
        return logits_processor._logits_gatherer(local_logits)

    def _warm_prompt_tokenbin(self, batch, hidden_states: torch.Tensor) -> None:
        """Seed RACER's copy-logit adjacency from prompt positions.

        Paper correspondence: Sec. 3.1 reuses the latest next-token logit
        distribution associated with a repeated vocabulary token. Prompt-time
        logits therefore provide valid initial rows for the same top-k adjacency
        used by Logits Tree expansion.

        SGLang adaptation: EXTEND already computed the prompt hidden states, so
        this path applies only the LM head in small chunks instead of rerunning
        the transformer. For a chunk of ``Tc`` prompt positions under TP:

            hidden_states : [Tc, H]
            local logits  : [Tc, V / TP]
            global logits : [Tc, V]
            top-k ids     : [Tc, racer_topk]

        The vocabulary gather happens before top-k, so the resulting ids are
        global vocabulary ids and are identical across TP ranks. Softmax is
        omitted because it preserves the logits ranking.
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
        topk = min(self.racer_topk, self.model_runner.model_config.vocab_size)

        warmup_start = time.perf_counter() if self.racer_stats_enabled else 0.0
        topk_chunks: list[torch.Tensor] = []
        num_chunks = 0
        for start in range(0, total, self.racer_prompt_warmup_chunk):
            end = min(total, start + self.racer_prompt_warmup_chunk)
            local_logits = logits_processor._compute_lm_head(
                hidden_states[start:end], lm_head
            )
            logits = self._gather_prompt_logits(logits_processor, local_logits)
            ids = torch.topk(logits, k=topk, dim=-1).indices
            topk_chunks.append(ids.cpu())
            num_chunks += 1
            del local_logits, logits

        topk_ids = torch.cat(topk_chunks, dim=0)
        self.ngram_corpus.seed_prompt_logits(
            [req.rid for req in batch.reqs],
            prompt_tokens.detach().cpu().numpy(),
            topk_ids.numpy(),
            prompt_lens,
        )

        if self.racer_stats_enabled:
            elapsed_ms = (time.perf_counter() - warmup_start) * 1000.0
            logger.info(
                "[RACER_PROMPT_WARMUP] tp=%d rows=%d chunks=%d topk=%d ms=%.3f",
                self.model_runner.server_args.tp_size,
                total,
                num_chunks,
                topk,
                elapsed_ms,
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
            self._stats_copy_logits_ms += self._update_copy_logits(batch, logits_output)

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