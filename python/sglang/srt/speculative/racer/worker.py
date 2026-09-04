from __future__ import annotations

import os

import torch

from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


class RACERWorker(NGRAMWorker):
    """RACER on top of SGLang's irregular-tree TARGET_VERIFY path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.racer_topk = int(os.getenv("SGLANG_RACER_TOPK", "9"))
        self.racer_ngram = int(os.getenv("SGLANG_RACER_NGRAM", "10"))
        self.racer_max_nodes = int(os.getenv("SGLANG_RACER_MAX_NODES", "10000"))
        self.max_trie_depth = int(os.getenv("SGLANG_RACER_HISTORY_WINDOW", "4096"))
        self.ngram_corpus = RacerDraftProvider(
            draft_token_num=self.draft_token_num,
            ngram=self.racer_ngram,
            topk=self.racer_topk,
            max_nodes=self.racer_max_nodes,
        )

    def _update_copy_logits(self, batch, logits_output) -> None:
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

    def forward_batch_generation(self, batch, on_publish=None) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        verify_input: NgramVerifyInput = batch.spec_info
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        if batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(batch, is_verify=True)
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            self._update_copy_logits(batch, logits_output)

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
