import itertools

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_disagg
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.utils.device import async_h2d


class Inkling2MTPVerifyExecutor(TargetVerifyExecutor):
    def __init__(self, *, draft_model, **kwargs):
        super().__init__(**kwargs)
        self.draft_model = draft_model

    def commit_hidden(
        self,
        *,
        batch,
        layout,
        hidden_strided,
        verify_window,
        logits_output,
        commit_lens,
        bs,
        run_compact,
    ):
        hidden = hidden_strided if run_compact else logits_output.hidden_states
        super().commit_hidden(
            batch=batch,
            layout=layout,
            hidden_strided=hidden_strided,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=commit_lens,
            bs=bs,
            run_compact=run_compact,
        )
        assert hidden is not None
        if bs:
            self.draft_model.store_verify_commit(
                req_pool_indices=batch.req_pool_indices,
                verify_hidden=hidden,
                commit_lens=commit_lens,
                stride=self.verify_num_draft_tokens,
            )


class Inkling2MTPWorker(DSparkWorkerV2):
    def __init__(self, *args, **kwargs):
        if get_disagg().disaggregation_mode != "null":
            raise ValueError("Inkling2 MTP does not support disaggregated serving")
        super().__init__(*args, **kwargs)
        self.draft_model.set_block_size(self.query_token_num)

    def _resolve_target_embed_tokens(self, target_model):
        return target_model.get_draft_input_embeddings()

    def _build_verify_executor(self, **kwargs):
        return Inkling2MTPVerifyExecutor(draft_model=self.draft_model, **kwargs)

    def init_attention_backends(self):
        super().init_attention_backends()
        draft_runner = self.draft_model_runner
        if (
            draft_runner.prefill_attention_backend_str != "fa4"
            or draft_runner.decode_attention_backend_str != "fa4"
        ):
            raise ValueError("Inkling2 MTP requires FA4 draft attention")
        draft_pool = draft_runner.token_to_kv_pool
        if isinstance(draft_pool, SWAKVPool):
            draft_window = draft_runner.sliding_window_size + 1
            target_window = self.target_worker.sliding_window_size
            if target_window is None or draft_window > target_window:
                raise ValueError("MTP sliding window must not exceed the target window")
        req_pool = self.model_runner.req_to_token_pool
        self.draft_model.ensure_propose_staging(req_pool.req_to_token.shape[0])

    def _on_prefill_target_hidden(self, batch, target_hidden):
        assert target_hidden is not None
        assert batch.extend_lens and all(n > 0 for n in batch.extend_lens)
        last_row_ids = async_h2d(
            list(itertools.accumulate(batch.extend_lens)),
            dtype=torch.int64,
            device=target_hidden.device,
        )
        last_rows = target_hidden.index_select(0, last_row_ids - 1)
        self.draft_model.store_f_rnn(batch.req_pool_indices, last_rows)
