import logging
from functools import partial
from types import SimpleNamespace
from typing import NamedTuple

import torch
from sglang.kernels.ops.grammar.bitmask_ops import apply_token_bitmask_inplace_triton
from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.constrained.grammar_graph import GrammarHostCallback
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.runtime_context import get_exec
from sglang.srt.sampling.verify_graph import VerifySamplingBuffers
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify


class EagleVerifyResult(NamedTuple):
    predict: torch.Tensor
    accept_lens: torch.Tensor
    accept_index: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor


class EagleVerifyEpilogue:
    def __init__(
        self, target_worker, width, max_bs=1, *, rejection_sampling=False, commit=None
    ):
        self.target_worker = target_worker
        self.commit = commit
        runner = target_worker.model_runner
        self.width = width
        self.max_bs = max_bs
        self.vocab_size = runner.model_config.vocab_size
        self.sampling = False
        self.sampling_top_k = self.vocab_size
        device = runner.device
        self.armed = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.info = VerifySamplingBuffers(
            max_bs,
            width,
            self.vocab_size,
            device,
            with_draft=rejection_sampling,
            with_logits_adjustments=True,
        )
        self.bonus_tokens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.new_seq_lens = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.predict = torch.zeros(max_bs * width, dtype=torch.int32, device=device)
        self.accept_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.accept_index = torch.full(
            (max_bs, width), -1, dtype=torch.int32, device=device
        )
        self.retrieve_index = torch.arange(max_bs * width, device=device).view(
            max_bs, width
        )
        self.retrieve_next_token = torch.arange(1, width + 1, device=device).repeat(
            max_bs, 1
        )
        self.retrieve_next_token[:, -1] = -1
        self.retrieve_next_sibling = torch.full_like(self.retrieve_index, -1)
        self.host_grammar = GrammarHostCallback(self.info.vocab_mask, width)
        self.host_grammar.bind(runner.model)

    def prepare(self, batch):
        self.armed.zero_()
        self.sampling = False
        info = batch.sampling_info
        bs = len(batch.reqs)
        if (
            batch.forward_mode.is_idle()
            or not 0 < bs <= self.max_bs
            or info.has_custom_logit_processor
            or getattr(info, "acc_linear_penalties", None) is not None
            or (
                batch.has_grammar
                and any(
                    req.grammar is not None
                    and not isinstance(req.grammar, XGrammarGrammar)
                    for req in batch.reqs
                )
            )
        ):
            return False
        self.sampling = not info.is_all_greedy
        small_top_k = all(0 < req.sampling_params.top_k <= 64 for req in batch.reqs)
        self.sampling_top_k = (
            min(64, self.vocab_size) if small_top_k else self.vocab_size
        )
        self.info.stage(info, bs)
        return True

    def arm(self, batch, barrier, draft_probs=None):
        bs = len(batch.reqs)
        if self.info.draft_distribution is not None and self.sampling:
            self.info.stage_draft(draft_probs, bs)
        self.armed[:bs].fill_(1)
        if batch.has_grammar:
            self.host_grammar.prepare([req.grammar for req in batch.reqs], barrier)
            batch.sampling_info.grammar_mask = None

    def read(self, batch):
        if batch.has_grammar:
            self.host_grammar.finish()
        # The overlap scheduler can retain a result past the next graph replay.
        bs = len(batch.reqs)
        return EagleVerifyResult(
            self.predict[: bs * self.width].clone(),
            self.accept_lens[:bs].clone(),
            self.accept_index[:bs].clone(),
            self.bonus_tokens[:bs].clone(),
            self.new_seq_lens[:bs].clone(),
        )

    def capture_hook(self, graph_runner, out, forward_batch, num_tokens):
        if graph_runner.model_runner.is_draft_worker or graph_runner.ragged_verify_mode:
            return
        bs = forward_batch.batch_size
        assert 0 < bs <= self.max_bs and num_tokens == bs * self.width
        info = self.info.for_batch(bs, is_all_greedy=not self.sampling)
        info.logits_adjustment_gate = self.armed[:bs].bool()
        verify_input = SimpleNamespace(
            draft_token=forward_batch.input_ids,
            draft_token_num=self.width,
            max_tree_depth=self.width,
            tree_topk=1,
            retrieve_index=self.retrieve_index[:bs],
            retrieve_next_token=self.retrieve_next_token[:bs],
            retrieve_next_sibling=self.retrieve_next_sibling[:bs],
            # Read only when the server flag routes eagle_sample to the chain
            # rejection sampler; None leaves the target-only branch untouched.
            draft_probs=(
                self.info.draft_distribution[:bs]
                if self.info.draft_distribution is not None
                else None
            ),
        )
        batch = SimpleNamespace(
            device=out.next_token_logits.device,
            forward_mode=forward_batch.forward_mode,
            seq_lens=forward_batch.seq_lens,
            sampling_info=info,
            req_pool_indices=forward_batch.req_pool_indices,
            mamba_track_indices=forward_batch.mamba_track_indices,
            # Graph-runner batches carry no tree_cache; the native mamba commit
            # reads only the tracking page from it, so source the static tree
            # page from the runner (get_schedule().page_size).
            tree_cache=SimpleNamespace(
                page_size=self.target_worker.model_runner.page_size
            ),
        )
        predict, lengths, indices = eagle_sample(
            verify_input,
            batch,
            out,
            SimpleNamespace(
                apply=lambda logits: apply_token_bitmask_inplace_triton(
                    logits, self.host_grammar.vocab_mask[:num_tokens]
                )
            ),
            target_max_top_k=self.sampling_top_k if self.sampling else None,
        )
        self.predict[:num_tokens].copy_(predict)
        self.accept_lens[:bs].copy_(lengths)
        self.accept_index[:bs].copy_(indices)
        fill_bonus_tokens_func(
            predict[indices], lengths, self.bonus_tokens[:bs], self.width, bs
        )
        self.new_seq_lens[:bs].copy_(forward_batch.seq_lens + lengths)
        # Disabled replays must neither advance recurrent state nor save snapshots.
        if self.commit is not None:
            self.commit(
                batch,
                lengths * self.armed[:bs],
                torch.where(self.armed[:bs, None].bool(), indices, -1),
                self.width,
            )


def install_eagle_verify_epilogue(target_worker, server_args):
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

    runner = target_worker.model_runner
    server_args = resolved_view(server_args)
    config = get_exec().graph.cuda_graph_config
    if (
        server_args.speculative_eagle_topk != 1
        or server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
        or server_args.speculative_adaptive
        or server_args.disable_cuda_graph
        or config is None
        or config.decode.backend != "full"
        or not config.decode.bs
        or server_args.enable_dp_attention
        or server_args.enable_two_batch_overlap
        or server_args.enable_pdmux
        or server_args.pp_size != 1
        or SIMULATE_ACC_LEN > 0
        or torch.device(runner.device).type != "cuda"
    ):
        return
    # Additional backend-specific rollback must be captured before admitting it.
    if (
        getattr(runner.token_to_kv_pool, "clear_unaccepted_c128_draft_states", None)
        is not None
    ):
        return
    commit = None
    if mambaish_config(runner.model_config) is not None:
        pool = runner.req_to_token_pool.mamba_pool
        # Virtual recurrent-state indices need translation outside the graph.
        if (
            not server_args.enable_linear_replayssm_spec
            or type(pool) is not MambaPool
            or not (pool.replayssm_spec_fold and pool.replayssm_is_kda)
        ):
            return
        commit = partial(commit_mamba_states_after_verify, target_worker)
    epilogue = EagleVerifyEpilogue(
        target_worker,
        server_args.speculative_num_draft_tokens,
        max(config.decode.bs),
        rejection_sampling=server_args.speculative_use_rejection_sampling,
        commit=commit,
    )
    runner.spec_verify_epilogue = epilogue
    runner.capture_tail_hooks.append(epilogue.capture_hook)
    logging.getLogger(__name__).info(
        "Enabled chain verify-graph sampling and finalization (max_bs=%d)",
        epilogue.max_bs,
    )
