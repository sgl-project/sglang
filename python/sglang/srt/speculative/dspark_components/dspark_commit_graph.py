from types import SimpleNamespace

import torch
from sglang.kernels.ops.grammar.bitmask_ops import apply_token_bitmask_inplace_triton
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptSampling,
    SelectMixedAccept,
    SoftmaxTemp,
    accept_greedy_triton,
    finalize_accept_lens_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildOutTokens,
)
from sglang.srt.constrained.grammar_graph import GrammarHostCallback
from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.srt.sampling.verify_graph import VerifySamplingBuffers
from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite


class DSparkStaticVerifyEpilogue(DsparkVerifyEpilogue):
    def __init__(
        self, *, vocab_size: int, target_worker=None, commit_mamba=None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.target_worker = target_worker
        self.commit_mamba = commit_mamba
        self.locations = torch.zeros(
            (self.max_bs, self.stride),
            dtype=torch.int64,
            device=self.verify_lens_buf.device,
        )
        self.positions = torch.zeros_like(self.locations.reshape(-1))
        self.sampling = False
        self.vocab_size = vocab_size
        self.sampling_top_k = vocab_size
        device = self.verify_lens_buf.device
        self.sampling_buffers = VerifySamplingBuffers(
            self.max_bs, self.stride, vocab_size, device, with_draft=True
        )
        self.host_grammar = None

    def enable_grammar_host_callback(self, target_model):
        if target_model is None:
            raise ValueError("Grammar host callbacks require a target model")
        self.host_grammar = GrammarHostCallback(
            self.sampling_buffers.vocab_mask, self.stride
        )
        self.host_grammar.bind(target_model)

    def stage_sampling(
        self, *, bs, sampling_info, draft_block, grammar_mask, max_top_k=None
    ) -> None:
        # Only fold-eligible requests reach sampling staging.
        self.inject_gate_buf.fill_(1)
        buffers = self.sampling_buffers
        self.sampling = sampling_info is not None and not sampling_info.is_all_greedy
        buffers.vocab_mask.fill_(-1)
        if grammar_mask is not None:
            buffers.vocab_mask[: bs * self.stride].copy_(grammar_mask.vocab_mask)
        if not self.sampling:
            return
        # Keep the common top-k=50 path sparse without reading a GPU scalar.
        self.sampling_top_k = (
            min(64, self.vocab_size)
            if max_top_k is not None and 0 < max_top_k <= 64
            else self.vocab_size
        )
        buffers.stage(
            sampling_info,
            bs,
            temperatures=draft_block.temperatures,
            greedy_mask=draft_block.greedy_mask,
        )
        buffers.stage_draft(
            draft_block.corrected_logits.reshape(bs, self.gamma, self.vocab_size), bs
        )

    def _draft_probs(self, bs):
        buffers = self.sampling_buffers
        return SoftmaxTemp.execute(
            logits=buffers.draft_distribution[:bs].reshape(bs * self.gamma, -1),
            temperatures=buffers.temperatures[:bs],
            rows_per_request=self.gamma,
        ).view(bs, self.gamma, -1)

    def begin_step(self, verify_lens, armed: bool) -> None:
        if not armed:
            self.sampling = False
            self.sampling_buffers.vocab_mask.fill_(-1)
        super().begin_step(verify_lens, armed)

    def _accept(self, input_ids, seq_lens, verify_lens, bs: int) -> torch.Tensor:
        apply_token_bitmask_inplace_triton(
            self.strided_logits[: bs * self.stride],
            self.sampling_buffers.vocab_mask[: bs * self.stride],
        )
        candidates = input_ids.view(bs, self.stride)
        correct_len, bonus, cap_trim_lens = accept_greedy_triton(
            candidates=candidates,
            target_logits=self.strided_logits[: bs * self.stride],
            verify_num_draft_tokens=self.stride,
            cutoff_verify_lens=verify_lens,
        )
        if self.sampling:
            buffers = self.sampling_buffers
            sampling_info = buffers.for_batch(bs, is_all_greedy=False)
            draft_probs = self._draft_probs(bs)
            sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
                candidates=candidates,
                target_logits=self.strided_logits[: bs * self.stride],
                draft_probs=draft_probs,
                sampling_info=sampling_info,
                draft_input=SimpleNamespace(
                    max_top_k=self.sampling_top_k, uniform_top_k_value=None
                ),
                gamma=self.gamma,
                verify_num_draft_tokens=self.stride,
                cutoff_verify_lens=verify_lens,
            )
            selected = SelectMixedAccept.execute(
                greedy_mask=buffers.greedy_mask[:bs],
                greedy_len=correct_len,
                greedy_bonus=bonus,
                greedy_trim=cap_trim_lens,
                sampling_len=sampling_len,
                sampling_bonus=sampling_bonus,
                sampling_trim=sampling_trim,
            )
            correct_len, bonus, cap_trim_lens = (
                selected.correct_len,
                selected.bonus,
                selected.cap_trim_lens,
            )
        site = (
            SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE
            if self.sampling
            else SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
        )
        self._tp_sync.sync(site, correct_len)
        self._tp_sync.sync(site, bonus)
        self._tp_sync.sync(site, cap_trim_lens)
        finalized = finalize_accept_lens_triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=seq_lens[:bs],
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=self.draft_tokens_buf[: bs * self.gamma].view(bs, self.gamma),
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.stride,
            gamma=self.gamma,
        )
        self.correct_len_buf[:bs].copy_(correct_len)
        self.bonus_buf[:bs].copy_(bonus)
        self.cap_trim_lens_buf[:bs].copy_(cap_trim_lens.to(torch.int32))
        self.commit_lens_buf[:bs].copy_(finalized.commit_lens)
        self.new_seq_lens_buf[:bs].copy_(finalized.new_seq_lens)
        self.out_tokens_buf[:bs].copy_(out_tokens.view(bs, self.stride))
        return finalized.commit_lens

    @property
    def folds_commit(self) -> bool:
        return self.commit_ctx is not None and type(self.commit_ctx.resolve_pool()) in (
            MHATokenToKVPool,
        )

    @property
    def folds_mamba_commit(self):
        if self.target_worker is None or self.commit_mamba is None:
            return False
        pool = getattr(
            self.target_worker.model_runner.req_to_token_pool, "mamba_pool", None
        )
        # Unified pools need virtual snapshot-index translation.
        return (
            type(pool) is MambaPool
            and pool.replayssm_spec_fold
            and pool.replayssm_is_kda
        )

    def _commit_mamba(self, forward_batch, commit_lens):
        if self.folds_mamba_commit:
            # Reuse verify metadata; shared request mappings can change after its WAR fence.
            # Graph-runner batches carry no tree_cache; the native commit helper
            # reads only the tracking page from it, so source the static tree
            # page from the runner (get_schedule().page_size).
            self.commit_mamba(
                batch=SimpleNamespace(
                    mamba_track_indices=forward_batch.mamba_track_indices,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    tree_cache=SimpleNamespace(
                        page_size=self.target_worker.model_runner.page_size
                    ),
                ),
                seq_lens_pre_verify=forward_batch.seq_lens,
                seq_lens_post_verify=forward_batch.seq_lens + commit_lens,
                commit_lens=commit_lens,
            )

    def prepare(self, window, bs: int) -> None:
        self.begin_step(None, armed=False)
        if bs > self.max_bs:
            return
        self.verify_lens_buf[:bs].fill_(self.stride)
        if not self.folds_commit:
            return
        self.locations[:bs].copy_(window.verify_cache_loc_2d)
        self.positions[: bs * self.stride].copy_(window.positions_2d.reshape(-1))

    @torch.inference_mode()
    def capture_hook(self, runner, out, forward_batch, num_tokens) -> None:
        if runner.model_runner.is_draft_worker or runner.ragged_verify_mode:
            return
        bs = forward_batch.batch_size
        self.strided_logits = out.next_token_logits
        verify_lens = self.verify_lens_buf[:bs]
        # Greedy bonus gathering needs one row even for inactive graph-padding slots.
        commit_lens = self._accept(
            forward_batch.input_ids,
            forward_batch.seq_lens,
            verify_lens.clamp_min(1),
            bs,
        )
        # All verify slots are reserved. Accepted seq_lens controls visibility;
        # the next draft overwrites rejected slots before attention reads them.
        if self.folds_commit:
            self._commit_hidden(out.hidden_states, bs, verify_lens.to(torch.int32))
        # Fallback replays and graph-padding slots must not advance recurrent state.
        self._commit_mamba(
            forward_batch,
            torch.minimum(commit_lens, verify_lens.to(torch.int32))
            * self.inject_gate_buf,
        )

    def _commit_hidden(self, hidden, bs, commit_lens):
        pool = self.commit_ctx.resolve_pool()
        width = hidden.shape[0] // bs
        locations = self.locations[:bs, :width]
        self.commit_ctx.draft_model.write_target_hidden_kv(
            target_hidden=hidden,
            pool=pool,
            positions=self.positions.view(self.max_bs, self.stride)[
                :bs, :width
            ].reshape(-1),
            cache_loc=locations.reshape(-1),
            cache_loc_2d=locations,
            commit_lens=commit_lens,
        )


class DSparkCompactVerifyEpilogue(DSparkStaticVerifyEpilogue):
    @torch.inference_mode()
    def capture_hook(self, runner, out, forward_batch, num_tokens):
        if runner.model_runner.is_draft_worker or not runner.ragged_verify_mode:
            return
        bs = forward_batch.batch_size
        self.strided_logits = self._ensure_out(
            self.strided_logits, out.next_token_logits
        )
        self.strided_hidden = self._ensure_out(self.strided_hidden, out.hidden_states)
        verify_lens = self.verify_lens_buf[:bs]
        self._scatter(out.next_token_logits, out.hidden_states, verify_lens, bs)
        commit_lens = self._accept(
            forward_batch.input_ids, forward_batch.seq_lens, verify_lens, bs
        )
        # Unsupported sampling replays target graphs but commits after eager acceptance.
        commit_lens = (
            torch.minimum(commit_lens, verify_lens.to(torch.int32))
            * self.inject_gate_buf
        )
        if self.folds_commit:
            self._commit_compact_hidden(out.hidden_states, verify_lens, bs, commit_lens)
        self._commit_mamba(forward_batch, commit_lens)

    def _accept(self, input_ids, seq_lens, verify_lens, bs):
        # Keep the eager path's full proposal; verify_lens bounds accepted tokens.
        if bs == 1:
            anchors = input_ids[:1]
        else:
            starts = verify_lens.cumsum(0) - verify_lens
            anchors = input_ids[starts.clamp_max(input_ids.numel() - 1)]
        candidates = torch.cat(
            (
                anchors[:, None],
                self.draft_tokens_buf[: bs * self.gamma].view(bs, self.gamma),
            ),
            dim=1,
        )
        return super()._accept(candidates, seq_lens, verify_lens.clamp_min(1), bs)

    def _commit_compact_hidden(self, hidden, verify_lens, bs, commit_lens):
        if bs == 1:
            self._commit_hidden(hidden, bs, commit_lens)
            return
        ends = verify_lens.cumsum(0)
        tokens = torch.arange(hidden.shape[0], device=hidden.device)
        rows = torch.searchsorted(ends, tokens, right=True).clamp_max(bs - 1)
        cols = tokens - (ends - verify_lens)[rows]
        indices = (rows * self.stride + cols).clamp_max(bs * self.stride - 1)
        valid = (cols < commit_lens[rows]) & (cols < verify_lens[rows])
        pool = self.commit_ctx.resolve_pool()
        locations = self.locations.reshape(-1)[indices]
        self.commit_ctx.draft_model.write_target_hidden_kv(
            target_hidden=hidden,
            pool=pool,
            positions=self.positions[indices],
            cache_loc=locations,
            cache_loc_2d=locations[:, None],
            commit_lens=valid.to(torch.int32),
        )
