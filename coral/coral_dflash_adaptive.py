"""Confidence-selected adaptive DFlash2 target verification (SGLang v0.5.21).

The draft still proposes the full block. The selector's path confidence picks one of
CORAL_ADAPTIVE_VERIFY_TIERS target-verify graphs (default 2,4,6,8). This keeps high-confidence
steps at the full block while low-confidence steps stop target verification early.

The policy is the calibrated cumulative-confidence rule from adaptive_verify_sim.py. It is
integer-exact for greedy decode: a shorter verify can only truncate positions the policy did not
expect to accept; the accepted prefix and bonus token still follow the target's argmax.

Requires greedy requests and a captured selector sampler. Sampled selector paths fall back to the
full block because the stock sampled accept kernel assumes one width.
"""
import os
import sys
from dataclasses import dataclass

import torch


def _log(message):
    print(f"[coral] {message} (pid {os.getpid()})", file=sys.stderr, flush=True)


@dataclass
class _TargetState:
    length: int
    attn_backend: object
    graph_runner: object


def _tiers():
    values = sorted({int(x) for x in os.environ.get("CORAL_ADAPTIVE_VERIFY_TIERS", "2,4,6,8").split(",") if x})
    if not values or values[-1] < 2:
        raise ValueError("CORAL_ADAPTIVE_VERIFY_TIERS must end with the full block")
    return values


def _threshold():
    value = float(os.environ.get("CORAL_ADAPTIVE_VERIFY_THRESHOLD", "0.5"))
    if not 0.0 < value <= 1.0:
        raise ValueError("CORAL_ADAPTIVE_VERIFY_THRESHOLD must be in (0, 1]")
    return value


def _tier_from_probabilities(probabilities, *, tiers, threshold):
    """Return the first tier that can verify the leading confident drafts.

    probabilities is [bs, gamma]: the selector's probability for the token it actually drafted at
    each position. d is the count whose cumulative product stays >= threshold; the verify width is
    d + 1 (anchor plus d drafts), rounded up to an available tier.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        return tiers[-1]
    safe = probabilities.clamp_min(0.0)
    cumulative = torch.cumprod(safe, dim=1)
    drafts = (cumulative >= threshold).sum(dim=1)
    widest = int(drafts.max().item())
    for tier in tiers:
        if tier >= widest + 1:
            return tier
    return tiers[-1]


def _wrap_sampler(mod):
    """Capture the selector's T=1 path softmax without changing drafted tokens."""
    sampler_cls = getattr(mod, "_SelectorDraftSampler", None)
    if sampler_cls is None or not hasattr(mod, "_selector_lattice"):
        return False

    original_lattice = mod._selector_lattice
    lattice_output = {}

    def selector_lattice(draft_model, pred_hidden, anchor_token_ids):
        ids, scores = original_lattice(draft_model, pred_hidden, anchor_token_ids)
        lattice_output["value"] = (ids, scores)
        return ids, scores

    original_init = sampler_cls.__init__
    original_call = sampler_cls.__call__

    def init(self, *args, **kwargs):
        max_bs = int(kwargs.get("max_bs", 0))
        if max_bs < 2:
            kwargs["max_bs"] = 2
        original_init(self, *args, **kwargs)
        self.q_soft = torch.zeros_like(self.q_out)

    def call(self, hidden_states, input_ids):
        result = original_call(self, hidden_states, input_ids)
        ids, scores = lattice_output.pop("value")
        batch_size, gamma, top_k = ids.shape
        tokens = self.out[: batch_size * gamma].view(batch_size, gamma)
        index = (ids == tokens[..., None]).float().argmax(-1)
        first_rows = scores[:, 0, 0].float()
        gathered_index = index[:, :-1, None, None].expand(-1, -1, 1, top_k)
        later_rows = scores[:, 1:].float().gather(2, gathered_index)[:, :, 0]
        self.q_soft[:batch_size].copy_(torch.softmax(torch.cat([first_rows[:, None], later_rows], 1), -1))
        return result

    mod._selector_lattice = selector_lattice
    sampler_cls.__init__ = init
    sampler_cls.__call__ = call
    return True


def _path_probabilities(worker, draft_tokens):
    sampler = getattr(worker, "_draft_sampler", None)
    if sampler is None or getattr(sampler, "q_soft", None) is None or getattr(sampler, "candidate_out", None) is None:
        return None
    if getattr(worker, "_selector_sample", None) is not None:
        return None

    block_size = int(worker.block_size)
    batch_size = draft_tokens.shape[0] // block_size
    if batch_size == 0:
        return None
    tokens = draft_tokens.view(batch_size, block_size)[:, 1:]
    candidate_ids = sampler.candidate_out[:batch_size]
    probabilities_full = sampler.q_soft[:batch_size].float()
    hit = candidate_ids == tokens[..., None]
    probabilities = torch.where(
        hit.any(-1),
        (probabilities_full * hit).sum(-1),
        torch.zeros_like(probabilities_full[..., 0]),
    )
    return probabilities


def _build_target_state(worker, length, capture_batch_sizes):
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.runtime_context import get_context

    model_runner = worker.model_runner
    backup_init = model_runner.init_new_workspace
    overrides = {
        "speculative_num_steps": length - 1,
        "speculative_num_draft_tokens": length,
    }
    if capture_batch_sizes is not None:
        overrides["cuda_graph_bs_decode"] = capture_batch_sizes
    context = get_context()
    saved = {name: context.config_leaf(name) for name in overrides}
    try:
        context.override("adaptive_dflash_capture", **overrides)
        attn_backend = model_runner._get_attention_backend(init_new_workspace=True)
        graph_runner = DecodeCudaGraphRunner(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=length - 1,
            speculative_num_draft_tokens=length,
        )
    finally:
        context.override("adaptive_dflash_restore", **saved)
        model_runner.init_new_workspace = backup_init
    return _TargetState(length=length, attn_backend=attn_backend, graph_runner=graph_runner)


def install(mod, log=_log):
    if not _wrap_sampler(mod):
        raise RuntimeError("adaptive DFlash verify requires _SelectorDraftSampler")

    worker_cls = mod.DFlashWorkerV2
    original_verify_input = mod.DFlashVerifyInput
    original_prepare = original_verify_input.prepare_for_verify
    original_adjust = mod.apply_dflash_verify_logits_adjustments
    original_accept = worker_cls._accept_block
    original_append = worker_cls._append_target_hidden_to_draft_kv_by_loc
    original_init_graphs = worker_cls.init_cuda_graphs
    current_worker = {}

    def init_graphs(self):
        original_init_graphs(self)
        tiers = _tiers()
        full = int(self.block_size)
        if tiers[-1] != full:
            raise ValueError(f"CORAL_ADAPTIVE_VERIFY_TIERS must end with block size {full}")
        capture_bs = [int(x) for x in os.environ.get("CORAL_ADAPTIVE_VERIFY_CAPTURE_BS", "1").split(",") if x]
        self._adaptive_target_states = {}
        self._adaptive_verify_len = full
        self._adaptive_target_states[full] = _TargetState(
            length=full,
            attn_backend=self.model_runner.attn_backend,
            graph_runner=self.model_runner.decode_cuda_graph_runner,
        )
        for length in tiers[:-1]:
            self._adaptive_target_states[length] = _build_target_state(self, length, capture_bs)
        log(
            "adaptive DFlash verify: tiers "
            + "/".join(map(str, tiers))
            + f", threshold {_threshold()}, selector path-confidence"
        )

    def make_verify_input(*args, **kwargs):
        worker = current_worker.get("worker")
        if worker is None or not hasattr(worker, "_adaptive_target_states"):
            return original_verify_input(*args, **kwargs)

        full = int(worker.block_size)
        draft_token = kwargs.get("draft_token", args[0] if args else None)
        if draft_token is None or draft_token.numel() % full:
            worker._adaptive_verify_len = full
        else:
            probabilities = _path_probabilities(worker, draft_token)
            if probabilities is None:
                worker._adaptive_verify_len = full
            else:
                worker._adaptive_verify_len = _tier_from_probabilities(
                    probabilities,
                    tiers=sorted(worker._adaptive_target_states),
                    threshold=_threshold(),
                )

        length = int(worker._adaptive_verify_len)
        state = worker._adaptive_target_states[length]
        worker._adaptive_full_block_size = full
        worker._adaptive_append_active = length != full
        worker.model_runner.attn_backend = state.attn_backend
        worker.model_runner.decode_cuda_graph_runner = state.graph_runner

        if length != full:
            batch_size = draft_token.numel() // full
            draft_token = draft_token.view(batch_size, full)[:, :length].reshape(-1)
            positions = kwargs.get("positions")
            if positions is not None:
                kwargs["positions"] = positions.view(batch_size, full)[:, :length].reshape(-1)
            kwargs["draft_token"] = draft_token
            kwargs["draft_token_num"] = length
        verify_input = original_verify_input(*args, **kwargs)
        verify_input._adaptive_block_size = full
        verify_input._adaptive_verify = length != full
        return verify_input

    def prepare_for_verify(self, batch, target_worker):
        if getattr(self, "_adaptive_verify", False):
            full = int(self._adaptive_block_size)
            length = int(self.draft_token_num)
            batch_size = self.draft_token.numel() // length
            batch.out_cache_loc = batch.out_cache_loc[: batch_size * length]
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = batch.seq_lens_cpu - (full - length)
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        return original_prepare(self, batch, target_worker)

    def adjust_logits(*, next_token_logits, sampling_info, draft_token_num):
        worker = current_worker.get("worker")
        length = int(getattr(worker, "_adaptive_verify_len", draft_token_num)) if worker is not None else draft_token_num
        return original_adjust(
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            draft_token_num=length,
        )

    def accept_block(self, *, candidates, next_token_logits, sampling_info, draft_input, prefix_lens, bs):
        worker = self
        full = int(worker.block_size)
        length = int(getattr(worker, "_adaptive_verify_len", full))
        if length == full:
            return original_accept(
                self,
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_input=draft_input,
                prefix_lens=prefix_lens,
                bs=bs,
            )

        ensure_accept_buffers = getattr(self, "_ensure_accept_bonus_buffers", None)
        if ensure_accept_buffers is not None:
            ensure_accept_buffers(bs)
        self.block_size = length
        try:
            accept_len, commit_lens, bonus, out_tokens, new_seq_lens, target_predict = original_accept(
                self,
                candidates=candidates[:, :length],
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_input=draft_input,
                prefix_lens=prefix_lens,
                bs=bs,
            )
        except Exception:
            self.block_size = full
            raise
        padded_tokens = out_tokens.new_empty((bs, full))
        padded_tokens[:, :length].copy_(out_tokens[:, :length])
        padded_tokens[:, length:].fill_(0)
        return accept_len, commit_lens, bonus, padded_tokens, new_seq_lens, target_predict

    def append_target_hidden(self, *, target_hidden, cache_loc, positions, **kwargs):
        if not getattr(self, "_adaptive_append_active", False):
            return original_append(
                self,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                **kwargs,
            )
        length = int(getattr(self, "_adaptive_verify_len", self.block_size))
        full = int(getattr(self, "_adaptive_full_block_size", self.block_size))
        if length != full:
            if target_hidden.shape[0] % full == 0:
                batch_size = target_hidden.shape[0] // full
            elif target_hidden.shape[0] % length == 0:
                batch_size = target_hidden.shape[0] // length
            else:
                raise ValueError(
                    "adaptive DFlash hidden-state row count is not divisible by "
                    f"full={full} or length={length}: {target_hidden.shape[0]}"
                )
            cache_loc_2d = kwargs.get("cache_loc_2d")
            if cache_loc_2d is not None:
                kwargs["cache_loc_2d"] = cache_loc_2d[:, :length]
            try:
                return original_append(
                    self,
                    target_hidden=target_hidden[: batch_size * length],
                    cache_loc=cache_loc[: batch_size * length],
                    positions=positions[: batch_size * length],
                    **kwargs,
                )
            finally:
                self.block_size = full
                self._adaptive_append_active = False
        return original_append(
            self,
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            **kwargs,
        )

    original_worker_init = worker_cls.__init__

    def worker_init(self, *args, **kwargs):
        result = original_worker_init(self, *args, **kwargs)
        current_worker["worker"] = self
        return result

    mod.DFlashVerifyInput = make_verify_input
    original_verify_input.prepare_for_verify = prepare_for_verify
    mod.apply_dflash_verify_logits_adjustments = adjust_logits
    worker_cls._accept_block = accept_block
    worker_cls._append_target_hidden_to_draft_kv_by_loc = append_target_hidden
    worker_cls.init_cuda_graphs = init_graphs
    worker_cls.__init__ = worker_init
