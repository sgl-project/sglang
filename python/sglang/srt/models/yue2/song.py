# SPDX-License-Identifier: Apache-2.0
"""ABC and semantic phases AR decoding graph sharing.

The semantic prefix extends the ABC prefix with tokens already processed 
by the model.

Instead of re-prefilling, the session feeds the remaining ABC tail and 
two separators as ordinary decode steps, then continues the semantic loop. 
"""
from __future__ import annotations

import os
import time

import torch

from .protocol import ABC_END, MUSIC_END, Sampling, SongRequest, token_prefixes
from .sampling import _decode_loop, synchronize

from .fastpath import FastSampler, fast_sampler_available
from .fastpath import GraphSampler, _load_topk_backend

from .protocol import ABC_END, MUSIC_END

@torch.inference_mode()
def generate_song_tokens(model, session, tokenizer, request: SongRequest, *,
                         abc_sampling: Sampling, semantic_sampling: Sampling,
                         cfg_scale: float = 1.0, negative_prefix_ids=None,
                         cancelled=None, on_token=None, use_fast_sampler=None):
    """Run both AR phases on one pooled session.

    Returns ``(abc_ids, codec_ids, timing, fed_codec)``. ``fed_codec`` counts
    the codec tokens already written into the session KV cache; the NAR stage
    feeds the remainder plus ``MUSIC_END``. The session is left open with the
    full AR KV (prefix + codec + MUSIC_END) resident.
    """
    if (cfg_scale == 1.0 and request.cot != "off"
            and os.environ.get("SGLANG_YUE2_STEP_GRAPH", "1") == "1"
            and session.device.type == "cuda"):
        try:
            return _generate_song_tokens_step_graph(
                model, session, tokenizer, request,
                abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
                cancelled=cancelled, on_token=on_token)
        except _StepGraphUnavailable:
            pass

    return _generate_song_tokens_looped(
        model, session, tokenizer, request,
        abc_sampling=abc_sampling, semantic_sampling=semantic_sampling,
        cfg_scale=cfg_scale, negative_prefix_ids=negative_prefix_ids,
        cancelled=cancelled, on_token=on_token, use_fast_sampler=use_fast_sampler)


def song_token_budget(abc_sampling: Sampling, semantic_sampling: Sampling) -> int:
    """Generated-token budget for one pooled song session.

    GraphAR sizes capacity as len(prefix) + max_tokens, so this counts only
    tokens decoded beyond the prefix: the ABC stream, the two separators
    (ABC_END, MUSIC_START), the codec stream and the terminating MUSIC_END
    that the NAR stage feeds, plus one slot of slack.
    """
    return (int(abc_sampling.max_tokens) + int(semantic_sampling.max_tokens) + 4)


class _StepGraphUnavailable(Exception):
    pass


def _generate_song_tokens_step_graph(model, session, tokenizer, request: SongRequest, *,
                                     abc_sampling: Sampling, semantic_sampling: Sampling,
                                     cancelled=None, on_token=None):
    """ABC + transition + semantic phases through the fused decode+sample graph.

    Every replay decodes the token in ``token_buf`` and samples the next one
    on-GPU; transition tokens (ABC tail, ABC_END, MUSIC_START) are forced via
    single-entry masks, so the whole song runs end-to-end without per-step
    CPU sampling work.
    """
    if _load_topk_backend() is None:
        raise _StepGraphUnavailable("DeepSelect backend missing")

    device = session.device
    
    # The captured graph is bound to the first sampler's buffers — reuse it
    # across requests (reseed + reset) instead of allocating a fresh one.
    sampler = getattr(session, "sampler", None)
    if sampler is None:
        sampler = GraphSampler(model.config.vocab_size, device,
                               torch.Generator(device=device).manual_seed(request.seed))
        if session.graph is not None and not session.step_mode:
            raise _StepGraphUnavailable("session already captured without a sampler")
        session.sampler = sampler
        session.step_mode = True
    else:
        sampler.generator.manual_seed(request.seed)
        sampler.reset_state()

    prefix = token_prefixes(request, tokenizer)
    budget = song_token_budget(abc_sampling, semantic_sampling)
    session.reopen([prefix], budget)
    synchronize(device)
    start = time.perf_counter()
    logits = session.prefill()
    execution, attention = "cuda_graph", session.attention_backend

    def sample_phase(phase, sampling, end, eager_logits):
        """One sampling phase: optional eager first sample, then replay loop.

        Returns (tokens, open, replays): ``tokens`` excludes a trailing EOS;
        ``replays`` equals the number of tokens the graph decoded in this
        phase (the semantic phase's first replay decodes MUSIC_START).
        """
        sampler.configure_phase(phase, sampling.temperature, sampling.top_p,
                                sampling.top_k, sampling.repetition_penalty,
                                sampling.penalty_window, end, sampling.min_tokens)
        blocked = sampling.min_tokens > 0
        if blocked:
            sampler.mask[0, end] = float("-inf")
        tokens = []
        if eager_logits is not None:
            # First sample (step 0) runs eagerly on the provided logits.
            sampler.sample_in_graph(eager_logits)
            token = int(sampler.token_buf.item())
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                return [], False, 0
            tokens.append(token)
            sampler.advance_eager()
        replays = 0
        open_ = True
        offset = 1 if eager_logits is not None else 0
        for _ in range(int(sampling.max_tokens) - offset):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            # Reference blocks the end token for sample indices < min_tokens;
            # replay r produces sample index r + offset.
            if blocked and replays + offset >= sampling.min_tokens:
                sampler.mask[0, end] = 0.0
                blocked = False
            session.graph.replay()
            replays += 1
            token = int(sampler.token_buf.item())
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                open_ = False
                break
            tokens.append(token)
        return tokens, open_, replays

    if request.abc is not None:
        # Skip ABC generation entirely
        abc_ids = tokenizer.encode(request.abc)
        semantic_prefix = prefix
        transition_seconds = 0.0
        abc_seconds = 0.0
        abc_open = True
    else:
        # Phase 1: ABC planning tokens (t_0 sampled from the prefill logits).
        abc_ids, abc_open, abc_replays = sample_phase(
            "abc", abc_sampling, ABC_END, logits[:1])
        fed_abc = abc_replays  # each replay decoded one ABC token
        abc_seconds = time.perf_counter() - start

        # Phase 2: forced transition into the semantic prefix.
        transition_start = time.perf_counter()
        semantic_prefix = token_prefixes(request, tokenizer, abc_ids=abc_ids)
        remaining = semantic_prefix[len(prefix) + fed_abc:]
        if not abc_open:
            # token_buf holds the sampled EOS token, which is never decoded;
            # replace it with the first transition token.
            sampler.token_buf.fill_(remaining[0])
            remaining = remaining[1:]
        for token in remaining:
            sampler.force_token(token)
            sampler.done_buf.zero_()
            session.graph.replay()
        transition_seconds = time.perf_counter() - transition_start

    # Phase 3: semantic codec tokens.
    t0 = time.perf_counter()
    codec_ids, semantic_open, semantic_replays = sample_phase(
        "semantic", semantic_sampling, MUSIC_END,
        logits[:1] if request.abc is not None else None)
    fed_codec = max(semantic_replays - (0 if request.abc is not None else 1), 0)

    semantic_timing = {"seconds": time.perf_counter() - t0, "prefill_seconds": 0.0,
                       "output_tokens": len(codec_ids), "content_tokens": len(codec_ids),
                       "transition_seconds": transition_seconds,
                       "truncated": bool(semantic_open), "execution": execution,
                       "attention": attention, "prefix_tokens": len(semantic_prefix),
                       "cfg_branches": 1, "sampler": "deepselect_step_graph"}
    
    abc_timing = {"seconds": abc_seconds, "prefill_seconds": 0.0,
                  "output_tokens": len(abc_ids), "content_tokens": len(abc_ids),
                  "truncated": bool(abc_open), "execution": execution,
                  "attention": attention, "prefix_tokens": len(prefix),
                  "cfg_branches": 1, "sampler": "deepselect_step_graph"}
    
    timing = {"abc": abc_timing, "semantic": semantic_timing,
              "total_seconds": time.perf_counter() - start}
    return abc_ids, codec_ids, timing, fed_codec


def _generate_song_tokens_looped(model, session, tokenizer, request: SongRequest, *,
                                 abc_sampling: Sampling, semantic_sampling: Sampling,
                                 cfg_scale: float = 1.0, negative_prefix_ids=None,
                                 cancelled=None, on_token=None, use_fast_sampler=None):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    config = model.config
    fast = fast_sampler_available() if use_fast_sampler is None else use_fast_sampler
    sampler = FastSampler(config.vocab_size, device) if fast and device.type == "cuda" else None

    prefix = token_prefixes(request, tokenizer)
    budget = song_token_budget(abc_sampling, semantic_sampling)
    prefixes = [prefix] if cfg_scale == 1 else [prefix, negative_prefix_ids]
    session.reopen(prefixes, budget)
    synchronize(device)
    start = time.perf_counter()
    logits = session.prefill()
    execution, attention = "cuda_graph", session.attention_backend

    def branch(logits):
        return logits[:1], logits[1:] if cfg_scale != 1 else None

    def step_fn(token):
        return branch(session.step(token))

    legacy_semantic = request.cot == "off"
    if request.abc is not None or legacy_semantic:
        # External ABC / cot=off: the prefix already is the full semantic
        # prefix — skip ABC generation and the transition entirely. cot=off
        # additionally restores the historical (vLLM-era) semantic arithmetic
        # via the reference sampler, matching upstream `legacy_off`.
        abc_ids = tokenizer.encode(request.abc) if request.abc is not None else []
        abc_open = False  # nothing generated -> nothing truncated
        semantic_prefix = prefix
        logits_pair = branch(logits)
        transition_seconds = 0.0
        abc_timing = {"seconds": 0.0, "prefill_seconds": 0.0, "output_tokens": len(abc_ids),
                      "content_tokens": len(abc_ids), "truncated": False,
                      "execution": execution, "attention": attention,
                      "prefix_tokens": len(prefix),
                      "cfg_branches": 1 if cfg_scale == 1 else 2,
                      "sampler": "external_abc" if request.abc is not None else "legacy_off"}
    else:
        legacy_semantic = False
        # Phase 1: ABC planning tokens (no CFG; both branches share the tokens).
        abc_ids, abc_timing, abc_open, fed_abc = _decode_loop(
            step_fn=step_fn, logits_pair=branch(logits), sampling=abc_sampling, phase="abc",
            cfg_scale=1.0, sampler=sampler,
            generator=torch.Generator(device=device if device.type in {"cpu", "cuda"} else "cpu").manual_seed(request.seed),
            device=device, legacy_off=False, cancelled=cancelled, on_token=on_token,
            history_buf=(torch.empty(abc_sampling.max_tokens, dtype=torch.long, device=device)
                         if sampler is not None else None),
            start=start, prefill_seconds=time.perf_counter() - start,
            execution=execution, attention=attention, prefix_tokens=len(prefix),
            cfg_branches=1 if cfg_scale == 1 else 2)

        # Phase 2: feed the semantic-prefix tail, then decode codec tokens.
        transition_start = time.perf_counter()
        semantic_prefix = token_prefixes(request, tokenizer, abc_ids=abc_ids)
        remaining = semantic_prefix[len(prefix) + fed_abc:]
        logits_pair = None
        for token in remaining:
            logits_pair = step_fn(torch.tensor([[token]], dtype=torch.long, device=device))
        transition_seconds = time.perf_counter() - transition_start

    codec_ids, semantic_timing, semantic_open, fed_codec = _decode_loop(
        step_fn=step_fn, logits_pair=logits_pair, sampling=semantic_sampling, phase="semantic",
        cfg_scale=cfg_scale, sampler=None if legacy_semantic else sampler,
        legacy_off=legacy_semantic,
        generator=torch.Generator(device=device if device.type in {"cpu", "cuda"} else "cpu").manual_seed(request.seed),
        device=device, cancelled=cancelled, on_token=on_token,
        history_buf=(torch.empty(semantic_sampling.max_tokens, dtype=torch.long, device=device)
                     if sampler is not None else None),
        start=start, prefill_seconds=abc_timing["prefill_seconds"],
        execution=execution, attention=attention, prefix_tokens=len(semantic_prefix),
        cfg_branches=1 if cfg_scale == 1 else 2)
    
    semantic_timing["transition_seconds"] = transition_seconds
    semantic_timing["truncated"] = bool(semantic_open)
    abc_timing["truncated"] = bool(abc_open)
    timing = {"abc": abc_timing, "semantic": semantic_timing,
              "total_seconds": time.perf_counter() - start}
    return abc_ids, codec_ids, timing, fed_codec
