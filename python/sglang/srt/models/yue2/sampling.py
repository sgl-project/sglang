# SPDX-License-Identifier: Apache-2.0
"""Request-local sampling, preserving mode-specific historical arithmetic."""
from __future__ import annotations
import os
import time
import torch
from .protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT

# NOTE (yiakwy) : same to https://github.com/multimodal-art-projection/YuE/blob/main/src/yue2/sampling.py

# TODO（yiakwy）: suport Yue2 sampling with SGLang batched sampling info

def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def window_penalty(logits, recent_ids, penalty):
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty ** freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(logits, sampling, history, step, phase, legacy_off=False):
    # vLLM's symbolic processor receives FP32 logits; historical off uses BF16.
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-sampling.penalty_window:], sampling.repetition_penalty)

    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature

    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., :3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


def _sample_next(sampler, logits, sampling, phase, history, history_buf, step,
                 generator, device, legacy_off):
    if sampler is not None:
        return sampler.sample_token(
            logits, phase, sampling.temperature, sampling.top_p, sampling.top_k,
            history_buf, len(history), sampling.repetition_penalty,
            sampling.penalty_window, sampling.min_tokens, step, generator,
        )
    scores = distribution(logits, sampling, history, step, phase, legacy_off)
    if sampling.temperature == 0:
        return scores.argmax(-1, keepdim=True)
    probabilities = scores.softmax(-1)
    if device.type == "mps":
        return torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
    return torch.multinomial(probabilities, 1, generator=generator)


def _decode_loop(*, step_fn, logits_pair, sampling, phase, cfg_scale, sampler, generator,
                 device, legacy_off, cancelled, on_token, history_buf, start,
                 prefill_seconds, execution, attention, prefix_tokens, cfg_branches):
    """Shared AR decode loop; ``step_fn(token)`` returns the next (cond, uncond) pair."""
    conditional, unconditional = logits_pair
    history, first, eos = [], None, False
    end = ABC_END if phase == "abc" else MUSIC_END
    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError(f"Cancelled during {phase}")
        # Preserve historical BF16 CFG subtraction/multiply/add before upcast.
        logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
        next_id = _sample_next(sampler, logits, sampling, phase, history, history_buf, step,
                               generator, device, legacy_off)
        token = int(next_id.item())
        if sampler is not None:
            history_buf[len(history)] = next_id.reshape(-1)[0]
        if first is None:
            first = time.perf_counter() - start
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break

        history.append(token)
        if step + 1 < sampling.max_tokens:
            conditional, unconditional = step_fn(next_id)

    synchronize(device)

    seconds = time.perf_counter() - start
    count = len(history) + int(eos)
    timing = {"seconds": seconds, "prefill_seconds": prefill_seconds,
              "ttft_seconds": first, "output_tokens": count, "content_tokens": len(history),
              "output_tps": count / seconds, "prefix_tokens": prefix_tokens,
              "cfg_branches": cfg_branches,
              "execution": execution, "attention": attention,
              "sampler": "deepselect" if sampler is not None else "reference"}

    fed = len(history) - (0 if eos else 1)
    return history, timing, (not eos), max(fed, 0)


@torch.inference_mode()
def generate_tokens(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, cancelled=None, on_token=None, use_cuda_graph=True,
                    use_fast_sampler=None):
    from .modeling_yue2 import StaticKVCache
    from .fastpath import FastSampler, fast_sampler_available
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")
    # The two stages deliberately reset their request-local seed, matching the preset.
    rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    config = model.config
    fast = fast_sampler_available() if use_fast_sampler is None else use_fast_sampler
    sampler = FastSampler(config.vocab_size, device) if fast and device.type == "cuda" else None

    def prefill(ids):
        cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
                              num_kv_heads=config.num_key_value_heads,
                              max_seq_len=len(ids) + sampling.max_tokens,
                              head_dim=config.head_dim, dtype=dtype, device=device)
        output = model(torch.tensor([ids], device=device), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        return output.logits[:, -1, :], output.past_key_values

    graph = None
    pooled = False
    positive_cache = negative_cache = None
    graph_enabled = use_cuda_graph and device.type == "cuda" and not getattr(model, "_yue2_fp8_originals", {})
    if graph_enabled:
        from .cuda_graph import default_session_pool
        pooled = default_session_pool is not None and os.environ.get("SGLANG_YUE2_GRAPH_POOL", "1") == "1"
    synchronize(device)
    start = time.perf_counter()
    try:
        if graph_enabled:
            if pooled:
                from .cuda_graph import default_session_pool as pool
                graph = pool.acquire(model, [prefix] if cfg_scale == 1 else [prefix, negative],
                                     sampling.max_tokens)
            else:
                from .cuda_graph import GraphAR
                graph = GraphAR(model, [prefix] if cfg_scale == 1 else [prefix, negative], sampling.max_tokens)
            logits = graph.prefill()
            logits_pair = (logits[:1], logits[1:] if cfg_scale != 1 else None)
            execution, attention = "cuda_graph", graph.attention_backend

            def step_fn(token):
                branch = graph.step(token)
                return branch[:1], branch[1:] if cfg_scale != 1 else None
        else:
            conditional, positive_cache = prefill(prefix)
            unconditional = None
            if cfg_scale != 1.0:
                unconditional, negative_cache = prefill(negative)
            logits_pair = (conditional, unconditional)
            execution, attention = "eager", "sdpa"

            def step_fn(token):
                out = model(token, past_key_values=positive_cache, use_cache=True,
                            logits_to_keep=1).logits[:, -1, :]
                if negative_cache is not None:
                    return out, model(token, past_key_values=negative_cache, use_cache=True,
                                      logits_to_keep=1).logits[:, -1, :]
                return out, None
        synchronize(device)
        prefill_seconds = time.perf_counter() - start
        history_buf = (torch.empty(sampling.max_tokens, dtype=torch.long, device=device)
                       if sampler is not None else None)
        history, timing, truncated, _fed = _decode_loop(
            step_fn=step_fn, logits_pair=logits_pair, sampling=sampling, phase=phase,
            cfg_scale=cfg_scale, sampler=sampler, generator=generator, device=device,
            legacy_off=legacy_off, cancelled=cancelled, on_token=on_token,
            history_buf=history_buf, start=start, prefill_seconds=prefill_seconds,
            execution=execution, attention=attention, prefix_tokens=len(prefix),
            cfg_branches=1 if cfg_scale == 1 else 2)
        if use_cuda_graph and not graph_enabled:
            timing["graph_fallback_reason"] = "fp8_not_graph_validated" if getattr(model, "_yue2_fp8_originals", {}) else "non_cuda_device"
        return history, timing, truncated
    finally:
        if graph is not None:
            if pooled:
                from .cuda_graph import default_session_pool as pool
                pool.release(graph)
            else:
                graph.close()
        positive_cache = negative_cache = None
