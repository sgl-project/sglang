# SPDX-License-Identifier: Apache-2.0
# From Diffuspace/Clover-35B-A3B-Preview sampling_clover_moe.py.
"""Response-only penalties for Clover block LowConfidence generation.

Presence, frequency and repetition use committed response history.
Length and thinking controls provide gradual stopping biases.
All state belongs to one generate request; prompt tokens are never accumulated.
"""

import math
from copy import copy

import torch

PENALTY_DEFAULTS = {
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "dllm_length_penalty": 0.0,
    "dllm_length_penalty_start": 8192,
    "dllm_length_penalty_window": 8192,
    "dllm_think_end_bias": 0.0,
    "dllm_think_end_start": 4096,
    "dllm_think_end_window": 4096,
    "dllm_think_end_token_id": None,
    "stop_token_ids": [],
}


def validate_threshold(value):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("Clover threshold must be a number in [0, 1]")


def resolve_stop_ids(gen, config):
    eos = (
        gen.eos_token_id
        if gen.eos_token_id is not None
        else getattr(config, "eos_token_id", None)
    )
    stops = set([eos] if isinstance(eos, int) else eos or ())
    stops.update(getattr(gen, "stop_token_ids", None) or ())
    pad = (
        gen.pad_token_id
        if gen.pad_token_id is not None
        else getattr(config, "pad_token_id", None)
    )
    return stops, pad if pad is not None else min(stops, default=0)


def validate_penalties(gen, vocab_size):
    for name, default in PENALTY_DEFAULTS.items():
        if getattr(gen, name, None) is None:
            setattr(gen, name, default)
    for name in ("presence_penalty", "frequency_penalty"):
        value = getattr(gen, name)
        if not math.isfinite(value) or not -2 <= value <= 2:
            raise ValueError(f"{name} must be finite and in [-2, 2]")
    if not math.isfinite(gen.repetition_penalty) or not 0 < gen.repetition_penalty <= 2:
        raise ValueError("repetition_penalty must be finite and in (0, 2]")
    for prefix, strength in (
        ("dllm_length_penalty", "dllm_length_penalty"),
        ("dllm_think_end", "dllm_think_end_bias"),
    ):
        value = getattr(gen, strength)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{strength} must be finite and nonnegative")
        for suffix, minimum in (("start", 0), ("window", 1)):
            name = f"{prefix}_{suffix}"
            value = getattr(gen, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
    close = gen.dllm_think_end_token_id
    if close is not None and (
        not isinstance(close, int) or not 0 <= close < vocab_size
    ):
        raise ValueError("dllm_think_end_token_id must be a valid token ID")
    if gen.dllm_think_end_bias and close is None:
        raise ValueError("dllm_think_end_bias requires dllm_think_end_token_id")
    if not isinstance(gen.stop_token_ids, (list, tuple)) or any(
        not isinstance(i, int) or not 0 <= i < vocab_size for i in gen.stop_token_ids
    ):
        raise ValueError("stop_token_ids must be a list of valid token IDs")


class CloverPenaltyState:
    def __init__(self, gen, vocab_size, device, stop_ids):
        self.gen = gen
        self.length = 0
        self.active = bool(
            gen.presence_penalty
            or gen.frequency_penalty
            or gen.repetition_penalty != 1
            or gen.dllm_length_penalty
            or gen.dllm_think_end_bias
        )
        self.counts = (
            torch.zeros(vocab_size, dtype=torch.int64, device=device)
            if self.active
            else None
        )
        self.stop_ids = sorted({i for i in stop_ids if 0 <= i < vocab_size})

    def commit(self, tokens):
        """Count each returned response token once, after full block resolution."""
        self.length += tokens.numel()
        if self.active:
            self.counts.index_add_(
                0, tokens, torch.ones_like(tokens, dtype=self.counts.dtype)
            )

    def apply(self, logits, block, mask_id):
        """Mutate fresh [block, vocab] logits before sampling and confidence."""
        if not self.active:
            return logits
        gen = self.gen
        # Uncommitted block tokens must not penalize themselves or other slots.
        # This snapshot changes only when the returned block is committed.
        seen = self.counts > 0
        # Keep SGLang's additive operation order and FP32 rounding boundaries.
        if gen.frequency_penalty:
            logits.sub_(self.counts.float() * gen.frequency_penalty)
        if gen.presence_penalty:
            logits.sub_(seen * gen.presence_penalty)
        if gen.repetition_penalty != 1:
            values = logits[:, seen]
            logits[:, seen] = torch.where(
                values < 0,
                values * gen.repetition_penalty,
                values / gen.repetition_penalty,
            )
        positions = self.length + torch.arange(
            1, logits.shape[0] + 1, device=logits.device
        )

        def ramp(start, window, strength):
            return ((positions.float() - start) / window).clamp(0, 1) * strength

        if gen.dllm_length_penalty and self.stop_ids:
            logits[:, self.stop_ids] += ramp(
                gen.dllm_length_penalty_start,
                gen.dllm_length_penalty_window,
                gen.dllm_length_penalty,
            )[:, None]
        if gen.dllm_think_end_bias:
            close = gen.dllm_think_end_token_id
            # In LowConfidence the whole block receives the bias until a close
            # token is visible or committed.
            visible_close = ((block != mask_id) & (block == close)).any()
            enabled = ~(seen[close] | visible_close)
            logits[:, close] += enabled * ramp(
                gen.dllm_think_end_start,
                gen.dllm_think_end_window,
                gen.dllm_think_end_bias,
            )
        return logits


def greedy_with_confidence(logits):
    prediction = logits.argmax(-1)
    confidence = logits.softmax(-1).gather(-1, prediction[:, None]).squeeze(-1)
    return prediction, confidence


def prepare_sampling_params(params, model_config, prompt_len, context_len):
    """Resolve checkpoint generation defaults before the generic sampler normalizes them."""
    defaults = copy(model_config.hf_generation_config)
    for key, value in (params.get("custom_params") or {}).items():
        if key in ("threshold", "max_denoise_steps", "refine_uncertain"):
            validate_decode_option(key, value, model_config.hf_config.block_size)
            continue
        if not key.startswith("dllm_") or key not in PENALTY_DEFAULTS:
            raise ValueError(f"Unsupported CloverPureBlock custom parameter: {key}")
        setattr(defaults, key, value)
    validate_penalties(defaults, model_config.vocab_size)
    for key in ("presence_penalty", "frequency_penalty", "repetition_penalty"):
        params.setdefault(key, getattr(defaults, key))
    params.setdefault("temperature", 0.0)
    width = model_config.hf_config.block_size
    stops, _ = resolve_stop_ids(defaults, model_config.hf_config)
    params["stop_token_ids"] = sorted(stops.union(params.get("stop_token_ids") or ()))
    budget = params.get("max_new_tokens")
    if budget is None:
        # Scheduler reserves two positions for admission and the KV allocator.
        available = max(0, context_len - prompt_len - 2) // width * width
        budget = (
            min(defaults.max_new_tokens, available)
            if defaults.max_new_tokens is not None
            else available
        )
        params["max_new_tokens"] = budget
    if prompt_len + ((budget + width - 1) // width) * width > context_len:
        raise ValueError(
            "Prompt plus full Clover preview blocks exceeds the context limit"
        )


def validate_decode_option(key, value, block_size):
    if key == "refine_uncertain" and type(value) is not bool:
        raise ValueError("refine_uncertain must be a boolean")
    if key == "threshold":
        validate_threshold(value)
    if (
        key == "max_denoise_steps"
        and value is not None
        and (type(value) is not int or not 1 <= value <= block_size)
    ):
        raise ValueError("max_denoise_steps must be within the Clover block width")
