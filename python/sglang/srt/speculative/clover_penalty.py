# SPDX-License-Identifier: Apache-2.0
import math
from collections import Counter

import torch


class CloverPenaltyState:
    def __init__(self):
        self._histories = {}
        self._committed = {}
        self._parameters = {}

    def retain(self, reqs):
        active = {id(req) for req in reqs}
        self._histories = {
            key: value for key, value in self._histories.items() if key in active
        }
        self._committed = {
            key: value for key, value in self._committed.items() if key in active
        }
        self._parameters = {
            key: value for key, value in self._parameters.items() if key in active
        }

    def parameters(self, req, device):
        sp = req.sampling_params
        values = (sp.frequency_penalty, sp.presence_penalty, sp.repetition_penalty)
        previous = self._parameters.get(id(req))
        if (
            previous is None
            or previous[0] is not req
            or previous[1] != values
            or previous[2].device != device
        ):
            params = torch.tensor(
                values, dtype=torch.float32, pin_memory=device.type == "cuda"
            ).to(device, non_blocking=True)
            self._parameters[id(req)] = (req, values, params)
            return params
        return previous[2]

    def history(self, req, vocab_size):
        previous = self._histories.get(id(req))
        output = req.output_ids
        if (
            previous is None
            or previous[0] is not req
            or previous[1] is not output
            or previous[2] > len(output)
        ):
            old_len, counts = 0, Counter()
        else:
            old_len, counts = previous[2:]
        counts.update(t for t in output[old_len:] if 0 <= t < vocab_size)
        self._histories[id(req)] = (req, output, len(output), counts)
        return counts

    def committed_counts(self, req, vocab_size, device):
        previous = self._committed.get(id(req))
        output = req.output_ids
        if (
            previous is None
            or previous[0] is not req
            or previous[1] is not output
            or previous[2] > len(output)
            or previous[3].numel() != vocab_size
            or previous[3].device != device
        ):
            old_len = 0
            counts = torch.zeros(vocab_size, device=device, dtype=torch.float32)
        else:
            old_len, counts = previous[2:]
        if old_len < len(output):
            delta = [t for t in output[old_len:] if 0 <= t < vocab_size]
            if delta:
                # Fresh pinned storage outlives the async copy, a reused one races it.
                ids = torch.tensor(
                    delta, dtype=torch.long, pin_memory=device.type == "cuda"
                ).to(device, non_blocking=True)
                counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
        self._committed[id(req)] = (req, output, len(output), counts)
        return counts


def validate_clover_penalties(req):
    sp = req.sampling_params
    custom = {
        "dllm_length_penalty": 10.0,
        "dllm_think_end_bias": 10.0,
        "dllm_think_end_start": 4096,
        **(sp.custom_params or {}),
    }
    for key in ("dllm_length_penalty", "dllm_think_end_bias"):
        value = custom.get(key, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return f"CLOVER custom_params.{key} must be finite and non-negative"
    for key, default, minimum in (
        ("dllm_length_penalty_start", 8192, 0),
        ("dllm_length_penalty_window", 8192, 1),
        ("dllm_think_end_start", 4096, 0),
        ("dllm_think_end_window", 4096, 1),
    ):
        value = custom.get(key, default)
        if type(value) is not int or value < minimum:
            return f"CLOVER custom_params.{key} must be an integer >= {minimum}"
    token = custom.get("dllm_think_end_token_id")
    tokenizer = req.tokenizer
    if custom.get("dllm_think_end_bias", 0) > 0 and token is None:
        encoded = (
            tokenizer.encode("</think>", add_special_tokens=False)
            if tokenizer is not None
            else []
        )
        if len(encoded) != 1 or encoded[0] == getattr(tokenizer, "unk_token_id", None):
            return "CLOVER think-end bias requires a single dllm_think_end_token_id"
        token = encoded[0]
    if token is not None:
        if (
            type(token) is not int
            or token < 0
            or (tokenizer is not None and token >= len(tokenizer))
        ):
            return "CLOVER dllm_think_end_token_id must be a valid vocabulary ID"
        custom["dllm_think_end_token_id"] = token
    sp.custom_params = custom
    return None


def _stop_tokens(req, vocab_size):
    tokenizer = req.tokenizer
    stops = set(req.sampling_params.stop_token_ids or ())
    stops.update(req.eos_token_ids or ())
    stops.update(getattr(tokenizer, "additional_stop_token_ids", None) or ())
    stops.add(getattr(tokenizer, "eos_token_id", None))
    return sorted(t for t in stops if t is not None and 0 <= t < vocab_size)


def apply_clover_penalties(logits, reqs, visible_tokens, *, preview=False, state=None):
    """Modify [batch, width, vocab] logits in place; explicitly disabled penalties are a no-op.

    Row zero of the preview is an already committed pending token; row one
    predicts the next output. Verify row zero predicts that same next output.
    Length ramps use actual output lengths plus these aligned offsets. Custom
    biases apply to speculative rounds; initial prefill sampling retains the
    native penalties.
    """
    _, width, vocab_size = logits.shape
    if state is None:
        state = CloverPenaltyState()
    state.retain(reqs)
    for row, req in enumerate(reqs):
        sp = req.sampling_params
        custom = sp.custom_params or {}
        length_bias = custom.get("dllm_length_penalty", 0.0)
        think_bias = custom.get("dllm_think_end_bias", 0.0)
        if not (
            sp.frequency_penalty
            or sp.presence_penalty
            or sp.repetition_penalty != 1
            or sp.min_new_tokens
            or length_bias
            or think_bias
        ):
            continue
        scores = logits[row]
        history = state.history(req, vocab_size)
        if sp.frequency_penalty or sp.presence_penalty or sp.repetition_penalty != 1:
            counts = state.committed_counts(req, vocab_size, logits.device)
            if (
                scores.is_cuda
                and scores.dtype == torch.float32
                and scores.stride(1) == 1
                and width <= 128
            ):
                from sglang.kernels.ops.sampling.clover_penalty import clover_penalty

                clover_penalty(
                    scores,
                    counts,
                    state.parameters(req, logits.device),
                    frequency=bool(sp.frequency_penalty),
                    presence=bool(sp.presence_penalty),
                    repetition=sp.repetition_penalty != 1,
                )
            else:
                seen = counts > 0
                if sp.frequency_penalty:
                    scores.sub_(counts * sp.frequency_penalty)
                if sp.presence_penalty:
                    scores.sub_(seen.to(scores.dtype) * sp.presence_penalty)
                if sp.repetition_penalty != 1:
                    scale = torch.where(seen, sp.repetition_penalty, 1.0)
                    from sglang.srt.sampling.penaltylib.repetition_penalty import (
                        apply_scaling_penalties,
                    )

                    apply_scaling_penalties(scores, scale[None, :])

        # One-based response positions of the token predicted by each row.
        first_position = len(req.output_ids) + (0 if preview else 1)
        positions = range(first_position, first_position + width)
        stops = _stop_tokens(req, vocab_size)
        if (
            length_bias
            and stops
            and not sp.ignore_eos
            and first_position + width - 1
            > custom.get("dllm_length_penalty_start", 8192)
        ):
            start = custom.get("dllm_length_penalty_start", 8192)
            window = custom.get("dllm_length_penalty_window", 8192)
            ramp = torch.tensor(
                [length_bias * min(max((p - start) / window, 0), 1) for p in positions],
                device=logits.device,
                dtype=logits.dtype,
            )
            scores[:, stops] += ramp[:, None]
        token = custom.get("dllm_think_end_token_id")
        if think_bias and token is not None and token not in history:
            start = custom.get("dllm_think_end_start", 4096)
            window = custom.get("dllm_think_end_window", 4096)
            # Only the first eligible position gets a bias. During verification
            # an uncommitted draft </think> must not disable its own bias.
            index = max(int(preview), start + 1 - first_position)
            if index < width:
                bias = think_bias * min((first_position + index - start) / window, 1)
                if preview:
                    bias = bias * ~(visible_tokens[row] == token).any()
                scores[index, token] += bias
        # Hard minimum wins over either positive bias. The EOS itself does not
        # count towards the required number of preceding response tokens.
        if sp.min_new_tokens and stops:
            num_blocked = min(width, max(0, sp.min_new_tokens - first_position + 1))
            scores[:num_blocked, stops] = -float("inf")
