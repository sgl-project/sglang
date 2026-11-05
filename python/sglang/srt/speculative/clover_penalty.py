# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import weakref

import torch


def copy_clover_sampling_info(sampling_info):
    """Disable standard penalties; Clover applies output penalties explicitly."""
    return dataclasses.replace(
        sampling_info,
        penalizer_orchestrator=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
    )


class CloverPenaltyState:
    """CPU history metadata; vocabulary-sized tensors belong to native penalizers."""

    def __init__(self):
        self._histories = {}
        self._owner = None
        self._synced = {}

    def merge(self, other):
        self._synced.update(other._synced)
        self._histories.update(other._histories)

    def retain(self, reqs):
        active = {id(req) for req in reqs}
        self._histories = {
            key: value for key, value in self._histories.items() if key in active
        }
        self._synced = {
            key: value for key, value in self._synced.items() if key in active
        }

    def history(self, req, vocab_size):
        previous = self._histories.get(id(req))
        output = req.output_ids
        if (
            previous is None
            or previous[0] is not req
            or previous[1] is not output
            or previous[2] > len(output)
        ):
            old_len, seen = 0, set()
        else:
            old_len, seen = previous[2:]
        seen.update(t for t in output[old_len:] if 0 <= t < vocab_size)
        self._histories[id(req)] = (req, output, len(output), seen)
        return seen

    def sync_native(self, orchestrator):
        """Sync committed suffixes into the rows carried by native filter and merge."""
        from sglang.srt.sampling import penaltylib

        reqs = orchestrator.reqs()
        self.retain(reqs)
        specs = (
            (penaltylib.BatchedFrequencyPenalizer, "cumulated_frequency_penalties", 0),
            (penaltylib.BatchedPresencePenalizer, "cumulated_presence_penalties", 0),
            (
                penaltylib.BatchedRepetitionPenalizer,
                "cumulated_repetition_penalties",
                1,
            ),
        )
        penalizers = [orchestrator.penalizers.get(cls) for cls, _, _ in specs]
        buffers = tuple(
            getattr(p, name) if p is not None and p.is_prepared() else None
            for p, (_, name, _) in zip(penalizers, specs)
        )
        if self._owner is not None and self._owner() is not orchestrator:
            self._synced.clear()
        self._owner = weakref.ref(orchestrator)
        if not any(buf is not None for buf in buffers):
            return buffers
        device = torch.device(orchestrator.device)
        vocab = orchestrator.vocab_size
        for row, req in enumerate(reqs):
            previous = self._synced.get(id(req))
            output = req.output_ids
            reset = (
                previous is None
                or previous[0] is not output
                or previous[1] > len(output)
            )
            old_len = 0 if reset else previous[1]
            if reset:
                for buf, (_, _, default) in zip(buffers, specs):
                    if buf is not None:
                        buf[row].fill_(default)
            changed = [t for t in output[old_len:] if 0 <= t < vocab]
            if buffers[0] is None:
                changed = list(dict.fromkeys(changed))
            if buffers[0] is None and not reset:
                # Without frequency penalties, only newly seen tokens need updates.
                previous_history = self._histories[id(req)][3]
                changed = [t for t in changed if t not in previous_history]
            self.history(req, vocab)
            if changed:
                ids = torch.tensor(
                    changed, dtype=torch.long, pin_memory=device.type == "cuda"
                ).to(device, non_blocking=True)
                frequency, presence, repetition = buffers
                if frequency is not None:
                    # The native update accepts only one token per request.
                    frequency[row].scatter_add_(
                        0, ids, penalizers[0].frequency_penalties[row].expand(ids.shape)
                    )
                if presence is not None:
                    presence[row].index_fill_(
                        0, ids, req.sampling_params.presence_penalty
                    )
                if repetition is not None:
                    repetition[row].index_fill_(
                        0, ids, req.sampling_params.repetition_penalty
                    )
            self._synced[id(req)] = (output, len(output))
        return buffers


def _apply_native_penalties(scores, additive, scaling):
    if (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.stride(1) == 1
        and scores.shape[0] <= 128
    ):
        from sglang.kernels.ops.sampling.clover_penalty import clover_penalty

        clover_penalty(scores, additive, scaling)
        return
    if additive is not None:
        scores.add_(additive)
    if scaling is not None:
        from sglang.srt.sampling.penaltylib.repetition_penalty import (
            apply_scaling_penalties,
        )

        apply_scaling_penalties(scores, scaling[None, :])


def validate_clover_penalties(req, *, generation_config=None):
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
        token = _default_think_end_token(tokenizer, generation_config)
        if token is None:
            return "CLOVER think-end bias requires a single dllm_think_end_token_id"
    if token is not None:
        if (
            type(token) is not int
            or token < 0
            or (req.vocab_size is not None and token >= req.vocab_size)
            or (tokenizer is not None and token >= len(tokenizer))
        ):
            return "CLOVER dllm_think_end_token_id must be a valid vocabulary ID"
        custom["dllm_think_end_token_id"] = token
    sp.custom_params = custom
    return None


def _default_think_end_token(tokenizer, generation_config):
    # The checkpoint names the id, so requests without a tokenizer still resolve it.
    if generation_config is not None:
        token = generation_config.to_dict().get("dllm_think_end_token_id")
        if token is not None:
            return token
    if tokenizer is None:
        return None
    encoded = tokenizer.encode("</think>", add_special_tokens=False)
    if len(encoded) != 1 or encoded[0] == tokenizer.unk_token_id:
        return None
    return encoded[0]


def _stop_tokens(req, vocab_size):
    tokenizer = req.tokenizer
    stops = set(req.sampling_params.stop_token_ids or ())
    stops.update(req.eos_token_ids or ())
    stops.update(getattr(tokenizer, "additional_stop_token_ids", None) or ())
    stops.add(getattr(tokenizer, "eos_token_id", None))
    return sorted(t for t in stops if t is not None and 0 <= t < vocab_size)


def apply_clover_penalties(
    logits, reqs, visible_tokens, *, state, preview=False, penalties=None
):
    """Apply penalties to [batch, width, vocab] logits in place.

    Row zero of the preview is an already committed pending token; row one
    predicts the next output. Verify row zero predicts that same next output.
    Length ramps use actual output lengths plus these aligned offsets. Output
    penalties and custom biases apply only to speculative rounds, not prefill.
    """
    _, width, vocab_size = logits.shape
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
            or length_bias
            or think_bias
        ):
            continue
        scores = logits[row]
        history = state.history(req, vocab_size)
        if sp.frequency_penalty or sp.presence_penalty or sp.repetition_penalty != 1:
            if penalties is None:
                raise ValueError(
                    "Clover output penalties require synchronized native buffers"
                )
            additive, scaling = (
                buf[row] if buf is not None else None for buf in penalties
            )
            _apply_native_penalties(scores, additive, scaling)

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
