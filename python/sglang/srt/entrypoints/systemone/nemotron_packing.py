# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Shared-state scoring selected by Nemotron Decision checkpoint settings."""

import math

from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult
from sglang.srt.runtime_context import get_schedule


def pack_decision_prompts(prompts, max_tokens, shared_prefix):
    if not prompts:
        raise ValueError("At least one question prompt is required")
    if any(not prompt for prompt in prompts):
        raise ValueError("Question prompts must not be empty")
    if any(len(prompt) > max_tokens for prompt in prompts):
        raise ValueError(
            "Each question’s full prompt, including the state and answer options, must fit within the input token limit"
        )
    shared = len(shared_prefix)
    if not shared or any(
        prompt[:shared] != shared_prefix or len(prompt) <= shared for prompt in prompts
    ):
        raise ValueError(
            "packed Decision prompts must share the exact state-end boundary"
        )
    groups, ids, indices, lengths = [], list(prompts[0][:shared]), [], []
    for i, prompt in enumerate(prompts):
        tail = prompt[shared:]
        if indices and len(ids) + len(tail) > max_tokens:
            groups.append((ids, [shared, *lengths], indices))
            ids, indices, lengths = list(prompt[:shared]), [], []
        ids.extend(tail)
        lengths.append(len(tail))
        indices.append(i)
    groups.append((ids, [shared, *lengths], indices))
    return groups


async def score_packed_decisions(
    manager, encoder, prompts, label_ids, request, shared_prefix, lora_path=None
):
    limit = min(
        encoder.max_prompt_tokens,
        max_joint_prompt_tokens(
            context_len=manager.context_len,
            num_reserved_tokens=manager.num_reserved_tokens,
            max_req_input_len=manager.max_req_input_len,
            max_prefill_tokens=get_schedule().max_prefill_tokens,
        ),
    )
    scores, logprobs, prompt_tokens = [None] * len(prompts), [None] * len(prompts), 0
    for ids, layout, indices in pack_decision_prompts(prompts, limit, shared_prefix):
        labels = sorted({token for i in indices for token in label_ids[i]})
        end, slots = layout[0], []
        for length in layout[1:]:
            end += length
            slots.append(end - 1)
        batch = GenerateReqInput(
            input_ids=ids,
            decision_layout=layout,
            token_indices_to_pool=slots,
            lora_path=lora_path,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=0,
            token_ids_logprob=labels,
        )
        response = await manager.generate_request(batch, request).__anext__()
        meta = response["meta_info"]
        raw_rows = meta["input_token_ids_logprobs"]
        if len(raw_rows) != len(indices):
            raise RuntimeError("packed Decision did not return every mask readout")
        rows = []
        for i, raw in zip(indices, raw_rows):
            values = {token: value for value, token, *_ in raw}
            rows.append([values[token] for token in label_ids[i]])
        prompt_tokens += meta["prompt_tokens"]
        for i, row in zip(indices, rows):
            maximum = max(row)
            weights = [math.exp(value - maximum) for value in row]
            total = math.fsum(weights)
            scores[i] = [weight / total for weight in weights]
            logprobs[i] = row
    return ScoreResult(
        scores=scores, token_logprobs=logprobs, prompt_tokens=prompt_tokens
    )
