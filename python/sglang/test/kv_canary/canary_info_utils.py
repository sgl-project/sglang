from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import requests

from sglang.kernels.ops.kv_canary.consts import (
    canary_chain_extend_hash,
    canary_chain_initial_hash,
    canary_info,
)
from sglang.srt.kv_canary.canary_info.checker import (
    CanaryInfoChecker,
    CanaryInfoMismatch,
)
from sglang.srt.kv_canary.canary_info.digest import sampling_digest
from sglang.srt.kv_canary.canary_info.tokenizer_manager_hook import _check_canary_infos
from sglang.srt.sampling.sampling_params import SamplingParams

CANARY_INFO_SAMPLING_PARAMS: tuple[dict[str, Any], ...] = (
    {"temperature": 0.0},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 1.0, "top_k": 20, "min_p": 0.05},
    {"temperature": 0.8, "sampling_seed": 1234},
)


def make_canary_info_prompt_ids(*, index: int, length: int) -> list[int]:
    return [1000 + ((index * 7919 + j * 31) % 20000) for j in range(length)]


def make_canary_info_payload(
    *, index: int, prompt_length: int, max_new_tokens: int
) -> dict[str, Any]:
    return {
        "input_ids": make_canary_info_prompt_ids(index=index, length=prompt_length),
        "sampling_params": {
            **CANARY_INFO_SAMPLING_PARAMS[index % len(CANARY_INFO_SAMPLING_PARAMS)],
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }


def reference_canary_infos(
    *,
    prompt: list[int],
    outputs: list[int],
    params: SamplingParams,
) -> list[int]:
    sampling = sampling_digest(params)
    digest = canary_chain_initial_hash()
    for position, token in enumerate(prompt):
        digest = canary_chain_extend_hash(digest, token, position)
    values = []
    for offset, token in enumerate(outputs):
        values.append(canary_info(digest, token, sampling))
        digest = canary_chain_extend_hash(digest, token, len(prompt) + offset)
    return values


def recheck_canary_infos(
    *,
    rid: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
    output_ids: list[int],
    canary_infos: Optional[list[int]],
) -> Optional[CanaryInfoMismatch]:
    checker = CanaryInfoChecker(
        rid=rid,
        sampling_params=SamplingParams(**sampling_params),
        prompt_ids=input_ids,
    )
    return _check_canary_infos(
        checker=checker, token_ids=output_ids, canary_infos=canary_infos
    )


def post_parallel_payloads(
    *, url: str, payloads: list[dict[str, Any]], timeout: float
) -> list[dict]:
    def _send(payload: dict[str, Any]) -> dict:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            return {"status_code": -1, "error": repr(exc)}
        try:
            if payload.get("stream") and resp.status_code == 200:
                chunks = [
                    json.loads(line[6:])
                    for line in resp.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                assert chunks, "Streaming response did not contain any output"
                return {"status_code": resp.status_code, "json": chunks[-1]}
            return {"status_code": resp.status_code, "json": resp.json()}
        except ValueError:
            return {"status_code": resp.status_code, "error": resp.text}

    with ThreadPoolExecutor(max_workers=max(1, len(payloads))) as pool:
        return list(pool.map(_send, payloads))
