from typing import List, NamedTuple, Optional

import requests


class ReferenceCapture(NamedTuple):

    text: str
    pieces: List[str]
    top2_gaps: List[float]


class Divergence(NamedTuple):

    token_index: int
    top2_gap: float


def greedy_request(url: str, prompt: str, max_new_tokens: int) -> str:
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    )
    resp.raise_for_status()
    return resp.json()["text"]


def capture_reference(url: str, prompt: str, max_new_tokens: int) -> ReferenceCapture:
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "top_logprobs_num": 2,
            "return_text_in_logprobs": True,
        },
    )
    resp.raise_for_status()
    meta = resp.json()["meta_info"]

    pieces = [entry[2] for entry in meta["output_token_logprobs"]]
    top2_gaps: List[float] = []
    for entry in meta["output_top_logprobs"]:
        if len(entry) >= 2:
            top2_gaps.append(abs(entry[0][0] - entry[1][0]))
        else:
            top2_gaps.append(float("inf"))

    return ReferenceCapture(
        text=resp.json()["text"], pieces=pieces, top2_gaps=top2_gaps
    )


def first_token_divergence(
    reference: ReferenceCapture, spec_text: str
) -> Optional[Divergence]:
    accumulated = ""
    for index, piece in enumerate(reference.pieces):
        candidate = accumulated + piece
        if spec_text.startswith(candidate):
            accumulated = candidate
            continue
        if candidate.startswith(spec_text):
            return None
        gap = (
            reference.top2_gaps[index]
            if index < len(reference.top2_gaps)
            else float("inf")
        )
        return Divergence(token_index=index, top2_gap=gap)
    return None
