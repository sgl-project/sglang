"""
Manual A/B throughput benchmark for the streaming-session scheduler-CPU path.

Not a CI gate. The per-step scheduler work this stresses is CPU-bound, and CI
H100 hosts differ by ~12% in CPU throughput -- comparable to the ~10% the
optimization buys (e.g. #27965, in-place fill_ids reconstruction) -- so an
absolute throughput floor cannot separate before/after across the runner pool.
Instead, when changing the streaming-session / scheduler hot path, run this on
ONE fixed machine before and after your change and compare the printed throughput.

The model is truncated to a single layer (--json-model-override-args) with overlap
scheduling disabled, so the GPU forward shrinks and the O(resident context)
scheduler work (batch assembly, prefix match, KV bookkeeping) dominates. On a
dedicated machine the run-to-run spread is ~0.2-2%, so a few % delta is real.
"""

import concurrent.futures
import json
import random
import time
import unittest
from dataclasses import dataclass
from typing import Optional

import requests

from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)

NUM_HIDDEN_LAYERS = 1
NUM_CONCURRENT = 16
CONTEXT_LEN = 30000
NUM_TURNS = 100


@dataclass
class _Session:
    session_id: str
    rid: Optional[str]


def _synthetic_input_ids(
    length: int, seed: int, token_id_start: int, token_id_count: int
) -> list[int]:
    return [token_id_start + ((seed + i) % token_id_count) for i in range(length)]


def _stream_generate(
    base_url: str, input_ids: list[int], session: _Session, output_len: int
) -> int:
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
            "session_params": {"id": session.session_id, "rid": session.rid},
        },
        stream=True,
    )
    completion_tokens = 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        meta = json.loads(data)["meta_info"]
        completion_tokens = int(meta["completion_tokens"])
        session.rid = meta["id"]
    return completion_tokens


def bench_serving_streaming(
    base_url: str,
    *,
    num_sessions: int,
    context_len: int,
    num_turns: int,
    input_len: int = 10,
    min_gen_len: int = 1,
    max_gen_len: int = 16,
    token_id_start: int = 1000,
    token_id_count: int = 1024,
    warmup: bool = True,
) -> dict:
    def open_and_prime(session_index: int) -> _Session:
        session_id = requests.post(
            base_url + "/open_session",
            json={"capacity_of_str_len": 0, "streaming": True},
        ).json()
        session = _Session(session_id=session_id, rid=None)
        prime_ids = _synthetic_input_ids(
            context_len, session_index, token_id_start, token_id_count
        )
        _stream_generate(base_url, prime_ids, session, output_len=1)
        return session

    def run_turns(session: _Session, session_index: int) -> int:
        rng = random.Random(session_index)
        output_tokens = 0
        for turn_index in range(num_turns):
            output_len = rng.randint(min_gen_len, max_gen_len)
            input_ids = _synthetic_input_ids(
                input_len,
                session_index * num_turns + turn_index,
                token_id_start,
                token_id_count,
            )
            output_tokens += _stream_generate(base_url, input_ids, session, output_len)
        return output_tokens

    def measure() -> dict:
        requests.post(base_url + "/flush_cache")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as pool:
            sessions = list(pool.map(open_and_prime, range(num_sessions)))
            start = time.perf_counter()
            output_tokens = sum(
                pool.map(
                    lambda args: run_turns(*args),
                    [(session, idx) for idx, session in enumerate(sessions)],
                )
            )
            duration = time.perf_counter() - start
        for session in sessions:
            requests.post(
                base_url + "/close_session", json={"session_id": session.session_id}
            )
        return {
            "output_throughput": output_tokens / duration,
            "total_output_tokens": output_tokens,
            "duration_s": duration,
        }

    if warmup:
        measure()
    return measure()


class TestStreamingSessionThroughput(StreamingSessionServerBase):
    model = "Qwen/Qwen3-0.6B"
    extra_args = [
        "--json-model-override-args",
        f'{{"num_hidden_layers": {NUM_HIDDEN_LAYERS}}}',
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "8192",
        "--schedule-policy",
        "fcfs",
        "--max-running-requests",
        "100",
        "--disable-overlap-schedule",
    ]

    def test_streaming_session_throughput(self):
        res = bench_serving_streaming(
            self.base_url,
            num_sessions=NUM_CONCURRENT,
            context_len=CONTEXT_LEN,
            num_turns=NUM_TURNS,
        )
        print(
            f"\n[streaming-session throughput] sessions={NUM_CONCURRENT} "
            f"context={CONTEXT_LEN} turns={NUM_TURNS} layers={NUM_HIDDEN_LAYERS}\n"
            f"  throughput={res['output_throughput']:.1f} tok/s "
            f"duration={res['duration_s']:.3f}s tokens={res['total_output_tokens']}\n"
            f"  (no threshold; compare before/after on the same machine)"
        )


if __name__ == "__main__":
    unittest.main()
