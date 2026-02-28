"""
Benchmark: Streaming Session Inter-Turn Latency

Measures per-turn latency across three modes as context grows:
  - no_session:        re-send full context each turn (radix tree prefix match)
  - regular_session:   session append (radix tree insert + match)
  - streaming_session: session append (O(1) KV direct transfer)

Each mode runs NUM_CONCURRENT parallel sessions, each doing NUM_TURNS sequential
requests (16 input / 8 output per turn).

Usage:
    python -m pytest bench_session_latency.py -s
    python -m unittest bench_session_latency.BenchSessionLatency.test_streaming_session
    python -m unittest bench_session_latency.BenchSessionLatency
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
from tabulate import tabulate

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=100, suite="stage-b-test-large-1-gpu")

NUM_TURNS = 300
INPUT_LEN = 16
GEN_LEN = 8
NUM_CONCURRENT = 4
TAIL_TURNS = 10
SAMPLE_TURNS = 8

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
) * 200

SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": GEN_LEN,
    "no_stop_trim": True,
    "skip_special_tokens": False,
    "ignore_eos": True,
}


@dataclass
class TurnResult:
    turn: int
    context_len: int
    cached_tokens: int
    prompt_tokens: int
    completion_tokens: int
    client_latency_ms: float
    e2e_latency_ms: float


@dataclass
class ModeResult:
    mode: str
    turns: List[TurnResult] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_input_chunks(
    tokenizer, num_turns: int, input_len: int, offset: int = 0
) -> List[List[int]]:
    all_ids = tokenizer.encode(FILLER_TEXT)
    if all_ids and all_ids[0] == tokenizer.bos_token_id:
        all_ids = all_ids[1:]

    start = offset * num_turns * input_len
    needed = start + num_turns * input_len
    while len(all_ids) < needed:
        all_ids = all_ids + all_ids
    chunks = [
        all_ids[start + i * input_len : start + (i + 1) * input_len]
        for i in range(num_turns)
    ]

    if tokenizer.bos_token_id is not None:
        chunks[0] = [tokenizer.bos_token_id] + chunks[0]

    return chunks


def _send_generate(base_url: str, payload: dict) -> dict:
    resp = requests.post(base_url + "/generate", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Generate failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _record_turn(
    turn_idx: int, context_len: int, meta: dict, client_latency_ms: float
) -> TurnResult:
    return TurnResult(
        turn=turn_idx + 1,
        context_len=context_len,
        cached_tokens=meta["cached_tokens"],
        prompt_tokens=meta["prompt_tokens"],
        completion_tokens=meta["completion_tokens"],
        client_latency_ms=client_latency_ms,
        e2e_latency_ms=meta.get("e2e_latency", 0) * 1000,
    )


# ---------------------------------------------------------------------------
# Single-session runners (called by worker threads)
# ---------------------------------------------------------------------------


def _run_one_no_session(
    base_url: str, tokenizer, chunks: List[List[int]]
) -> ModeResult:
    result = ModeResult(mode="no_session")
    accumulated_ids: List[int] = []

    for turn_idx, chunk_ids in enumerate(chunks):
        accumulated_ids.extend(chunk_ids)

        t0 = time.perf_counter()
        response = _send_generate(
            base_url,
            {"input_ids": accumulated_ids, "sampling_params": SAMPLING_PARAMS},
        )
        client_lat = (time.perf_counter() - t0) * 1000

        meta = response["meta_info"]
        result.turns.append(
            _record_turn(turn_idx, len(accumulated_ids), meta, client_lat)
        )
        result.outputs.append(response["text"])

        output_ids = tokenizer.encode(response["text"])
        if output_ids and output_ids[0] == tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        accumulated_ids.extend(output_ids)

    return result


def _run_one_session(
    base_url: str, chunks: List[List[int]], streaming: bool = False
) -> ModeResult:
    mode = "streaming_session" if streaming else "regular_session"
    result = ModeResult(mode=mode)

    capacity = sum(len(c) for c in chunks) + len(chunks) * GEN_LEN + 1024
    open_payload: dict = {"capacity_of_str_len": capacity}
    if streaming:
        open_payload["streaming"] = True
    session_id = requests.post(base_url + "/open_session", json=open_payload).json()

    rid = None
    context_len = 0

    for turn_idx, chunk_ids in enumerate(chunks):
        context_len += len(chunk_ids)

        t0 = time.perf_counter()
        response = _send_generate(
            base_url,
            {
                "input_ids": chunk_ids,
                "session_params": {"id": session_id, "rid": rid},
                "sampling_params": SAMPLING_PARAMS,
            },
        )
        client_lat = (time.perf_counter() - t0) * 1000

        meta = response["meta_info"]
        rid = meta["id"]
        context_len += meta["completion_tokens"]

        result.turns.append(_record_turn(turn_idx, context_len, meta, client_lat))
        result.outputs.append(response["text"])

    requests.post(base_url + "/close_session", json={"session_id": session_id})
    return result


# ---------------------------------------------------------------------------
# Stats & reporting
# ---------------------------------------------------------------------------


def _collect_latencies(
    results: List[ModeResult], last_n: Optional[int] = None
) -> List[float]:
    lats = []
    for r in results:
        turns = r.turns[1:]  # skip turn 1
        if last_n is not None:
            turns = r.turns[-last_n:]
        lats.extend(t.client_latency_ms for t in turns)
    return lats


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_mode_table(result: ModeResult, label: str = ""):
    tag = f"{result.mode} ({label})" if label else result.mode
    print(f"\n  [{tag}]  {len(result.turns)} turns")

    n = len(result.turns)
    if n <= SAMPLE_TURNS * 2:
        indices = list(range(n))
    else:
        indices = list(range(SAMPLE_TURNS)) + [-1] + list(range(n - SAMPLE_TURNS, n))

    rows = []
    for idx in indices:
        if idx == -1:
            rows.append(["..."] * 5)
            continue
        t = result.turns[idx]
        rows.append(
            [
                t.turn,
                t.context_len,
                t.cached_tokens,
                f"{t.client_latency_ms:.1f}ms",
                f"{t.e2e_latency_ms:.1f}ms",
            ]
        )
    print(
        tabulate(
            rows,
            headers=["Turn", "Context", "Cached", "Client Lat", "E2E Lat"],
            colalign=("right",) * 5,
        )
    )


def _print_summary(all_results: Dict[str, List[ModeResult]]):
    stats = [
        (
            mode,
            _avg(_collect_latencies(rs)),
            _avg(_collect_latencies(rs, last_n=TAIL_TURNS)),
        )
        for mode, rs in all_results.items()
    ]
    base_all, base_tail = (stats[0][1] or 1.0), (stats[0][2] or 1.0)
    tail_label = f"last {TAIL_TURNS}"

    print(f"\n  SUMMARY  ({NUM_CONCURRENT} sessions x {NUM_TURNS} turns)")
    rows = [
        [
            mode,
            f"{a:.1f}ms",
            f"{t:.1f}ms",
            f"{base_all / a:.2f}x" if a else "inf",
            f"{base_tail / t:.2f}x" if t else "inf",
        ]
        for mode, a, t in stats
    ]
    print(
        tabulate(
            rows,
            headers=[
                "Mode",
                "Avg (all)",
                f"Avg ({tail_label})",
                "Speedup (all)",
                f"Speedup ({tail_label})",
            ],
            colalign=("left", "right", "right", "right", "right"),
        )
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class BenchSessionLatency(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "flashinfer",
                "--enable-streaming-session",
            ],
        )
        cls.tokenizer = get_tokenizer(cls.model)

        requests.post(cls.base_url + "/flush_cache")
        _send_generate(
            cls.base_url,
            {
                "input_ids": cls.tokenizer.encode("Hello world"),
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        )

        cls.all_results: Dict[str, List[ModeResult]] = {}

    @classmethod
    def tearDownClass(cls):
        if len(cls.all_results) > 1:
            _print_summary(cls.all_results)
        kill_process_tree(cls.process.pid)

    def _run_concurrent_no_session(self) -> List[ModeResult]:
        requests.post(self.base_url + "/flush_cache")

        def run_one(session_idx):
            chunks = _generate_input_chunks(
                self.tokenizer, NUM_TURNS, INPUT_LEN, offset=session_idx
            )
            return _run_one_no_session(self.base_url, self.tokenizer, chunks)

        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as pool:
            return list(pool.map(run_one, range(NUM_CONCURRENT)))

    def _run_concurrent_session(self, streaming: bool = False) -> List[ModeResult]:
        requests.post(self.base_url + "/flush_cache")

        def run_one(session_idx):
            chunks = _generate_input_chunks(
                self.tokenizer, NUM_TURNS, INPUT_LEN, offset=session_idx
            )
            return _run_one_session(self.base_url, chunks, streaming=streaming)

        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as pool:
            return list(pool.map(run_one, range(NUM_CONCURRENT)))

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------

    def test_no_session(self):
        results = self._run_concurrent_no_session()
        self.__class__.all_results["no_session"] = results
        _print_mode_table(results[0], label="session 0")

    def test_regular_session(self):
        results = self._run_concurrent_session(streaming=False)
        self.__class__.all_results["regular_session"] = results
        _print_mode_table(results[0], label="session 0")

    def test_streaming_session(self):
        results = self._run_concurrent_session(streaming=True)
        self.__class__.all_results["streaming_session"] = results
        _print_mode_table(results[0], label="session 0")

        reg_list = self.__class__.all_results.get("regular_session")
        if reg_list:
            reg_out = reg_list[0].outputs
            stm_out = results[0].outputs
            mismatches = sum(1 for a, b in zip(reg_out, stm_out) if a != b)
            self.assertEqual(
                mismatches,
                0,
                f"regular vs streaming (session 0): {mismatches}/{len(reg_out)} turns differ",
            )

            reg_tail = _avg(_collect_latencies(reg_list, last_n=TAIL_TURNS))
            stm_tail = _avg(_collect_latencies(results, last_n=TAIL_TURNS))
            speedup = reg_tail / stm_tail if stm_tail > 0 else float("inf")
            self.assertGreaterEqual(
                speedup,
                2.0,
                f"streaming should be >=2x faster on last {TAIL_TURNS} turns "
                f"(regular={reg_tail:.1f}ms, streaming={stm_tail:.1f}ms, speedup={speedup:.2f}x)",
            )


if __name__ == "__main__":
    unittest.main()
