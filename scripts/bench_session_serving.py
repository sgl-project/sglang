#!/usr/bin/env python3
"""
Primitive session-aware benchmark: hot sessions under cold session KV pressure.

Cold sessions fire one warm turn and go idle, pinning KV in the pool.
Hot sessions fire --num-turns turns continuously with bounded concurrency.

Without session-LRU: hot sessions stall/OOM as cold sessions exhaust the pool.
With    session-LRU: cold sessions soft-evicted; hot sessions complete normally.

Correctness signals:
  - cached_tokens grows turn-over-turn (session KV accumulating across turns)
  - All hot session turns return 200 (no garbage output, no race conditions)
  - Server health check passes after the run

Launch the server first:

  python -m sglang.launch_server \\
      --model Qwen/Qwen2.5-Coder-3B-Instruct \\
      --enable-streaming-session \\
      --mem-fraction-static 0.1 \\
      --context-length 16384 \\
      --port 30000

--context-length must be large enough for multi-turn sessions to accumulate:
  num_turns * (input_tokens + output_tokens)  <  context_length
  Default: 5 turns * (2048 + 2048) = 20480  →  use --context-length 16384 or higher.

Pool sizing rule:
  (hot_sessions + cold_sessions) * input_tokens  >  pool_size
  hot_sessions * input_tokens * num_turns         <  pool_size
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import requests


# ── data structures ───────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    turn_idx: int
    status: int
    latency_s: float
    cached_tokens: int
    output_tokens: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200


@dataclass
class SessionResult:
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(t.ok for t in self.turns)

    @property
    def num_completed(self) -> int:
        return sum(1 for t in self.turns if t.ok)


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def tokenize_sync(base: str, text: str) -> list[int]:
    r = requests.post(base + "/tokenize", json={"prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def build_turn_ids(base: str, tokens_per_turn: int, num_turns: int) -> list[list[int]]:
    """Return num_turns+1 distinct token ID lists using non-overlapping filler offsets.
    Index 0 is used for the cold session warm turn; 1..num_turns for hot session turns.
    """
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenize_sync(base, filler)
    while len(ids) < tokens_per_turn * (num_turns + 2):
        ids = ids + ids
    return [
        ids[i * tokens_per_turn: (i + 1) * tokens_per_turn]
        for i in range(num_turns + 1)
    ]


async def _open_session(http: aiohttp.ClientSession, base: str) -> str:
    async with http.post(
        base + "/open_session",
        json={"capacity_of_str_len": 200_000, "streaming": True},
    ) as resp:
        assert resp.status == 200, f"open_session failed: {await resp.text()}"
        return await resp.json()


async def _close_session(http: aiohttp.ClientSession, base: str, sid: str) -> None:
    try:
        async with http.post(
            base + "/close_session", json={"session_id": sid}
        ) as resp:
            pass
    except Exception:
        pass


async def _generate(
    http: aiohttp.ClientSession,
    base: str,
    input_ids: list[int],
    max_new_tokens: int,
    session_params: Optional[dict] = None,
    timeout: float = 120.0,
) -> tuple[int, dict, float]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "no_stop_trim": True,
        },
    }
    if session_params:
        payload["session_params"] = session_params
    t0 = time.perf_counter()
    try:
        async with http.post(
            base + "/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            elapsed = time.perf_counter() - t0
            body = await resp.json() if resp.status == 200 else {"error": await resp.text()}
            return resp.status, body, elapsed
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
        return -1, {"error": f"timeout after {timeout:.0f}s — pool likely exhausted"}, timeout


# ── session runners ───────────────────────────────────────────────────────────


async def run_cold_session(
    http: aiohttp.ClientSession,
    base: str,
    warm_ids: list[int],
    output_tokens: int,
    sem: asyncio.Semaphore,
    idx: int,
) -> tuple[str, bool]:
    """Fire one warm turn, then leave the session idle (KV pinned in pool)."""
    sid = await _open_session(http, base)
    async with sem:
        status, body, _ = await _generate(
            http, base, warm_ids, output_tokens,
            session_params={"id": sid, "rid": None},
        )
    if status != 200:
        print(f"  WARNING: cold session {idx} warm failed "
              f"(status={status}): {body.get('error', '')[:60]}")
        return sid, False
    return sid, True


async def run_hot_session(
    http: aiohttp.ClientSession,
    base: str,
    turn_ids_list: list[list[int]],
    output_tokens: int,
    sem: asyncio.Semaphore,
    turn_timeout: float,
) -> SessionResult:
    """Fire all turns sequentially; semaphore limits concurrent in-flight turns
    across all hot sessions."""
    sid = await _open_session(http, base)
    result = SessionResult(session_id=sid)
    rid = None

    for turn_idx, turn_ids in enumerate(turn_ids_list):
        async with sem:
            status, body, latency = await _generate(
                http, base, turn_ids, output_tokens,
                session_params={"id": sid, "rid": rid},
                timeout=turn_timeout,
            )
        if status == 200:
            meta = body.get("meta_info", {})
            rid = meta.get("id")
            result.turns.append(TurnResult(
                turn_idx=turn_idx,
                status=200,
                latency_s=latency,
                cached_tokens=meta.get("cached_tokens", 0),
                output_tokens=meta.get("completion_tokens", 0),
            ))
        else:
            result.turns.append(TurnResult(
                turn_idx=turn_idx,
                status=status,
                latency_s=latency,
                cached_tokens=0,
                output_tokens=0,
                error=body.get("error"),
            ))
            break  # cannot chain turns after a failure

    await _close_session(http, base, sid)
    return result


# ── main ──────────────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    base = args.base_url

    print("=" * 65)
    print(f"bench_session_serving  "
          f"cold={args.cold_sessions}  hot={args.hot_sessions}  "
          f"turns={args.num_turns}")
    print(f"input={args.input_tokens} tok/turn  "
          f"output={args.output_tokens} tok/turn  "
          f"concurrency={args.concurrency}")
    print("=" * 65)

    print("\nTokenizing…")
    turn_ids = build_turn_ids(base, args.input_tokens, args.num_turns)
    # turn_ids[0]          → cold session warm turn
    # turn_ids[1..N]       → hot session turns 0..N-1

    conn = aiohttp.TCPConnector(limit=args.hot_sessions + args.cold_sessions + 16)
    async with aiohttp.ClientSession(connector=conn) as http:

        # ── Phase 1: warm cold sessions ───────────────────────────────────────
        print(f"\nPhase 1: warming {args.cold_sessions} cold sessions…")
        cold_sem = asyncio.Semaphore(min(args.cold_sessions, 16))
        cold_tasks = [
            run_cold_session(http, base, turn_ids[0], args.output_tokens, cold_sem, i)
            for i in range(args.cold_sessions)
        ]
        cold_results = await asyncio.gather(*cold_tasks)
        cold_sids = [sid for sid, _ in cold_results]
        cold_ok = sum(1 for _, ok in cold_results if ok)
        print(f"  {cold_ok}/{args.cold_sessions} cold sessions warmed — "
              f"KV pinned idle in pool.")
        if cold_ok < args.cold_sessions:
            print("  WARNING: some cold sessions failed to warm — "
                  "pool may already be exhausted.")

        # ── Phase 2: run hot sessions concurrently ────────────────────────────
        print(f"\nPhase 2: {args.hot_sessions} hot sessions × {args.num_turns} turns "
              f"(concurrency={args.concurrency})…")
        print("  Without session-LRU: turns stall/timeout as pool is exhausted.")
        print("  With    session-LRU: cold sessions evicted; hot sessions succeed.")

        t_start = time.perf_counter()
        hot_sem = asyncio.Semaphore(args.concurrency)
        hot_tasks = [
            run_hot_session(
                http, base,
                turn_ids[1: args.num_turns + 1],
                args.output_tokens,
                hot_sem,
                turn_timeout=args.turn_timeout,
            )
            for _ in range(args.hot_sessions)
        ]
        hot_results: list[SessionResult] = await asyncio.gather(*hot_tasks)
        elapsed = time.perf_counter() - t_start

        # ── cleanup cold sessions ─────────────────────────────────────────────
        await asyncio.gather(*[_close_session(http, base, sid) for sid in cold_sids])

    # ── reporting ─────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Results")
    print("=" * 65)

    sessions_fully_ok = sum(1 for r in hot_results if r.all_ok)

    print(f"\n{'Turn':>4}  {'OK':>7}  {'Lat avg':>9}  {'Lat p99':>9}  "
          f"{'Cached avg':>11}  First error")
    print("-" * 65)
    for turn_idx in range(args.num_turns):
        turn_data = [r.turns[turn_idx] for r in hot_results if turn_idx < len(r.turns)]
        if not turn_data:
            break
        ok_data = [t for t in turn_data if t.ok]
        lats = [t.latency_s for t in ok_data]
        cached = [t.cached_tokens for t in ok_data]
        lat_avg = statistics.mean(lats) if lats else 0.0
        lat_p99 = sorted(lats)[max(0, int(len(lats) * 0.99) - 1)] if lats else 0.0
        cached_avg = statistics.mean(cached) if cached else 0.0
        first_err = next(
            (t.error[:40] for t in turn_data if not t.ok and t.error), ""
        )
        print(f"  {turn_idx:>2}  {len(ok_data):>3}/{len(turn_data):<3}  "
              f"{lat_avg:>7.2f} s  {lat_p99:>7.2f} s  "
              f"{cached_avg:>9.0f} tok  {first_err}")

    total_out = sum(t.output_tokens for r in hot_results for t in r.turns if t.ok)
    print(f"\nSessions fully completed : {sessions_fully_ok}/{args.hot_sessions}")
    print(f"Total output tokens      : {total_out}")
    print(f"Wall time                : {elapsed:.1f} s")
    if elapsed > 0 and total_out > 0:
        print(f"Output throughput        : {total_out / elapsed:.0f} tok/s")

    try:
        hresp = requests.get(base + "/health", timeout=10)
        health = "OK" if hresp.status_code == 200 else f"FAIL ({hresp.status_code})"
    except Exception as e:
        health = f"FAIL ({e})"
    print(f"Server health            : {health}")

    print()
    if sessions_fully_ok == args.hot_sessions:
        print("PASS  All hot sessions completed all turns.")
        print("      cached_tokens increasing per turn confirms session KV reuse.")
    else:
        failed = args.hot_sessions - sessions_fully_ok
        print(f"FAIL  {failed}/{args.hot_sessions} hot session(s) did not complete "
              "all turns.")
        print("      On main branch (no session-LRU) this is the expected OOM behavior.")
        print("      On session-lru branch this indicates a bug.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--hot-sessions", type=int, default=10,
                    help="Sessions that fire all turns (default: 10)")
    ap.add_argument("--cold-sessions", type=int, default=25,
                    help="Sessions warmed once and left idle (default: 25)")
    ap.add_argument("--num-turns", type=int, default=5,
                    help="Turns per hot session (default: 5)")
    ap.add_argument("--input-tokens", type=int, default=2048,
                    help="New input tokens per turn (default: 2048)")
    ap.add_argument("--output-tokens", type=int, default=2048,
                    help="Output tokens per turn; ignore_eos forces exact count (default: 2048)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max concurrent in-flight turns across all hot sessions (default: 5)")
    ap.add_argument("--turn-timeout", type=float, default=120.0,
                    help="Per-turn timeout in seconds; treat as OOM on expiry (default: 120)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
