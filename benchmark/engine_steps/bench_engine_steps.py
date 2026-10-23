"""What one prefill or decode step costs at a fixed, steady batch.

Each DP rank warms a long shared prefix; every request adds a random, uncached
--new-tokens suffix and is pinned to its rank. Prefill sends one-token requests,
--tokens-per-step new tokens per rank per step; decode streams --batch-per-rank
requests per rank and times --measure-steps steps.

Throughput counts whole steps only: the window starts and ends at step boundaries
(the scheduler's stamps under --enable-metrics, else response arrival gaps).
--server-log cross-checks the engine's per-step log lines over the same window; it
needs the server on this host's clock, with SGLANG_LOG_MS=1. A requested check
fails unless every rank has the fixed batch size and at least 80% coverage.
The GPU denominator comes from the server's TP/PP/DP topology, unless overridden
with --num-gpus. Local package/GPU/environment metadata belongs to the client
host in HTTP mode; server configuration is recorded separately.

  python bench_engine_steps.py prefill --base-url URL --dp 4 --tokens-per-step 8192
  python bench_engine_steps.py decode --base-url URL --dp 8 --batch-per-rank 64
  python bench_engine_steps.py prefill --engine --dp 4 --model-path M --dp-size 4 ...
"""

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import distributions
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(\.\d+)?\S*(?: DP(\d+))?[^\]]*\] "
_RATE = r"throughput \(token/s\): ([\d.]+)"
_PREFILL = r"Prefill batch.*?#new-token: (\d+)().*?input "
_DECODE = r"Decode batch, #running-req: (\d+).*?(?:accept len: ([\d.]+).*?)?gen "
LINES = {
    "prefill": re.compile(_TS + _PREFILL + _RATE),
    "decode": re.compile(_TS + _DECODE + _RATE),
}


def step_groups(times, size):
    """(end, count) groups of times, about `size` to a step: any gap over half the
    median of the len / size largest gaps splits, so a stall cannot hide a boundary."""
    t = sorted(times)
    gaps = [b - a for a, b in zip(t, t[1:])]
    if not gaps:
        return [(t[-1], len(t))] if t else []
    top = sorted(gaps)[-max(1, len(t) // size - 1) :]
    split, groups, start = 0.5 * top[len(top) // 2], [], 0
    for i, gap in enumerate(gaps):
        if gap > split:
            groups.append((t[i], i + 1 - start))
            start = i + 1
    return groups + [(t[-1], len(t) - start)]


def group_window(groups, lo, hi):
    """(t0, t1, items, ms between group ends) of the groups that end between the lo
    and hi fractions of all items."""
    if not groups or not 0 <= lo < hi <= 1:
        raise SystemExit("need step groups and 0 <= --lo < --hi <= 1")
    total, cum, ends = sum(c for _, c in groups), 0, []
    for end, c in groups:
        cum += c
        ends.append((end, cum))
    i = next(k for k, (_, c) in enumerate(ends) if c >= lo * total)
    j = max((k for k, (_, c) in enumerate(ends) if c <= hi * total), default=-1)
    if j <= i:
        raise SystemExit(f"no whole step between the {lo} and {hi} fractions")
    gaps = [(b[0] - a[0]) * 1000 for a, b in zip(ends[i:j], ends[i + 1 : j + 1])]
    return ends[i][0], ends[j][0], ends[j][1] - ends[i][1], gaps


def log_steps(lines, mode, t0, t1, size):
    """Engine lines whose interval since the rank's previous line lies inside [t0, t1]
    (epoch seconds), reduced to ms per step per rank at batch `size`."""
    per_rank, spans, prev, accept, seen = {}, {}, {}, [], set()
    for m in filter(None, map(LINES[mode].search, lines)):
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        ts, step = ts + float(m.group(2) or 0), 0.001 if m.group(2) else 1.0
        rank = int(m.group(3) or 0)
        start, prev[rank] = prev.get(rank, 0.0), ts
        if start < t0 or ts + step > t1:  # stamps are truncated to `step`
            continue
        seen.add(int(m.group(4)))
        if int(m.group(4)) == size:
            accept_len = float(m.group(5) or 1.0)
            ms = size * accept_len / float(m.group(6)) * 1000
            per_rank.setdefault(rank, []).append(ms)
            lo, hi = spans.get(rank, (start, ts + step))
            spans[rank] = min(lo, start), max(hi, ts + step)
            accept += [accept_len] if m.group(5) else []
    check = {
        "ms_per_step_per_rank": {r: statistics.mean(v) for r, v in per_rank.items()}
    }
    check["coverage"] = min(((b - a) / (t1 - t0) for a, b in spans.values()), default=0)
    check["sizes_seen"] = sorted(seen)
    if accept:
        check["accept_len"] = statistics.mean(accept)
    return check


def gpu_count(info, dp, explicit=None):
    """Attention DP partitions TP; ordinary DP replicates the TP/PP group."""
    replicas = info.get("dp_size", 1)
    if "attn_dp_size" in info:
        ranks = replicas * info["attn_dp_size"]
    elif info.get("enable_dp_attention"):
        ranks, replicas = replicas, 1  # legacy attention-DP server arguments
    else:
        ranks = replicas
    if "dp_size" in info and dp != ranks:
        raise SystemExit(f"--dp {dp} does not match the server's {ranks} ranks")
    if explicit is not None:
        return explicit
    if "tp_size" not in info or "dp_size" not in info:
        raise SystemExit("server topology is unavailable; pass --num-gpus explicitly")
    return info["tp_size"] * info.get("pp_size", 1) * replicas


def check_cache(meta, new_tokens):
    new = meta["prompt_tokens"] - meta["cached_tokens"]
    if new != new_tokens:
        raise RuntimeError(
            f"expected {new_tokens} uncached suffix tokens, got {new}: "
            "the shared prefix was evicted/missed or part of a suffix was cached"
        )
    return new


def check_logs(check, result, size, dp):
    """A requested cross-check must cover every rank and the fixed batch."""
    per_rank = check["ms_per_step_per_rank"]
    errors = []
    if set(per_rank) != set(range(dp)):
        errors.append(f"expected ranks {list(range(dp))}, got {sorted(per_rank)}")
    if set(check["sizes_seen"]) != {size}:
        errors.append(f"expected batch size {size}, got {check['sizes_seen']}")
    if check["coverage"] < 0.8:
        errors.append("engine lines cover less than 80% of the window")
    if per_rank:
        check["ms_per_step"] = statistics.mean(per_rank.values())
        check["diff_pct"] = (result["ms_per_step"] / check["ms_per_step"] - 1) * 100
        if abs(check["diff_pct"]) > 0.5:
            errors.append("engine log disagrees by more than 0.5%")
    check.update(agree=not errors, errors=errors)
    return check


class EngineTransport:
    def __init__(self, server_args):
        import sglang as sgl

        self.engine = sgl.Engine(server_args=server_args)
        self.run = self.engine.loop.run_until_complete

    async def generate(self, ids, params, rank, stream):
        out = await self.engine.async_generate(
            input_ids=ids, sampling_params=params, routed_dp_rank=rank, stream=stream
        )
        if not stream:
            yield time.time(), out["meta_info"]
            return
        async for chunk in out:
            yield time.time(), chunk["meta_info"]

    async def start_profile(self, body):
        from sglang.srt.managers.io_struct import ProfileReq

        await self.engine.tokenizer_manager.start_profile(ProfileReq(**body))

    def server_info(self):
        return self.engine.get_server_info()

    def flush_cache(self):
        ret = self.engine.flush_cache()
        return None if ret.success else ret.message or "flush failed"

    def finish(self):
        try:
            self.engine.tokenizer_manager.abort_request(
                abort_all=True
            )  # decode leftovers
        finally:
            self.engine.shutdown()


class HttpTransport:
    def __init__(self, url, timeout):
        import aiohttp
        import orjson  # a 128K-id body: json.dumps would starve the server

        self.url, self.session, self.dumps = url.rstrip("/"), None, orjson.dumps
        self.timeout = timeout
        self.new_session = lambda: aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=timeout),
            connector=aiohttp.TCPConnector(limit=0),
        )
        self.loop = asyncio.new_event_loop()
        self.run = self.loop.run_until_complete

    def post(self, path, body):
        self.session = self.session or self.new_session()
        headers = {"Content-Type": "application/json"}
        return self.session.post(
            self.url + path, data=self.dumps(body), headers=headers
        )

    async def generate(self, ids, params, rank, stream):
        body = dict(input_ids=ids, sampling_params=params, routed_dp_rank=rank)
        async with self.post("/generate", {**body, "stream": stream}) as resp:
            resp.raise_for_status()
            if not stream:
                yield time.time(), (await resp.json())["meta_info"]
                return
            async for raw in resp.content:
                line = raw.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    yield time.time(), json.loads(line[5:])["meta_info"]

    async def start_profile(self, body):
        async with self.post("/start_profile", body) as resp:
            resp.raise_for_status()

    def server_info(self):
        with urlopen(self.url + "/server_info", timeout=self.timeout) as response:
            return json.load(response)

    def flush_cache(self):
        try:
            with urlopen(
                Request(self.url + "/flush_cache", method="POST"), timeout=self.timeout
            ):
                pass
        except HTTPError as e:
            return e.read().decode()

    def finish(self):
        try:
            if self.session is not None:
                self.run(self.session.close())
        finally:
            self.loop.close()


def nvidia_smi(*args):
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def environment():
    pkgs = {"sglang", "sglang-kernel", "sgl-deep-gemm", "sgl-flash-mla", "torch"}
    vers = {
        d.metadata["Name"].lower().replace("_", "-"): d.version for d in distributions()
    }
    gpus = nvidia_smi("--query-gpu=index,name,uuid,driver_version", "--format=csv")
    env = {k: v for k, v in os.environ.items() if k.startswith("SGLANG_")}
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return dict(
        versions={p: vers.get(p) for p in sorted(pkgs)},
        gpus=gpus[1:],
        visible_devices=devices,
        sglang_env=env,
    )


async def gather_requests(coros):
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def batch(args, tp, prompt, n_each, params, done, *, check_prefix=False):
    """n_each requests per rank, about eight steps' worth in flight on each."""
    in_flight = max(1, args.tokens_per_step // args.new_tokens) * 8 + 1

    async def rank_worker(r):
        sem = asyncio.Semaphore(in_flight)

        async def collect():
            async for t, meta in tp.generate(prompt(r), params, r, False):
                new = meta["prompt_tokens"] - meta["cached_tokens"]
                if check_prefix:
                    check_cache(meta, args.new_tokens)
                done.append((t, meta.get("prefill_finished_time"), new))

        async def one():
            async with sem:
                await asyncio.wait_for(collect(), timeout=args.timeout)

        await gather_requests(one() for _ in range(n_each))

    await gather_requests(rank_worker(r) for r in range(args.dp))


async def prefill(args, tp, prompt):
    nc, B = args.new_tokens, args.tokens_per_step
    per_step = max(1, B // nc)
    params = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
    await batch(
        args, tp, prompt, args.warmup_steps * per_step, params, [], check_prefix=True
    )
    done = []
    await batch(args, tp, prompt, args.requests, params, done, check_prefix=True)
    stamp = "scheduler" if all(s for _, s, _ in done) else "arrival"
    times = [s if stamp == "scheduler" else t for t, s, _ in done]
    t0, t1, _, group_ms = group_window(
        step_groups(times, per_step * args.dp), args.lo, args.hi
    )
    tokens = sum(n for t, (*_, n) in zip(times, done) if t0 < t <= t1)
    steps = tokens / (B * args.dp)
    if steps < 3:
        raise SystemExit("the window holds under three steps; raise --requests")
    if len(group_ms) > 1.05 * steps + 1:  # a short step finishes fewer requests
        raise SystemExit(
            f"{len(group_ms)} steps for {steps:.1f} full ones: the queue ran dry"
        )
    if args.profile_steps:
        await tp.start_profile(args.profile)
        await batch(args, tp, prompt, (args.profile_steps + 4) * per_step, params, [])
    return dict(
        steps=steps,
        seconds=t1 - t0,
        tokens=tokens,
        wall=(t0, t1),
        group_ms=group_ms,
        size=B,
        stamp=stamp,
    )


async def decode(args, tp, prompt):
    n, events, first = args.batch_per_rank, [], set()
    params = {"max_new_tokens": args.output_len, "temperature": 0.0, "ignore_eos": True}

    async def one(r, k):
        seen = 0
        async for t, meta in tp.generate(prompt(r), params, r, True):
            if (r, k) not in first:
                check_cache(meta, args.new_tokens)
            count = meta["completion_tokens"]
            if count < seen:
                raise RuntimeError("stream completion_tokens decreased")
            if count > seen:
                events.append((t, count - seen))
            seen = count
            first.add((r, k))

    async def wait(cond, why):
        deadline = time.time() + args.timeout
        while not cond():
            if ended := next((task for task in tasks if task.done()), None):
                exc = ended.exception()  # None: it finished
                raise SystemExit(f"a request ended early ({exc!r}); see --output-len")
            if time.time() > deadline:
                raise SystemExit(f"timed out after {args.timeout:.0f} s: " + why())
            await asyncio.sleep(0.1)

    def started():
        per_rank = [sum(1 for rank, _ in first if rank == r) for r in range(args.dp)]
        return (
            f"requests started per rank {per_rank} of {n}; the server may admit "
            "fewer (see --max-running-requests over the DP size)"
        )

    def steps_after(t_all):
        return len(step_groups([e[0] for e in events if e[0] > t_all], n * args.dp))

    tasks = [asyncio.ensure_future(one(r, k)) for r in range(args.dp) for k in range(n)]
    try:
        await wait(lambda: len(first) == n * args.dp, started)  # the full batch runs
        t_all, need = time.time(), args.warmup_steps + args.measure_steps + 2
        await wait(
            lambda: steps_after(t_all) >= need,
            lambda: f"{steps_after(t_all)} of {need} steps",
        )
        if args.profile_steps:
            await tp.start_profile(args.profile)
            await asyncio.sleep(2 + args.profile_steps * 0.1)
    finally:  # also on failure, so no stream outlives the event loop
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    groups = step_groups([e[0] for e in events if e[0] > t_all], n * args.dp)
    window = [end for end, _ in groups[args.warmup_steps :][: args.measure_steps + 1]]
    t0, t1 = window[0], window[-1]
    tokens = sum(k for t, k in events if t0 < t <= t1)
    return dict(
        steps=args.measure_steps,
        seconds=t1 - t0,
        tokens=tokens,
        wall=(t0, t1),
        group_ms=[(b - a) * 1000 for a, b in zip(window, window[1:])],
        size=n,
        stamp="arrival",
    )


OPTIONS = [  # flag, type, default, help
    ("--base-url", str, None, "a running server (or pass --engine)"),
    ("--dp", int, None, "attention DP ranks to pin requests to"),
    (
        "--num-gpus",
        int,
        None,
        "GPUs to divide throughput by (default: server topology)",
    ),
    ("--context", int, 131072, "prompt tokens: shared prefix plus suffix"),
    ("--new-tokens", int, 4096, "random, uncached suffix tokens per request"),
    ("--tokens-per-step", int, 65536, "prefill: new tokens per rank per step"),
    ("--requests", int, None, "prefill: requests per rank (default 31B/new, max 512)"),
    ("--batch-per-rank", int, 64, "decode: running requests per rank"),
    ("--output-len", int, 2048, "decode: max new tokens; must outlast the run"),
    ("--warmup-steps", int, None, "steps before the window (prefill 5, decode 50)"),
    ("--measure-steps", int, 60, "decode: steps in the window"),
    ("--lo", float, 0.2, "prefill: window start, as a fraction of completions"),
    ("--hi", float, 0.9, "prefill: window end, as a fraction of completions"),
    ("--profile-steps", int, 0, "profile this many steps after the window"),
    ("--profile-dir", str, "/tmp/engine_steps_profile", "profile output directory"),
    ("--timeout", float, 600, "seconds a stalled server gets before the run fails"),
    ("--seed", int, 0, "seed of the random ids"),
    ("--out", str, None, "write the JSON record here"),
]
WORKLOAD = [flag[2:].replace("-", "_") for flag, *_ in OPTIONS if flag != "--out"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["prefill", "decode"])
    for flag, typ, default, help_ in OPTIONS:
        ap.add_argument(flag, type=typ, default=default, help=help_)
    ap.add_argument("--engine", action="store_true", help="in-process engine")
    ap.add_argument("--server-log", nargs="+", default=[], help="logs to cross-check")
    ap.add_argument("--sample-gpu", action="store_true", help="local SM clock, power")
    ap.add_argument("--no-flush", action="store_true", help="keep the server's cache")
    if "--engine" in sys.argv:
        from sglang.srt.server_args import ServerArgs

        ServerArgs.add_cli_args(ap)
    args = ap.parse_args()
    if args.engine == bool(args.base_url) or not args.dp:
        ap.error("pass --dp and exactly one of --engine and --base-url")
    for key in (
        "dp",
        "num_gpus",
        "new_tokens",
        "tokens_per_step",
        "requests",
        "batch_per_rank",
        "output_len",
        "measure_steps",
        "timeout",
    ):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            ap.error(f"--{key.replace('_', '-')} must be positive and finite")
    if args.context < args.new_tokens or not 0 <= args.lo < args.hi <= 1:
        ap.error("need --context >= --new-tokens and 0 <= --lo < --hi <= 1")
    if args.mode == "prefill" and args.tokens_per_step % args.new_tokens:
        ap.error("--tokens-per-step must be a multiple of --new-tokens")
    if args.warmup_steps is None:
        args.warmup_steps = 5 if args.mode == "prefill" else 50
    if args.warmup_steps < 0 or args.profile_steps < 0:
        ap.error("warmup and profile step counts must be nonnegative")
    nc, B = args.new_tokens, args.tokens_per_step
    args.requests = args.requests or min(math.ceil(31 * B / nc), 512)
    args.profile = dict(output_dir=args.profile_dir, num_steps=args.profile_steps)
    if args.engine:
        from sglang.srt.server_args import ServerArgs

        tp = EngineTransport(ServerArgs.from_cli_args(args))
    else:
        tp = HttpTransport(args.base_url, args.timeout)

    try:
        run_benchmark(args, tp)
    finally:
        tp.finish()


def run_benchmark(args, tp):
    import numpy as np

    nc = args.new_tokens
    rng = np.random.default_rng(args.seed)

    def ids(n):  # plain vocabulary ids, clear of special tokens
        return rng.integers(1000, 100000, size=n).tolist()

    prefixes = [ids(args.context - nc) for _ in range(args.dp)]

    def prompt(r):
        return prefixes[r] + ids(nc)

    info = tp.server_info()
    args.num_gpus = gpu_count(info, args.dp, args.num_gpus)
    if info.get("enable_eplb"):
        raise SystemExit("EPLB is on: rebalance steps would land in the window")
    if args.mode == "decode" and info.get("stream_interval", 1) != 1:
        raise SystemExit("decode needs --stream-interval 1: one streamed chunk a step")
    # A warm cache (an earlier run with the same seed) would serve the suffixes
    if not args.no_flush and (err := tp.flush_cache()):
        raise SystemExit(f"flushing the server's cache failed: {err}")

    async def run():
        warm = {"max_new_tokens": 1, "temperature": 0.0}
        for _ in range(2):  # the second pass hits: later requests reuse the prefix
            await batch(args, tp, prompt, 1, warm, [])
        return await (prefill if args.mode == "prefill" else decode)(args, tp, prompt)

    query = ["--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits"]
    if devices := os.environ.get("CUDA_VISIBLE_DEVICES"):
        query += ["-i", devices]  # nvidia-smi itself ignores the variable
    sample_output = tempfile.TemporaryFile(mode="w+t") if args.sample_gpu else None
    sampler, sample_text = None, ""
    try:
        if sample_output is not None:
            sampler = subprocess.Popen(
                ["nvidia-smi", *query, "-lms", "500"], stdout=sample_output, text=True
            )
        res = tp.run(run())
    finally:
        if sampler:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        if sample_output is not None:
            sample_output.seek(0)
            sample_text = sample_output.read()
            sample_output.close()
    size = res.pop("size")
    steps, span, tokens = res["steps"], res["seconds"], res["tokens"]
    res.update(
        ms_per_step=span / steps * 1000, tok_s_per_gpu=tokens / span / args.num_gpus
    )
    if args.mode == "decode":
        res["accept_len"] = tokens / (steps * size * args.dp)
    if sampler:
        rows = [r.split(",") for r in sample_text.splitlines() if "," in r]
        for key, col in zip(("sm_clock_mhz", "power_w"), zip(*rows)):
            res[key + "_median"] = statistics.median(float(x) for x in col)
    lines = [line for path in args.server_log for line in open(path, errors="replace")]
    check = (
        check_logs(log_steps(lines, args.mode, *res["wall"], size), res, size, args.dp)
        if args.server_log
        else {}
    )
    record = dict(
        mode=args.mode,
        transport="engine" if args.engine else "http",
        environment_scope="engine_host" if args.engine else "benchmark_client",
        argv=sys.argv,
        workload={k: getattr(args, k) for k in WORKLOAD},
        result=res,
        cross_check=check,
        server_info=info,
        **environment(),
    )
    print(
        f"{args.mode}: {res['ms_per_step']:.2f} ms/step, "
        f"{res['tok_s_per_gpu']:.0f} tok/s/GPU over {steps:.2f} whole steps "
        f"({res['stamp']} times)"
    )
    if "ms_per_step" in check:
        print(
            f"engine log: {check['ms_per_step']:.2f} ms/step, {check['diff_pct']:+.2f}%"
        )
    for error in check.get("errors", []):
        print(f"ERROR: {error}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1, default=str)
    if check.get("agree") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
