"""What one prefill or decode step costs at a fixed, steady batch.

Each DP rank warms a long shared prefix; every request adds a random, uncached
--new-tokens suffix and is pinned to its rank. Prefill sends one-token requests,
--tokens-per-step new tokens per rank per step; decode streams --batch-per-rank
requests per rank and times --measure-steps steps.

Throughput counts whole steps only: the window starts and ends at step boundaries
(the scheduler's stamps under --enable-metrics, else response arrival gaps).
--server-log cross-checks the engine's per-step log lines over the same window; it
needs the server on this host's clock, with SGLANG_LOG_MS=1. In HTTP mode the
package, GPU and environment fields describe the benchmark client; server_info
records the server.

--prefill-completions saves measured request timestamps before scoring, including
partial completions on failure. This diagnostic file is not a validated result.

L3 prefill (--prefix-source l3) writes a distinct prefix per request, then flushes
only the device cache. Every measured prefix must be entirely off-device.
The store must hold all warmup, measured and optional profile prefixes. Use an
isolated fresh store for each A/B arm and the same seed/configuration; unlike a
device flush, a new store removes suffixes persisted by an earlier arm.

  python bench_engine_steps.py prefill --base-url URL --dp 2 --prefix-source l3
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
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
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
    """(t0, t1): the ends of the first and last groups that end between the lo and
    hi fractions of all items."""
    if not groups:
        raise SystemExit("no completions to window")
    total, cum, ends = sum(c for _, c in groups), 0, []
    for end, c in groups:
        cum += c
        ends.append((end, cum))
    i = next(k for k, (_, c) in enumerate(ends) if c >= lo * total)
    j = max((k for k, (_, c) in enumerate(ends) if c <= hi * total), default=-1)
    if j <= i:
        raise SystemExit(f"no whole step between the {lo} and {hi} fractions")
    return ends[i][0], ends[j][0]


def log_steps(lines, mode, t0, t1, size, tol=None):
    """Per rank, the engine's ms per step over [t0, t1] (epoch seconds). With `tol`, t0
    and t1 are scheduler step ends: each rank logs a line within `tol` of every step end,
    so the lines from the one at t0 to the one at t1 are the window's steps. Otherwise
    the lines whose interval since the rank's previous line lies inside the window, at
    batch `size`."""
    per_rank, spans, prev, accept, seen = {}, {}, {}, [], set()
    edges, inside = {}, []
    for m in filter(None, map(LINES[mode].search, lines)):
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        ts, step = ts + float(m.group(2) or 0), 0.001 if m.group(2) else 1.0
        rank = int(m.group(3) or 0)
        start, prev[rank] = prev.get(rank, 0.0), ts
        if tol is not None:
            if m.group(2) and t0 - tol < ts <= t1 + tol:
                edges.setdefault(rank, []).append(ts)
                if ts < t1 - tol:  # the batches launched inside the window
                    seen.add(int(m.group(4)))
            continue
        if start < t0 or ts + step > t1:  # stamps are truncated to `step`
            continue
        seen.add(int(m.group(4)))
        inside.append((m, ts, step, rank, start))
    if size is None and inside:
        size = statistics.mode(int(m.group(4)) for m, *_ in inside)
    for m, ts, step, rank, start in inside:
        if int(m.group(4)) == size:
            accept_len = float(m.group(5) or 1.0)
            # The engine's input throughput counts replayed tokens with the new ones
            replay = re.search(r"#replay-token: (\d+)", m.string)
            tokens = size * accept_len + (int(replay.group(1)) if replay else 0)
            ms = tokens / float(m.group(6)) * 1000
            per_rank.setdefault(rank, []).append(ms)
            lo, hi = spans.get(rank, (start, ts + step))
            spans[rank] = min(lo, start), max(hi, ts + step)
            accept += [accept_len] if m.group(5) else []
    if tol is not None:
        whole = {
            r: ts
            for r, ts in edges.items()
            if len(ts) > 1 and ts[0] - t0 <= tol and t1 - ts[-1] <= tol
        }
        return {
            "ms_per_step_per_rank": {
                r: (ts[-1] - ts[0]) / (len(ts) - 1) * 1000 for r, ts in whole.items()
            },
            "complete": bool(prev) and whole.keys() == prev.keys(),
            "sizes_seen": sorted(seen),
        }
    check = {
        "ms_per_step_per_rank": {r: statistics.mean(v) for r, v in per_rank.items()}
    }
    check["coverage"] = min(((b - a) / (t1 - t0) for a, b in spans.values()), default=0)
    check["complete"] = check["coverage"] >= 0.8
    check["sizes_seen"] = sorted(seen)
    if accept:
        check["accept_len"] = statistics.mean(accept)
    return check


def steps_blurred(group_ms):
    median = statistics.median(group_ms)
    return any(not 0.6 * median <= ms <= 1.6 * median for ms in group_ms)


def gpu_count(info, dp, explicit=None):
    """Attention DP partitions TP; ordinary DP replicates the TP/PP group."""
    replicas = info.get("dp_size", 1)
    ranks = replicas * info.get("attn_dp_size", 1)
    if "dp_size" in info and dp != ranks:
        raise SystemExit(f"--dp {dp} does not match the server's {ranks} ranks")
    if explicit is not None:
        return explicit
    if "tp_size" not in info or "dp_size" not in info:
        raise SystemExit("server topology is unavailable; pass --num-gpus explicitly")
    return info["tp_size"] * info.get("pp_size", 1) * replicas


def check_cache(meta, new_tokens, restored=False):
    new = meta["prompt_tokens"] - meta["cached_tokens"]
    if new != new_tokens:
        raise RuntimeError(
            f"expected {new_tokens} uncached suffix tokens, got {new}: "
            "the shared prefix was evicted/missed or part of a suffix was cached"
        )
    if restored:
        # The direct linker reports its off-device loads as host hits.
        tiers = meta.get("cached_tokens_details") or {}
        off_device = tiers.get("host", 0) + tiers.get("storage", 0)
        if tiers.get("device") != 0 or off_device != meta["cached_tokens"]:
            raise RuntimeError(
                f"expected the whole prefix restored from L3, got tiers {tiers or None}"
            )
    return new


def check_logs(check, result, size, dp):
    """Every error voids the cross-check and fails the run."""
    per_rank = check["ms_per_step_per_rank"]
    errors = []
    if set(per_rank) != set(range(dp)):
        errors.append(f"expected ranks {list(range(dp))}, got {sorted(per_rank)}")
    seen = check["sizes_seen"]
    if size is None:  # prefill: one step size, near the measured new tokens per step
        target = result["new_tokens_per_step"]
        if len(seen) != 1 or abs(seen[0] - target) > 0.01 * target:
            errors.append(f"expected one batch size near {target}, got {seen}")
    elif set(seen) != {size}:
        errors.append(f"expected batch size {size}, got {seen}")
    if not check["complete"]:
        errors.append(
            "engine lines miss part of the window on some rank; set SGLANG_LOG_MS=1 "
            "(and --enable-metrics for prefill) on the server"
        )
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

    async def stop_profile(self):
        await self.engine.tokenizer_manager.stop_profile()

    def server_info(self):
        return self.engine.get_server_info()

    def flush_cache(self):
        ret = self.engine.flush_cache()
        return None if ret.success else ret.message or "flush failed"

    def finish(self):
        try:  # abort decode leftovers
            self.engine.tokenizer_manager.abort_request(abort_all=True)
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

    async def stop_profile(self):
        async with self.post("/stop_profile", {}) as resp:
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


def package_version(name):
    try:
        return version(name)  # the first one on sys.path, the one import resolves
    except PackageNotFoundError:
        spec = find_spec(name.replace("-", "_"))
        return spec and spec.origin  # a source tree without metadata: its path


def environment():
    pkgs = {"sglang", "sglang-kernel", "sgl-deep-gemm", "sgl-flash-mla", "torch"}
    gpus = nvidia_smi("--query-gpu=index,name,uuid,driver_version", "--format=csv")
    env = {k: v for k, v in os.environ.items() if k.startswith("SGLANG_")}
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return dict(
        versions={p: package_version(p) for p in sorted(pkgs)},
        gpus=gpus[1:],
        visible_devices=devices,
        sglang_env=env,
    )


async def batch(
    args,
    tp,
    prompt,
    n_each,
    params,
    done,
    what,
    *,
    check_prefix=False,
    on_complete=None,
):
    """n_each requests per rank, submitted round-robin over the ranks so none starts
    ahead, with the configured steps' worth in flight on each."""
    in_flight = (
        max(1, args.tokens_per_step // args.new_tokens) * args.in_flight_steps + 1
    )
    limits = [asyncio.Semaphore(in_flight) for _ in range(args.dp)]

    async def collect(r):
        async for t, meta in tp.generate(prompt(r), params, r, False):
            new = meta["prompt_tokens"] - meta["cached_tokens"]
            if check_prefix:
                check_cache(
                    meta,
                    args.new_tokens,
                    getattr(args, "prefix_source", "device") == "l3",
                )
            done.append((t, meta.get("prefill_finished_time"), new, r))
            if on_complete is not None:
                await on_complete(r)

    async def one(r):
        async with limits[r]:
            await asyncio.wait_for(collect(r), timeout=args.timeout)

    tasks = [
        asyncio.ensure_future(one(r)) for _ in range(n_each) for r in range(args.dp)
    ]
    finished, last, shown = 0, time.time(), time.time()
    try:
        while finished < len(tasks):
            await asyncio.wait([t for t in tasks if not t.done()], timeout=1)
            for task in tasks:
                if task.done() and task.exception():
                    raise task.exception()
            now, n = time.time(), sum(t.done() for t in tasks)
            if n > finished:
                finished, last = n, now
            if now - last > args.timeout:
                raise SystemExit(
                    f"{what}: no request finished for {args.timeout:.0f} s"
                )
            if now - shown >= 5 or finished == len(tasks):
                print(f"{what}: {finished}/{len(tasks)} requests", flush=True)
                shown = now
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def profile_prefill(args, tp, prompt, params):
    """Profile a separate, continuously fed burst once every rank has warmed up;
    the captured steps come from the trace."""
    per_step = max(1, args.tokens_per_step // args.new_tokens)
    warmup = max(
        per_step, math.ceil(args.requests * args.lo), args.warmup_steps * per_step
    )
    reserve = (args.profile_steps + 4) * per_step
    if warmup + reserve > args.requests:
        raise SystemExit(
            f"profile needs {warmup} warmup + {reserve} remaining requests per rank; "
            "raise --requests or lower --lo/--profile-steps"
        )
    completed, trigger = [0] * args.dp, []

    def check_remaining():
        # A drained burst would put queue-drain steps into the trace.
        remaining = [args.requests - n for n in completed]
        if min(remaining) < reserve:
            raise RuntimeError(
                f"profile burst is draining: remaining per rank {remaining}, "
                f"need at least {reserve}; raise --requests"
            )

    async def on_complete(rank):
        completed[rank] += 1
        if trigger or min(completed) < warmup:
            return
        check_remaining()
        trigger.append(completed.copy())
        # Only this request awaits the ACK; its peers keep the batch supplied.
        await tp.start_profile(args.profile)
        check_remaining()

    try:
        await batch(
            args,
            tp,
            prompt,
            args.requests,
            params,
            [],
            "profile",
            check_prefix=True,
            on_complete=on_complete,
        )
        if not trigger:
            raise RuntimeError("profile did not start after all ranks warmed up")
    except BaseException:
        # A failed or cancelled start may still have reached the engine.
        if trigger:
            try:
                await asyncio.wait_for(tp.stop_profile(), timeout=args.timeout)
            except Exception as exc:
                print(f"profile cleanup failed: {exc!r}", file=sys.stderr)
        raise
    return {"trigger_completions_per_rank": trigger[0]}


async def prefill(args, tp, prompt):
    nc, B = args.new_tokens, args.tokens_per_step
    per_step = max(1, B // nc)
    params = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
    n_warmup = args.warmup_steps * per_step
    await batch(args, tp, prompt, n_warmup, params, [], "warmup", check_prefix=True)
    done = []
    drained = False
    try:
        await batch(
            args, tp, prompt, args.requests, params, done, "measured", check_prefix=True
        )
        drained = True
    finally:
        # Persist after timed tasks are joined, before scoring can reject the run.
        if path := getattr(args, "prefill_completions", None):
            with open(path, "w") as f:
                json.dump(
                    dict(
                        schema_version=1,
                        phase="measured_prefill",
                        requests_drained=drained,
                        validation="not_evaluated",
                        columns=[
                            "arrival_time_s",
                            "scheduler_time_s",
                            "new_tokens",
                            "rank",
                        ],
                        completions=done,
                    ),
                    f,
                )
                f.write("\n")
    stamp = "scheduler" if all(s for _, s, _, _ in done) else "arrival"
    times = [s if stamp == "scheduler" else t for t, s, _, _ in done]
    # The window comes from all completions; steps from each rank's own ends, since
    # one rank's ends can trail another's and split a step shared across ranks.
    T0, T1 = group_window(step_groups(times, per_step * args.dp), args.lo, args.hi)
    spans, rank_steps, per_rank, group_ms = [], [], [], []
    for r in range(args.dp):
        mine = [(t, n) for t, (_, _, n, rr) in zip(times, done) if rr == r]
        groups = step_groups([t for t, _ in mine], per_step)
        all_ends = [end for end, _ in groups]
        if r == 0:  # rank 0's ends inside the window anchor every rank's window
            inside = [end for end in all_ends if T0 <= end <= T1]
            if len(inside) < 4:
                raise SystemExit("the window holds under three steps; raise --requests")
            anchors = inside[0], inside[-1]
        a, b = (min(all_ends, key=lambda end: abs(end - x)) for x in anchors)
        ends = [end for end in all_ends if a <= end <= b]
        counts = [c for end, c in groups if a < end <= b]
        if min(counts) < 0.8 * statistics.median(counts):  # a short step
            raise SystemExit(
                f"rank {r} steps finished {min(counts)} to {max(counts)} requests: "
                "the queue ran dry"
            )
        spans.append((a, b))
        rank_steps.append(len(ends) - 1)
        per_rank.append(sum(n for t, n in mine if a < t <= b))
        group_ms += [(y - x) * 1000 for x, y in zip(ends, ends[1:])]
    if len(set(rank_steps)) > 1:
        raise SystemExit(f"ranks ran unequal steps in the window: {rank_steps}")
    # Requests count at completion, so window edges can shift a few between ranks;
    # a rank short by a whole step's budget missed steps.
    if max(per_rank) - min(per_rank) > B:
        raise SystemExit(f"ranks served unequal tokens in the window: {per_rank}")
    # Count steps, not tokens / B: replayed tokens (e.g. the encoder's SWA window)
    # share the step budget, so a full step can carry fewer than B new tokens.
    steps, tokens = rank_steps[0], sum(per_rank)
    t0 = statistics.mean(a for a, _ in spans)
    t1 = statistics.mean(b for _, b in spans)
    profile = (
        await profile_prefill(args, tp, prompt, params) if args.profile_steps else None
    )
    return dict(
        **({"profile": profile} if profile is not None else {}),
        steps=steps,
        seconds=t1 - t0,
        tokens=tokens,
        wall=(t0, t1),
        group_ms=group_ms,
        size=round(tokens / (steps * args.dp)),
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
        deadline, shown = time.time() + args.timeout, time.time()
        while not cond():
            if time.time() - shown >= 5:
                print(why(), flush=True)
                shown = time.time()
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


def positive(cast):
    def parse(text):
        value = cast(text)
        if not math.isfinite(value) or value <= 0:
            raise argparse.ArgumentTypeError(f"{text} is not positive and finite")
        return value

    return parse


pos_int, pos_float = positive(int), positive(float)
OPTIONS = [  # flag, type, default, help
    ("--base-url", str, None, "a running server (or pass --engine)"),
    ("--dp", pos_int, None, "attention DP ranks to pin requests to"),
    ("--num-gpus", pos_int, None, "GPUs to divide by (default: server topology)"),
    ("--prefix-source", str, "device", "prefill prefix source: device or l3"),
    ("--context", int, 131072, "prompt tokens: shared prefix plus suffix"),
    ("--new-tokens", pos_int, 4096, "random, uncached suffix tokens per request"),
    ("--tokens-per-step", pos_int, 65536, "prefill: new tokens per rank per step"),
    ("--requests", pos_int, None, "prefill: requests per rank (31B/new, max 512)"),
    (
        "--in-flight-steps",
        pos_int,
        8,
        "steps worth of requests in flight per rank, plus one; keep fixed across comparisons",
    ),
    ("--batch-per-rank", pos_int, 64, "decode: running requests per rank"),
    ("--output-len", pos_int, 2048, "decode: max new tokens; must outlast the run"),
    ("--warmup-steps", int, None, "steps before the window (prefill 5, decode 50)"),
    ("--measure-steps", pos_int, 60, "decode: steps in the window"),
    ("--lo", float, 0.2, "prefill: window start, as a fraction of completions"),
    ("--hi", float, 0.9, "prefill: window end, as a fraction of completions"),
    ("--profile-steps", int, 0, "profile this many steps after the window"),
    ("--profile-dir", str, "/tmp/engine_steps_profile", "profile output directory"),
    ("--profile-activities", str, None, "e.g. GPU (default: the server's)"),
    ("--timeout", pos_float, 600, "seconds a stalled server gets before failing"),
    ("--seed", int, 0, "seed of the random ids"),
    ("--out", str, None, "write the JSON record here"),
    (
        "--prefill-completions",
        str,
        None,
        "save raw prefill completions before validation",
    ),
]
WORKLOAD = [
    flag[2:].replace("-", "_")
    for flag, *_ in OPTIONS
    if flag not in ("--out", "--prefill-completions")
]


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
    ap.add_argument("--profile-no-stack", action="store_true", help="skip stacks")
    if "--engine" in sys.argv:
        from sglang.srt.server_args import ServerArgs

        ServerArgs.add_cli_args(ap)
    args = ap.parse_args()
    if args.prefix_source not in ("device", "l3"):
        ap.error("--prefix-source must be device or l3")
    if args.prefix_source == "l3":
        if args.mode != "prefill":
            ap.error("--prefix-source l3 is a prefill mode")
        if args.no_flush:
            ap.error(
                "--prefix-source l3 requires a device flush after writing prefixes"
            )
        if args.context <= args.new_tokens:
            ap.error("--prefix-source l3 requires a nonempty prefix")
    if args.engine == bool(args.base_url) or not args.dp:
        ap.error("pass --dp and exactly one of --engine and --base-url")
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
    if args.profile_activities:
        args.profile["activities"] = args.profile_activities.split(",")
    if args.profile_no_stack:
        args.profile["with_stack"] = False
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

    nc, l3 = args.new_tokens, getattr(args, "prefix_source", "device") == "l3"
    rng = np.random.default_rng(args.seed)

    def ids(n, generator=rng):  # plain vocabulary ids, clear of special tokens
        return generator.integers(1000, 100000, size=n).tolist()

    if l3:
        sent = [0] * args.dp
        phase = 1

        def prompt(r):
            sent[r] += 1
            gen = np.random.default_rng([args.seed, 0, r, sent[r]])
            suffix = np.random.default_rng([args.seed, phase, r, sent[r]])
            return ids(args.context - nc, gen) + ids(nc, suffix)

    else:
        prefixes = [ids(args.context - nc) for _ in range(args.dp)]
        # Rank semaphores admit requests independently. Keep suffix assignment
        # reproducible when their completion order changes between runs.
        suffix_rngs = [np.random.default_rng([args.seed, r, 1]) for r in range(args.dp)]

        def prompt(r):
            return prefixes[r] + ids(nc, suffix_rngs[r])

    info = tp.server_info()
    args.num_gpus = gpu_count(info, args.dp, args.num_gpus)
    if l3 and (
        not info.get("enable_unified_cache_external_linker")
        or info.get("unified_cache_external_linker_backend") != "mooncake"
    ):
        raise SystemExit(
            "--prefix-source l3 needs the Mooncake unified external-linker"
        )
    if info.get("enable_eplb"):
        raise SystemExit("EPLB is on: rebalance steps would land in the window")
    if args.mode == "decode" and info.get("stream_interval", 1) != 1:
        raise SystemExit("decode needs --stream-interval 1: one streamed chunk a step")
    # A warm cache (an earlier run with the same seed) would serve the suffixes
    if not args.no_flush and (err := tp.flush_cache()):
        raise SystemExit(f"flushing the server's cache failed: {err}")

    warm = {"max_new_tokens": 1, "temperature": 0.0}
    if l3:
        per_step = max(1, args.tokens_per_step // nc)
        need = args.warmup_steps * per_step + args.requests
        need += args.requests if args.profile_steps else 0
        tp.run(batch(args, tp, prompt, need, warm, [], "L3 prefix write"))
        # Prefixes repeat, suffixes do not. Rank-local seeds keep inputs identical
        # between arms even when ranks submit in a different order.
        sent[:] = [0] * args.dp
        phase = 2
        # The direct-linker reset joins pending loads/offloads before clearing L1.
        if err := tp.flush_cache():
            raise SystemExit(f"flushing the device cache failed: {err}")

    async def run():
        for _ in range(0 if l3 else 2):
            await batch(args, tp, prompt, 1, warm, [], "prefix warmup")
        return await (prefill if args.mode == "prefill" else decode)(args, tp, prompt)

    query = [
        "--query-gpu=timestamp,clocks.sm,power.draw",
        "--format=csv,noheader,nounits",
    ]
    if devices := os.environ.get("CUDA_VISIBLE_DEVICES"):
        query += ["-i", devices]  # nvidia-smi itself ignores the variable
    sample_output = tempfile.TemporaryFile(mode="w+t") if args.sample_gpu else None
    sampler, sample_text = None, ""
    try:
        if sample_output is not None:
            sampler = subprocess.Popen(
                ["nvidia-smi", *query, "-lms", "250"], stdout=sample_output, text=True
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
    if args.mode == "prefill":
        res["new_tokens_per_step"] = size
    steps, span, tokens = res["steps"], res["seconds"], res["tokens"]
    res.update(
        ms_per_step=span / steps * 1000, tok_s_per_gpu=tokens / span / args.num_gpus
    )
    if args.mode == "decode":
        res["accept_len"] = tokens / (steps * size * args.dp)
    if sampler:
        rows = [r.split(", ") for r in sample_text.splitlines() if ", " in r]
        t0, t1 = res["wall"]
        in_window = [
            r[1:]
            for r in rows
            if t0 <= datetime.strptime(r[0], "%Y/%m/%d %H:%M:%S.%f").timestamp() <= t1
        ]
        for key, col in zip(("sm_clock_mhz", "power_w"), zip(*in_window)):
            res[key + "_median"] = statistics.median(float(x) for x in col)
    lines = [line for path in args.server_log for line in open(path, errors="replace")]
    tol = min(res["group_ms"]) / 4000 if res["stamp"] == "scheduler" else None
    check = {}
    if args.server_log:
        # Prefill steps carry the engine's batch, not --tokens-per-step: match any size
        fixed = None if args.mode == "prefill" else size
        check = log_steps(lines, args.mode, *res["wall"], fixed, tol)
        check = check_logs(check, res, fixed, args.dp)
    errors = check.get("errors", [])
    if args.mode == "decode" and steps_blurred(res["group_ms"]):
        errors = errors + [
            "decode steps stray past 0.6-1.6x their median: streamed chunks blur the "
            "step boundaries at this step length, so ms/step is unreliable"
        ]
    record = dict(
        mode=args.mode,
        transport="engine" if args.engine else "http",
        environment_scope="engine_host" if args.engine else "benchmark_client",
        argv=sys.argv,
        workload={k: getattr(args, k) for k in WORKLOAD},
        result=res,
        cross_check=check,
        errors=errors,
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
    for error in errors:
        print(f"ERROR: {error}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1, default=str)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
