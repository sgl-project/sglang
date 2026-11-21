"""Deterministic LM-head /v1/score workloads and open-loop SLO measurement."""

import asyncio
import gc
import math
import random
from collections import Counter
from functools import wraps
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import aiohttp
import msgspec

MODES = ("pointwise", "pointwise-mis", "setwise", "setwise-mis")
_gc_users = 0
_gc_was_enabled = False


def _without_client_gc(function):
    @wraps(function)
    async def run(*args, **kwargs):
        global _gc_users, _gc_was_enabled
        if _gc_users == 0:
            _gc_was_enabled = gc.isenabled()
            # Retained probe cycles must be collected outside all arrival windows.
            gc.collect()
            gc.disable()
        _gc_users += 1
        try:
            return await function(*args, **kwargs)
        finally:
            _gc_users -= 1
            if _gc_users == 0 and _gc_was_enabled:
                gc.enable()

    return run


def _positive(name, value, *, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            f"{name} must be a finite positive {'integer' if integer else 'number'}"
        )


class Workload(msgspec.Struct, frozen=True):
    mode: str
    query_tokens: int = 120
    candidate_tokens: int = 180
    candidates: int = 10
    labels: tuple[int, ...] = (9542, 874)
    anchor_token: str = "<|object_ref_start|>"
    anchor_id: int = 248047
    seed: int = 42
    model: str = "Qwen/Qwen3.5-4B"
    token_ids: tuple[int, ...] = tuple(range(100, 1000))
    special_token_ids: tuple[int, ...] = ()
    delimiter_id: int = 9999
    apply_softmax: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        for name in ("query_tokens", "candidate_tokens", "candidates"):
            _positive(name, getattr(self, name), integer=True)
        if self.mode.startswith("setwise") and self.candidates % 2:
            raise ValueError("setwise modes require an even candidate count")
        for name in ("labels", "token_ids", "special_token_ids"):
            values = getattr(self, name)
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} must contain nonnegative token IDs")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct IDs")
        for name in ("anchor_id", "delimiter_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.seed) is not int or type(self.apply_softmax) is not bool:
            raise ValueError("seed must be an integer and apply_softmax a boolean")
        if not self.model or not self.anchor_token or not self.labels:
            raise ValueError("model, anchor_token and labels must not be empty")
        if set(self.labels) & {self.anchor_id, self.delimiter_id}:
            raise ValueError("labels must not be anchor or delimiter IDs")
        if not self.content_ids:
            raise ValueError("no ordinary content tokens remain")

    @property
    def content_ids(self):
        excluded = set(self.special_token_ids) | set(self.labels)
        excluded.update((self.anchor_id, self.delimiter_id))
        return tuple(v for v in self.token_ids if v not in excluded)

    @property
    def items(self):
        return 2 if self.mode.startswith("setwise") else self.candidates

    @property
    def anchors_per_item(self):
        return self.candidates // 2 if self.mode.startswith("setwise") else 0


def make_workload(*, mode, tokenizer, model="Qwen/Qwen3.5-4B") -> Workload:
    """Validate the shared two-label workload using an already loaded tokenizer."""
    labels = [
        tokenizer.encode(text, add_special_tokens=False) for text in (" yes", " no")
    ]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("yes/no labels must each encode to one ordinary token")
    special_ids = tuple(tokenizer.all_special_ids)
    label_ids = tuple(ids[0] for ids in labels)
    if any(v < 0 or v >= tokenizer.vocab_size or v in special_ids for v in label_ids):
        raise ValueError("labels must be ordinary base-vocabulary tokens")
    workload = Workload(
        mode, model=model, labels=label_ids, special_token_ids=special_ids
    )
    if tokenizer.encode(workload.anchor_token, add_special_tokens=False) != [
        workload.anchor_id
    ] or workload.anchor_id >= len(tokenizer):
        raise ValueError(f"expected dedicated anchor ID {workload.anchor_id}")
    if any(v >= len(tokenizer) for v in workload.content_ids):
        raise ValueError("content token pool is outside tokenizer vocabulary")
    return workload


def build_requests(workload: Workload, count: int) -> list[dict]:
    """Cycle 16 seeded varied payloads, identical within each MIS-off/on pair.

    Anchors are additional to content tokens. MIS is a server flag, never a
    payload flag. Callers must validate IDs against their actual tokenizer.
    Repeated entries share dictionaries; callers must not mutate payloads.
    """
    _positive("count", count, integer=True)
    rng, pool, requests = random.Random(workload.seed), workload.content_ids, []
    for _ in range(min(count, 16)):
        query = rng.choices(pool, k=workload.query_tokens)
        candidates = [
            rng.choices(pool, k=workload.candidate_tokens)
            for _ in range(workload.candidates)
        ]
        width = workload.anchors_per_item
        items = candidates
        if width:
            items = [
                [
                    token
                    for item in candidates[start : start + width]
                    for token in item + [workload.anchor_id]
                ]
                for start in range(0, workload.candidates, width)
            ]
        request = dict(
            model=workload.model,
            query=query,
            items=items,
            label_token_ids=list(workload.labels),
            apply_softmax=workload.apply_softmax,
            item_first=False,
        )
        if width:
            request["score_extraction_token"] = workload.anchor_token
        requests.append(request)
    return [requests[index % len(requests)] for index in range(count)]


def validate_response(response: Any, workload: Workload) -> None:
    """Require exact dimensions, finite probabilities and selected-label sums."""
    scores = response.get("scores") if isinstance(response, dict) else None
    if not isinstance(scores, list) or len(scores) != workload.items:
        raise ValueError(f"expected scores with {workload.items} items")
    rows = []
    for item in scores:
        if workload.anchors_per_item:
            if not isinstance(item, list) or len(item) != workload.anchors_per_item:
                raise ValueError("wrong anchor count")
            rows.extend(item)
        else:
            rows.append(item)
    for row in rows:
        if not isinstance(row, list) or len(row) != len(workload.labels):
            raise ValueError("wrong label count")
        if any(
            type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v)
            for v in row
        ):
            raise ValueError("scores must be finite probabilities in [0, 1]")
        if workload.apply_softmax and not math.isclose(sum(row), 1.0, abs_tol=1e-4):
            raise ValueError("selected-label probabilities do not sum to one")


def _percentiles(values):
    ordered = sorted(values)
    return {
        **{
            f"p{p}": ordered[math.ceil(len(ordered) * p / 100) - 1] if ordered else None
            for p in (50, 95, 99)
        },
        "max": ordered[-1] if ordered else None,
    }


async def _request(session, url, payload, workload, deadline, index, validator):
    loop = asyncio.get_running_loop()
    dispatched, error, detail = loop.time(), None, None
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error, detail = "http", f"HTTP {response.status}"
            else:
                data = await response.json()
                validate_response(data, workload)
                if validator is not None:
                    try:
                        outcome = validator(payload, data)
                        if outcome is not None and outcome is not True:
                            raise ValueError("sentinel validator rejected response")
                    except (AssertionError, ValueError) as exc:
                        error, detail = "sentinel", str(exc)
    except asyncio.TimeoutError:
        error, detail = "timeout", "request timeout"
    except aiohttp.ClientError as exc:
        error, detail = "transport", str(exc) or type(exc).__name__
    except (ValueError, TypeError) as exc:
        error, detail = "response", str(exc)
    finished = loop.time()
    return dict(
        index=index,
        scheduled_s=deadline,
        dispatch_s=dispatched,
        finished_s=finished,
        latency_ms=(finished - deadline) * 1000,
        dispatch_lag_ms=max(0.0, dispatched - deadline) * 1000,
        error=error,
        detail=detail,
    )


def _offset_records(records, start):
    for record in records:
        for key in ("scheduled_s", "dispatch_s", "finished_s"):
            if record[key] is not None:
                record[key] -= start


async def _warmup(session, url, requests, workload, validator):
    loop, records = asyncio.get_running_loop(), []
    start = loop.time()
    for index, payload in enumerate(requests):
        records.append(
            await _request(
                session, url, payload, workload, loop.time(), index, validator
            )
        )
        if records[-1]["error"]:
            break
    _offset_records(records, start)
    failed = sum(r["error"] is not None for r in records)
    return dict(
        requested=len(requests),
        completed=len(records),
        successful=len(records) - failed,
        failed=failed,
        not_attempted=len(requests) - len(records),
        errors=dict(Counter(r["error"] for r in records if r["error"])),
        elapsed_s=loop.time() - start,
        records=records,
    )


async def _arrivals(session, url, requests, workload, qps, cap, validator):
    loop = asyncio.get_running_loop()
    records, backlog, active, tasks = [], [], set(), []

    async def send(index, deadline):
        records.append(
            await _request(
                session, url, requests[index], workload, deadline, index, validator
            )
        )

    start, peak = loop.time(), 0
    window_end = start + len(requests) / qps
    try:
        for index in range(len(requests)):
            deadline = start + index / qps
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            backlog.append(len(active))
            if len(active) >= cap:
                records.append(
                    dict(
                        index=index,
                        scheduled_s=deadline,
                        dispatch_s=None,
                        finished_s=loop.time(),
                        latency_ms=None,
                        dispatch_lag_ms=None,
                        error="overload",
                        detail="max_in_flight reached; arrival rejected",
                    )
                )
                continue
            task = asyncio.create_task(send(index, deadline))
            tasks.append(task)
            active.add(task)
            peak = max(peak, len(active))
            task.add_done_callback(active.discard)
        # Completion speed must not shorten the offered window.
        await asyncio.sleep(max(0.0, window_end - loop.time()))
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return records, backlog, peak, start, loop.time()


def _summarize(result, records, backlog, peak, start, end):
    policy, qps, count = result["policy"], result["offered_qps"], result["requested"]
    window_s = count / qps
    window_end, elapsed = start + window_s, max(window_s, end - start)
    successful = [r for r in records if r["error"] is None]
    dispatched = [r for r in records if r["dispatch_s"] is not None]
    errors = dict(Counter(r["error"] for r in records if r["error"]))
    latency = _percentiles([r["latency_ms"] for r in successful])
    lag = _percentiles([r["dispatch_lag_ms"] for r in dispatched])
    quarter = max(1, count // 4)
    early, late = sum(backlog[:quarter]) / quarter, sum(backlog[-quarter:]) / quarter
    drain, achieved = max(0.0, end - window_end), len(successful) / elapsed
    invalid, reasons = [], []
    if errors.get("overload"):
        invalid.append("generator_overload")
    if lag["max"] is None or lag["max"] > policy["max_dispatch_lag_s"] * 1000:
        invalid.append("generator_starvation")
    if count < policy["min_samples"]:
        invalid.append("insufficient_samples")
    if window_s < policy["min_duration_s"]:
        invalid.append("insufficient_duration")
    if len(records) != count or len(dispatched) != count:
        invalid.append("incomplete_schedule")
    if len(successful) != count:
        reasons.append("request_failures")
    if latency["p99"] is None or latency["p99"] > policy["slo_ms"]:
        reasons.append("p99_slo")
    if achieved < qps * (1 - policy["throughput_tolerance"]):
        reasons.append("throughput_shortfall")
    if late - early > policy["max_backlog_growth"]:
        reasons.append("backlog_growth")
    if drain > policy["max_drain_s"]:
        reasons.append("excessive_drain")
    workload = result["workload"]
    result.update(
        passed=not (invalid or reasons),
        valid=not invalid,
        reasons=invalid + reasons,
        scheduled=len(records),
        not_scheduled=count - len(records),
        dispatched=len(dispatched),
        completed=len(dispatched),
        successful=len(successful),
        failed=count - len(successful),
        errors=errors,
        achieved_qps=achieved,
        dispatched_qps=len(dispatched) / elapsed,
        items_per_s=achieved
        * (2 if workload["mode"].startswith("setwise") else workload["candidates"]),
        decisions_per_s=achieved * workload["candidates"],
        label_scores_per_s=achieved * workload["candidates"] * len(workload["labels"]),
        latency_ms=latency,
        dispatch_lag_ms=lag,
        arrival_window_s=window_s,
        elapsed_s=elapsed,
        total_elapsed_s=result["warmup"]["elapsed_s"] + elapsed,
        drain_s=drain,
        completed_in_window=sum(r["finished_s"] <= window_end for r in dispatched),
        backlog=dict(
            peak=peak,
            at_window_end=sum(r["finished_s"] > window_end for r in records),
            in_flight_at_window_end=sum(
                r["dispatch_s"] <= window_end < r["finished_s"] for r in dispatched
            ),
            early_mean=early,
            late_mean=late,
            growth=late - early,
            at_arrival=backlog,
        ),
        records=sorted(records, key=lambda r: r["index"]),
    )
    _offset_records(records, start)
    return result


@_without_client_gc
async def run_load(
    base_url: str,
    workload: Workload,
    *,
    qps: float,
    num_requests: int,
    warmup: int = 5,
    max_in_flight: int = 256,
    timeout_s: float = 30.0,
    min_samples: int = 100,
    min_duration_s: float = 10.0,
    slo_ms: float = 500.0,
    throughput_tolerance: float = 0.05,
    max_dispatch_lag_s: float = 0.05,
    max_drain_s: float = 0.5,
    max_backlog_growth: float = 2.0,
    metadata: Optional[dict] = None,
    response_validator: Optional[Callable[[dict, dict], Optional[bool]]] = None,
) -> dict:
    """Measure absolute open-loop arrivals, including dispatch lag in latency.

    Validity describes the offered schedule, not server success. Every requested
    response and warmup must succeed to pass. Optional synchronous numerical
    validation returns None/True or rejects with False/ValueError/AssertionError.
    """
    policy = dict(
        timeout_s=timeout_s,
        min_samples=min_samples,
        min_duration_s=min_duration_s,
        slo_ms=slo_ms,
        throughput_tolerance=throughput_tolerance,
        max_dispatch_lag_s=max_dispatch_lag_s,
        max_drain_s=max_drain_s,
        max_backlog_growth=max_backlog_growth,
        max_in_flight=max_in_flight,
    )
    for name, value in dict(
        qps=qps, num_requests=num_requests, warmup=warmup, **policy
    ).items():
        _positive(
            name,
            value,
            integer=name in {"num_requests", "warmup", "min_samples", "max_in_flight"},
        )
    if throughput_tolerance >= 1 or slo_ms > 500 or max_dispatch_lag_s > 0.05:
        raise ValueError(
            "require tolerance < 1, SLO <= 500 ms and dispatch lag <= 50 ms"
        )
    if not math.isfinite(num_requests / qps) or num_requests / qps <= 0:
        raise ValueError("num_requests / qps must be finite and positive")
    if response_validator is not None and not callable(response_validator):
        raise ValueError("response_validator must be callable")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base_url must be an HTTP(S) URL")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials, query or fragment")
    requests = build_requests(workload, warmup + num_requests)
    policy.update(
        client_gc="collected_before_probe_and_paused",
        response_validator_enabled=response_validator is not None,
    )
    result = dict(
        workload=msgspec.to_builtins(workload),
        metadata=metadata or {},
        offered_qps=qps,
        requested=num_requests,
        policy=policy,
    )
    url = base_url.rstrip("/") + "/v1/score"
    # No hidden connector queue: only the explicit arrival-rejection cap applies.
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as session:
        result["warmup"] = await _warmup(
            session, url, requests[:warmup], workload, response_validator
        )
        if result["warmup"]["failed"]:
            result.update(
                passed=False,
                valid=False,
                reasons=["warmup_failed"],
                scheduled=0,
                dispatched=0,
                completed=0,
                successful=0,
                failed=0,
                not_scheduled=num_requests,
                errors={},
                achieved_qps=0.0,
                latency_ms=_percentiles([]),
                dispatch_lag_ms=_percentiles([]),
                records=[],
            )
            return result
        measurement = await _arrivals(
            session,
            url,
            requests[warmup:],
            workload,
            qps,
            max_in_flight,
            response_validator,
        )
    return _summarize(result, *measurement)
