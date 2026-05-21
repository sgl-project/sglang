"""Long-context cold-vs-warm quality and TTFT probe for SGLang fuzzy KV reuse.

This is the production validation shape for SemanticEmbedding:

1. For each variant, flush cache and run that variant cold with donor
   registration disabled.
2. Flush cache again, run the cluster seed as the only intended donor, then run
   that one variant warm with donor registration disabled.
3. Score warm output against cold output for the exact same variant prompt.

The key quality metrics are ROUGE-L and token F1 between cold-variant and
warm-variant responses. Seed-vs-variant overlap is deliberately not used for
quality, because those are different prompts.

The script can also score a completed raw run with an SGLang server log:

    python scripts/run_long_quality_probe.py score \
        --raw long_quality_raw.json \
        --log-file sglang-server.log \
        --out long_quality_scored.json
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

import aiohttp


FUZZY_CANDIDATE_TYPES = {"partial_80", "partial_60", "paraphrase"}
EXACT_CONTROL_TYPES = {"exact"}
NEGATIVE_CONTROL_TYPES = {"diverse"}

FUZZY_SUCCESS_RE = re.compile(
    r"Fuzzy match success:\s+rid=(?P<rid>\S+)\s+cached=(?P<cached>\d+),\s+"
    r"prompt=(?P<prompt>\d+),\s+offset=(?P<offset>-?\d+)"
)
FUZZY_FAIL_RE = re.compile(
    r"Fuzzy match failed:\s+rid=(?P<rid>\S+)\s+no suitable match found"
)
QUALITY_REUSE_RE = re.compile(r"(?:quality_reuse|reuse_ratio)=(?P<value>[0-9.]+)")
QUALITY_COSINE_RE = re.compile(r"(?:quality_cosine|cosine|similarity)=(?P<value>[0-9.]+)")
QUALITY_TIER_RE = re.compile(r"(?:quality_tier|tier)=(?P<value>[A-Za-z0-9_.:-]+)")
QUALITY_PASSED_RE = re.compile(
    r"(?:quality_passed|passed_quality_gate)=(?P<value>True|False|true|false|1|0)"
)

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def warn(message: str) -> None:
    print(f"\n*** WARNING: {message}\n", file=sys.stderr, flush=True)


def require_or_warn_fuzzy_events(
    log_text: str,
    *,
    require: bool,
    source: str,
) -> None:
    events = parse_fuzzy_events(log_text) if log_text else {}
    if events:
        return

    message = (
        f"No fuzzy success/failure events were parsed from {source}. "
        "Cached-token attribution requires an SGLang server log from a build "
        "that emits 'Fuzzy match success: rid=... cached=... prompt=...'. "
        "Without that log, TTFT and response-quality metrics are still usable, "
        "but cached_tokens, fire_rate, and cache_fraction will be incomplete."
    )
    if require:
        raise RuntimeError(message)
    warn(message)


def longest_common_prefix(left: List[int], right: List[int]) -> int:
    n = min(len(left), len(right))
    for i in range(n):
        if left[i] != right[i]:
            return i
    return n


def words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1 over lowercase word tokens."""
    ref = words(reference)
    cand = words(candidate)
    if not ref or not cand:
        return 0.0
    n = len(cand)
    prev = [0] * (n + 1)
    for rw in ref:
        cur = [0] * (n + 1)
        for j, cw in enumerate(cand, start=1):
            cur[j] = prev[j - 1] + 1 if rw == cw else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_f1(reference: str, candidate: str) -> float:
    ref = words(reference)
    cand = words(candidate)
    if not ref or not cand:
        return 0.0
    overlap = sum((Counter(ref) & Counter(cand)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _copy_action_name(action: Any) -> str:
    return str(getattr(action, "value", action)).lower()


def _is_copy_action(slot: Any) -> bool:
    return (
        getattr(slot, "donor_pos", None) is not None
        and "copy_from_donor" in _copy_action_name(getattr(slot, "action", ""))
    )


def _alignment_diagnostics(
    donor_tokens: List[int],
    target_tokens: List[int],
    *,
    chunk_size: int,
) -> Dict[str, Any]:
    """Report internal reusable-token shape from semblend_core alignment.

    This is intentionally separate from SGLang's realized cache count. SemBlend
    can discover many reusable regions, while the current PR only realizes one
    contiguous block for the upstream-friendly |exact|fuzzy|miss| path.
    """
    base = {
        "available": False,
        "error": None,
        "target_tokens": len(target_tokens),
        "internal_reuse_ratio": None,
        "internal_reuse_tokens": None,
        "segment_count": None,
        "longest_segment_tokens": None,
        "top_segment_tokens": [],
        "exact_chunks": None,
        "fuzzy_chunks": None,
        "fuzzy_recompute_chunks": None,
        "mean_fuzzy_confidence": None,
    }
    if not donor_tokens or not target_tokens:
        base.update(
            {
                "available": True,
                "internal_reuse_ratio": 0.0,
                "internal_reuse_tokens": 0,
                "segment_count": 0,
                "longest_segment_tokens": 0,
                "top_segment_tokens": [],
                "exact_chunks": 0,
                "fuzzy_chunks": 0,
                "fuzzy_recompute_chunks": 0,
                "mean_fuzzy_confidence": 1.0,
            }
        )
        return base

    try:
        from semblend_core.alignment import compute_alignment
    except Exception as e:
        base["error"] = f"semblend_core unavailable: {type(e).__name__}: {e}"
        return base

    try:
        alignment = compute_alignment(
            donor_tokens,
            target_tokens,
            chunk_size=chunk_size,
            fuzzy=True,
        )
    except Exception as e:
        base["error"] = f"alignment failed: {type(e).__name__}: {e}"
        return base

    slots = sorted(
        [slot for slot in alignment.slot_actions if _is_copy_action(slot)],
        key=lambda slot: (int(slot.target_pos), int(slot.donor_pos)),
    )
    segment_lengths: List[int] = []
    run_len = 0
    prev_target = None
    prev_donor = None
    for slot in slots:
        target_pos = int(slot.target_pos)
        donor_pos = int(slot.donor_pos)
        if (
            run_len
            and prev_target is not None
            and prev_donor is not None
            and target_pos == prev_target + 1
            and donor_pos == prev_donor + 1
        ):
            run_len += 1
        else:
            if run_len:
                segment_lengths.append(run_len)
            run_len = 1
        prev_target = target_pos
        prev_donor = donor_pos
    if run_len:
        segment_lengths.append(run_len)

    segment_lengths.sort(reverse=True)
    base.update(
        {
            "available": True,
            "internal_reuse_ratio": float(alignment.reuse_ratio),
            "internal_reuse_tokens": len(slots),
            "segment_count": len(segment_lengths),
            "longest_segment_tokens": segment_lengths[0] if segment_lengths else 0,
            "top_segment_tokens": segment_lengths[:8],
            "exact_chunks": int(getattr(alignment, "exact_chunks", 0)),
            "fuzzy_chunks": int(getattr(alignment, "fuzzy_chunks", 0)),
            "fuzzy_recompute_chunks": int(
                getattr(alignment, "fuzzy_recompute_chunks", 0)
            ),
            "mean_fuzzy_confidence": float(
                getattr(alignment, "mean_fuzzy_confidence", 1.0)
            ),
        }
    )
    return base


def longest_common_run(left: List[int], right: List[int]) -> Dict[str, int]:
    if not left or not right:
        return {"tokens": 0, "donor_start": 0, "target_start": 0}
    match = difflib.SequenceMatcher(
        None,
        left,
        right,
        autojunk=False,
    ).find_longest_match(0, len(left), 0, len(right))
    return {
        "tokens": int(match.size),
        "donor_start": int(match.a),
        "target_start": int(match.b),
    }


def build_reuse_diagnostics(
    seed_tokens: List[int],
    variant_tokens: List[int],
    *,
    chunk_size: int,
) -> Dict[str, Any]:
    exact_prefix_tokens = longest_common_prefix(seed_tokens, variant_tokens)
    suffix_tokens = variant_tokens[exact_prefix_tokens:]
    return {
        "exact_prefix_tokens": exact_prefix_tokens,
        "seed_prompt_tokens": len(seed_tokens),
        "variant_prompt_tokens": len(variant_tokens),
        "remaining_prompt_tokens": len(suffix_tokens),
        "full_longest_exact_run": longest_common_run(seed_tokens, variant_tokens),
        "suffix_longest_exact_run": longest_common_run(seed_tokens, suffix_tokens),
        "full_prompt_alignment": _alignment_diagnostics(
            seed_tokens,
            variant_tokens,
            chunk_size=chunk_size,
        ),
        "suffix_alignment": _alignment_diagnostics(
            seed_tokens,
            suffix_tokens,
            chunk_size=chunk_size,
        ),
    }


async def tokenize_prompt(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "add_special_tokens": True,
    }
    async with session.post(
        f"{endpoint}/tokenize",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        data = await resp.json()
        if resp.status >= 400:
            raise RuntimeError(f"tokenize failed {resp.status}: {data}")
        return {"count": int(data["count"]), "tokens": list(data.get("tokens", []))}


async def flush_cache(session: aiohttp.ClientSession, endpoint: str, timeout_s: float) -> None:
    last_error = None
    for attempt in range(3):
        try:
            async with session.post(
                f"{endpoint}/flush_cache",
                params={"timeout": timeout_s},
                timeout=aiohttp.ClientTimeout(total=max(30, timeout_s + 10)),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"flush_cache failed {resp.status}: {text}")
                return
        except aiohttp.ClientConnectionError as e:
            last_error = e
            if attempt == 2:
                raise
            await asyncio.sleep(1.0)
    if last_error is not None:
        raise last_error


async def completion_stream(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    rid: str,
    extra_key: str,
    max_tokens: int,
    cache_end_pos: Optional[int],
) -> Dict[str, Any]:
    """Run a streaming completion and capture TTFT and full response."""
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "rid": rid,
        "extra_key": extra_key,
    }
    if cache_end_pos is not None:
        body["cache_start_pos"] = 0
        body["cache_end_pos"] = cache_end_pos

    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    chunks: List[str] = []
    finish_reason = None
    usage = None

    async with session.post(
        f"{endpoint}/v1/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"completion failed {resp.status}: {text[:1000]}")
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                text = choice.get("text") or ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")

    return {
        "rid": rid,
        "extra_key": extra_key,
        "ttft_ms": ttft_ms,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "response": "".join(chunks),
        "finish_reason": finish_reason,
        "usage": usage,
    }


def selected_variations(cluster: Dict[str, Any], overlap_types: set[str]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for variation in cluster["variations"]:
        otype = variation["overlap_type"]
        if otype in overlap_types and otype not in seen:
            out.append(variation)
            seen.add(otype)
    missing = overlap_types - seen
    if missing:
        raise ValueError(f"cluster {cluster['cluster_id']} missing variants {sorted(missing)}")
    return out


async def run_cluster(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    cluster: Dict[str, Any],
    overlap_types: set[str],
) -> Dict[str, Any]:
    cid = cluster["cluster_id"]
    source = cluster["source_dataset"]
    length = int(cluster["target_token_length"])
    variations = selected_variations(cluster, overlap_types)
    print(f"[{cid}] {source} {length} tokens", flush=True)

    tokenized: Dict[str, Dict[str, Any]] = {
        "seed": await tokenize_prompt(
            session,
            args.endpoint,
            args.model,
            cluster["seed_text"],
        )
    }
    for variation in variations:
        tokenized[variation["overlap_type"]] = await tokenize_prompt(
            session, args.endpoint, args.model, variation["text"]
        )
    token_counts: Dict[str, int] = {
        key: int(value["count"]) for key, value in tokenized.items()
    }
    reuse_diagnostics: Dict[str, Dict[str, Any]] = {}
    if not args.skip_reuse_diagnostics:
        for variation in variations:
            otype = variation["overlap_type"]
            reuse_diagnostics[otype] = build_reuse_diagnostics(
                tokenized["seed"]["tokens"],
                tokenized[otype]["tokens"],
                chunk_size=args.reuse_diagnostic_chunk_size,
            )

    cold_by_variant: Dict[str, Dict[str, Any]] = {}
    seed_by_variant: Dict[str, Dict[str, Any]] = {}
    warm_by_variant: Dict[str, Dict[str, Any]] = {}

    for variation in variations:
        otype = variation["overlap_type"]

        # Cold phase: no seed donor, no donor registration from variants.
        await flush_cache(session, args.endpoint, args.flush_timeout)
        rid = f"{cid}-cold-{otype}"
        result = await completion_stream(
            session,
            args.endpoint,
            args.model,
            variation["text"],
            rid=rid,
            extra_key=f"cold-{cid}-{otype}",
            max_tokens=args.max_tokens,
            cache_end_pos=0,
        )
        result["prompt_tokens"] = token_counts[otype]
        cold_by_variant[otype] = result
        print(
            f"  cold[{otype:12s}] ttft={fmt_ms(result['ttft_ms'])} "
            f"prompt={token_counts[otype]} resp={len(result['response'])}",
            flush=True,
        )

        # Warm phase: seed is the only intended donor for this variant. Flushing
        # per variant prevents exact-cache entries from earlier warm variants
        # from contaminating TTFT.
        await flush_cache(session, args.endpoint, args.flush_timeout)
        seed = await completion_stream(
            session,
            args.endpoint,
            args.model,
            cluster["seed_text"],
            rid=f"{cid}-seed-{otype}",
            extra_key=f"warm-{cid}-{otype}",
            max_tokens=args.max_tokens,
            cache_end_pos=token_counts["seed"],
        )
        seed["prompt_tokens"] = token_counts["seed"]
        seed_by_variant[otype] = seed
        print(
            f"  seed[{otype:12s}] ttft={fmt_ms(seed['ttft_ms'])} "
            f"prompt={token_counts['seed']} resp={len(seed['response'])}",
            flush=True,
        )
        await asyncio.sleep(args.donor_wait_s)

        rid = f"{cid}-warm-{otype}"
        result = await completion_stream(
            session,
            args.endpoint,
            args.model,
            variation["text"],
            rid=rid,
            extra_key=f"warm-{cid}-{otype}",
            max_tokens=args.max_tokens,
            cache_end_pos=0,
        )
        result["prompt_tokens"] = token_counts[otype]
        warm_by_variant[otype] = result
        print(
            f"  warm[{otype:12s}] ttft={fmt_ms(result['ttft_ms'])} "
            f"prompt={token_counts[otype]} resp={len(result['response'])}",
            flush=True,
        )

    representative_seed = next(iter(seed_by_variant.values())) if seed_by_variant else None
    return {
        "cluster_id": cid,
        "source_dataset": source,
        "target_token_length": length,
        "seed_token_count": cluster.get("seed_token_count"),
        "token_counts": token_counts,
        "variations_meta": [
            {
                "overlap_type": v["overlap_type"],
                "expected_token_overlap": v.get("expected_token_overlap"),
                "description": v.get("description"),
                "reuse_diagnostics": reuse_diagnostics.get(v["overlap_type"]),
            }
            for v in variations
        ],
        "seed": representative_seed,
        "seed_by_variant": seed_by_variant,
        "cold_by_variant": cold_by_variant,
        "warm_by_variant": warm_by_variant,
    }


def parse_fuzzy_events(log_text: str) -> Dict[str, Dict[str, Any]]:
    events: Dict[str, Dict[str, Any]] = {}
    for line in log_text.splitlines():
        m = FUZZY_SUCCESS_RE.search(line)
        if m:
            rid = m.group("rid")
            event = {
                "fuzzy_fired": True,
                "cached_tokens": int(m.group("cached")),
                "matched_prompt_tokens": int(m.group("prompt")),
                "offset": int(m.group("offset")),
                "line": line,
            }
            reuse = QUALITY_REUSE_RE.search(line)
            cosine = QUALITY_COSINE_RE.search(line)
            tier = QUALITY_TIER_RE.search(line)
            passed = QUALITY_PASSED_RE.search(line)
            if reuse:
                event["quality_reuse_ratio"] = float(reuse.group("value"))
            if cosine:
                event["quality_cosine_similarity"] = float(cosine.group("value"))
            if tier:
                event["quality_confidence_tier"] = tier.group("value")
            if passed:
                value = passed.group("value").lower()
                event["quality_passed_gate"] = value in ("true", "1")
            events[rid] = event
            continue
        m = FUZZY_FAIL_RE.search(line)
        if m:
            rid = m.group("rid")
            events.setdefault(
                rid,
                {
                    "fuzzy_fired": False,
                    "cached_tokens": 0,
                    "matched_prompt_tokens": 0,
                    "offset": None,
                    "line": line,
                },
            )
    return events


def nested_get(doc: Dict[str, Any], *keys: str) -> Any:
    cur: Any = doc
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def score_raw(raw: Dict[str, Any], log_text: str = "") -> Dict[str, Any]:
    events = parse_fuzzy_events(log_text) if log_text else {}
    pairs = []
    for cluster in raw["cluster_results"]:
        for meta in cluster["variations_meta"]:
            otype = meta["overlap_type"]
            cold = cluster["cold_by_variant"][otype]
            warm = cluster["warm_by_variant"][otype]
            seed = cluster.get("seed_by_variant", {}).get(otype) or cluster.get("seed")
            rid = warm["rid"]
            event = events.get(rid, {})
            cached = int(event.get("cached_tokens", 0) or 0)
            prompt_tokens = warm.get("prompt_tokens") or cluster["token_counts"].get(otype)
            cold_ttft = cold.get("ttft_ms")
            warm_ttft = warm.get("ttft_ms")
            reuse = meta.get("reuse_diagnostics") or {}
            suffix_internal = nested_get(
                reuse, "suffix_alignment", "internal_reuse_tokens"
            )
            suffix_longest = nested_get(
                reuse, "suffix_alignment", "longest_segment_tokens"
            )
            suffix_exact_run = nested_get(
                reuse, "suffix_longest_exact_run", "tokens"
            )
            full_internal = nested_get(
                reuse, "full_prompt_alignment", "internal_reuse_tokens"
            )
            full_longest = nested_get(
                reuse, "full_prompt_alignment", "longest_segment_tokens"
            )
            pair = {
                "cluster_id": cluster["cluster_id"],
                "source_dataset": cluster["source_dataset"],
                "target_token_length": cluster["target_token_length"],
                "overlap_type": otype,
                "expected_token_overlap": meta.get("expected_token_overlap"),
                "cold_rid": cold["rid"],
                "warm_rid": warm["rid"],
                "seed_rid": seed.get("rid") if seed else None,
                "prompt_tokens": prompt_tokens,
                "seed_prompt_tokens": seed.get("prompt_tokens") if seed else None,
                "cold_variant_ttft_ms": cold_ttft,
                "warm_variant_ttft_ms": warm_ttft,
                "seed_ttft_ms": seed.get("ttft_ms") if seed else None,
                "ttft_speedup_warm_vs_cold": (
                    cold_ttft / warm_ttft if cold_ttft and warm_ttft else None
                ),
                "cached_tokens_in_warm": cached,
                "cache_fraction_of_prompt": (
                    cached / prompt_tokens if cached and prompt_tokens else 0.0
                ),
                "fuzzy_fired": bool(cached > 0),
                "fuzzy_event": event or None,
                "reuse_diagnostics": reuse or None,
                "exact_prefix_tokens": reuse.get("exact_prefix_tokens"),
                "remaining_prompt_tokens": reuse.get("remaining_prompt_tokens"),
                "suffix_internal_reuse_tokens": suffix_internal,
                "suffix_longest_segment_tokens": suffix_longest,
                "suffix_longest_exact_run_tokens": suffix_exact_run,
                "suffix_segment_count": nested_get(
                    reuse, "suffix_alignment", "segment_count"
                ),
                "suffix_internal_reuse_ratio": nested_get(
                    reuse, "suffix_alignment", "internal_reuse_ratio"
                ),
                "full_internal_reuse_tokens": full_internal,
                "full_longest_segment_tokens": full_longest,
                "full_segment_count": nested_get(
                    reuse, "full_prompt_alignment", "segment_count"
                ),
                "full_internal_reuse_ratio": nested_get(
                    reuse, "full_prompt_alignment", "internal_reuse_ratio"
                ),
                "realized_fraction_of_suffix_internal": ratio(
                    cached, suffix_internal
                ),
                "realized_fraction_of_suffix_longest": ratio(
                    cached, suffix_longest
                ),
                "realized_fraction_of_suffix_exact_run": ratio(
                    cached, suffix_exact_run
                ),
                "alignment_fraction_of_suffix_exact_run": ratio(
                    suffix_longest, suffix_exact_run
                ),
                "realized_fraction_of_full_internal": ratio(cached, full_internal),
                "realized_fraction_of_full_longest": ratio(cached, full_longest),
                "unrealized_suffix_internal_tokens": (
                    suffix_internal - cached
                    if suffix_internal is not None
                    else None
                ),
                "rouge_l_cold_vs_warm": rouge_l(cold["response"], warm["response"]),
                "token_f1_cold_vs_warm": token_f1(cold["response"], warm["response"]),
                "ppl_ratio": None,
                "cold_response": cold["response"],
                "warm_response": warm["response"],
                "seed_response": seed.get("response") if seed else "",
            }
            pairs.append(pair)

    return {
        "pairs": pairs,
        "aggregates": build_aggregates(pairs),
        "fuzzy_events_by_rid": events,
    }


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in values if x is not None]
    return statistics.fmean(xs) if xs else None


def aggregate(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    cold_mean = mean(p["cold_variant_ttft_ms"] for p in pairs)
    warm_mean = mean(p["warm_variant_ttft_ms"] for p in pairs)
    return {
        "n": len(pairs),
        "fuzzy_fire_rate": (
            sum(1 for p in pairs if p["fuzzy_fired"]) / len(pairs) if pairs else 0.0
        ),
        "mean_prompt_tokens": mean(p["prompt_tokens"] for p in pairs),
        "mean_cached_tokens": mean(p["cached_tokens_in_warm"] for p in pairs),
        "mean_cache_fraction": mean(p["cache_fraction_of_prompt"] for p in pairs),
        "mean_exact_prefix_tokens": mean(p["exact_prefix_tokens"] for p in pairs),
        "mean_remaining_prompt_tokens": mean(
            p["remaining_prompt_tokens"] for p in pairs
        ),
        "mean_suffix_internal_reuse_tokens": mean(
            p["suffix_internal_reuse_tokens"] for p in pairs
        ),
        "mean_suffix_longest_segment_tokens": mean(
            p["suffix_longest_segment_tokens"] for p in pairs
        ),
        "mean_suffix_longest_exact_run_tokens": mean(
            p["suffix_longest_exact_run_tokens"] for p in pairs
        ),
        "mean_suffix_segment_count": mean(p["suffix_segment_count"] for p in pairs),
        "mean_realized_fraction_of_suffix_internal": mean(
            p["realized_fraction_of_suffix_internal"] for p in pairs
        ),
        "mean_realized_fraction_of_suffix_longest": mean(
            p["realized_fraction_of_suffix_longest"] for p in pairs
        ),
        "mean_realized_fraction_of_suffix_exact_run": mean(
            p["realized_fraction_of_suffix_exact_run"] for p in pairs
        ),
        "mean_alignment_fraction_of_suffix_exact_run": mean(
            p["alignment_fraction_of_suffix_exact_run"] for p in pairs
        ),
        "mean_unrealized_suffix_internal_tokens": mean(
            p["unrealized_suffix_internal_tokens"] for p in pairs
        ),
        "mean_cold_ttft_ms": cold_mean,
        "mean_warm_ttft_ms": warm_mean,
        "mean_warm_minus_cold_ttft_ms": mean(
            (
                p["warm_variant_ttft_ms"] - p["cold_variant_ttft_ms"]
                if p["cold_variant_ttft_ms"] is not None
                and p["warm_variant_ttft_ms"] is not None
                else None
            )
            for p in pairs
        ),
        "mean_ttft_speedup_warm_vs_cold": mean(
            p["ttft_speedup_warm_vs_cold"] for p in pairs
        ),
        "ttft_speedup_ratio_of_means": (
            cold_mean / warm_mean if cold_mean and warm_mean else None
        ),
        "mean_rouge_l_cold_vs_warm": mean(p["rouge_l_cold_vs_warm"] for p in pairs),
        "mean_token_f1_cold_vs_warm": mean(p["token_f1_cold_vs_warm"] for p in pairs),
    }


def build_aggregates(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_overlap = defaultdict(list)
    by_source = defaultdict(list)
    by_length = defaultdict(list)
    for pair in pairs:
        by_overlap[pair["overlap_type"]].append(pair)
        by_source[pair["source_dataset"]].append(pair)
        by_length[str(pair["target_token_length"])].append(pair)

    fuzzy_candidates = [
        p for p in pairs if p["overlap_type"] in FUZZY_CANDIDATE_TYPES
    ]
    hit_only = [p for p in fuzzy_candidates if p["fuzzy_fired"]]
    return {
        "all": aggregate(pairs),
        "fuzzy_candidates": aggregate(fuzzy_candidates),
        "hit_only": aggregate(hit_only),
        "exact_control": aggregate(
            [p for p in pairs if p["overlap_type"] in EXACT_CONTROL_TYPES]
        ),
        "negative_control": aggregate(
            [p for p in pairs if p["overlap_type"] in NEGATIVE_CONTROL_TYPES]
        ),
        "by_overlap_type": {k: aggregate(v) for k, v in sorted(by_overlap.items())},
        "by_source": {k: aggregate(v) for k, v in sorted(by_source.items())},
        "by_length": {k: aggregate(v) for k, v in sorted(by_length.items())},
    }


def fmt_ms(value: Optional[float]) -> str:
    return "NONE" if value is None else f"{value:.0f}ms"


def _csv_filter(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def load_clusters(
    path: str,
    limit: Optional[int],
    lengths: Optional[str] = None,
    sources: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with open(path) as f:
        doc = json.load(f)
    clusters = doc["clusters"] if isinstance(doc, dict) else doc
    allowed_lengths = (
        {int(x) for x in _csv_filter(lengths) or set()} if lengths else None
    )
    allowed_sources = _csv_filter(sources)
    if allowed_lengths is not None:
        clusters = [
            c for c in clusters if int(c["target_token_length"]) in allowed_lengths
        ]
    if allowed_sources is not None:
        clusters = [c for c in clusters if c["source_dataset"] in allowed_sources]
    return clusters[:limit] if limit else clusters


def parse_overlap_types(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


async def run_command(args: argparse.Namespace) -> None:
    clusters = load_clusters(args.clusters, args.limit, args.lengths, args.sources)
    overlap_types = parse_overlap_types(args.overlap_types)
    print(f"Loaded {len(clusters)} clusters from {args.clusters}", flush=True)
    print(f"Variants: {sorted(overlap_types)}", flush=True)

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{args.endpoint}/health",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"server unhealthy: {resp.status}")

        cluster_results = []
        partial_path = args.raw_out + ".partial"
        for cluster in clusters:
            result = await run_cluster(session, args, cluster, overlap_types)
            cluster_results.append(result)
            with open(partial_path, "w") as f:
                json.dump(
                    {
                        "config": vars(args),
                        "cluster_results": cluster_results,
                    },
                    f,
                    indent=2,
                )

    raw = {
        "config": vars(args),
        "cluster_results": cluster_results,
    }
    with open(args.raw_out, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Wrote raw results: {args.raw_out}", flush=True)

    log_text = ""
    if args.log_file and os.path.exists(args.log_file):
        with open(args.log_file) as f:
            log_text = f.read()
        require_or_warn_fuzzy_events(
            log_text,
            require=args.require_fuzzy_log_events,
            source=args.log_file,
        )
    elif args.log_file:
        message = (
            f"Log file was requested but does not exist: {args.log_file}. "
            "Cached-token attribution will be incomplete."
        )
        if args.require_fuzzy_log_events:
            raise RuntimeError(message)
        warn(message)
    else:
        message = (
            "--log-file was omitted. Cached-token attribution requires the "
            "SGLang server log; cached_tokens, fire_rate, and cache_fraction "
            "will be incomplete."
        )
        if args.require_fuzzy_log_events:
            raise RuntimeError(message)
        warn(message)
    scored = score_raw(raw, log_text)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": scored}, f, indent=2)
    print(json.dumps(scored["aggregates"], indent=2), flush=True)
    print(f"Wrote scored results: {args.out}", flush=True)


def score_command(args: argparse.Namespace) -> None:
    with open(args.raw) as f:
        raw = json.load(f)
    with open(args.log_file) as f:
        log_text = f.read()
    require_or_warn_fuzzy_events(
        log_text,
        require=args.require_fuzzy_log_events,
        source=args.log_file,
    )
    scored = score_raw(raw, log_text)
    with open(args.out, "w") as f:
        json.dump({"config": raw.get("config", {}), "results": scored}, f, indent=2)
    print(json.dumps(scored["aggregates"], indent=2), flush=True)
    print(f"Wrote scored results: {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--clusters", required=True)
    run.add_argument("--raw-out", default="long_quality_raw.json")
    run.add_argument("--out", default="long_quality_scored.json")
    run.add_argument("--log-file", default=None)
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--donor-wait-s", type=float, default=5.0)
    run.add_argument("--flush-timeout", type=float, default=60.0)
    run.add_argument(
        "--reuse-diagnostic-chunk-size",
        type=int,
        default=16,
        help="Chunk size used for offline SemBlend reuse-shape diagnostics.",
    )
    run.add_argument(
        "--skip-reuse-diagnostics",
        action="store_true",
        help="Skip offline SemBlend alignment diagnostics while running.",
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--lengths", default=None, help="Comma-separated token buckets")
    run.add_argument("--sources", default=None, help="Comma-separated source datasets")
    run.add_argument(
        "--overlap-types",
        default="exact,partial_80,partial_60,paraphrase,diverse",
    )
    run.add_argument(
        "--require-fuzzy-log-events",
        action="store_true",
        help=(
            "Fail if --log-file is omitted or if no fuzzy success/failure "
            "events can be parsed from the server log."
        ),
    )

    score = sub.add_parser("score")
    score.add_argument("--raw", required=True)
    score.add_argument("--log-file", required=True)
    score.add_argument("--out", required=True)
    score.add_argument(
        "--require-fuzzy-log-events",
        action="store_true",
        help="Fail if no fuzzy success/failure events can be parsed from the log.",
    )

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run_command(args))
    elif args.command == "score":
        score_command(args)


if __name__ == "__main__":
    main()
