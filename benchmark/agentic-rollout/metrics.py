"""Prometheus parsing and label-aware cache measurements."""

import math

from prometheus_client.parser import text_string_to_metric_families


def metric_values(text):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if math.isfinite(sample.value)
    }


def counter_rate(previous, current, name, seconds):
    keys = {key for key in previous.keys() | current.keys() if key[0] == name}
    if not keys or seconds <= 0:
        return None
    # Missing series and counter resets are gaps, not zero or negative rates.
    if any(
        k not in previous
        or k not in current
        or not math.isfinite(previous[k])
        or not math.isfinite(current[k])
        or current[k] < previous[k]
        for k in keys
    ):
        return None
    return sum(current[k] - previous[k] for k in keys) / seconds


def values(samples, name):
    return {
        labels: value
        for (metric, labels), value in samples.items()
        if metric == "sglang:" + name
    }


def gauge(samples, name):
    items = values(samples, name)
    return (
        sum(items.values())
        if items and all(math.isfinite(v) for v in items.values())
        else None
    )


def percent(used, capacity):
    return (
        100 * used / capacity
        if used is not None and capacity is not None and capacity > 0
        else None
    )


def occupancy(samples, names):
    tiers = [values(samples, name) for name in names]
    if not all(tiers) or any(t.keys() != tiers[0].keys() for t in tiers):
        return [None] * (len(names) + 1)
    totals = [sum(t.values()) for t in tiers]
    if not all(math.isfinite(v) for v in totals):
        return [None] * (len(names) + 1)
    return [*totals, percent(sum(totals[:-1]), totals[-1])]


def hit_percentages(before, after):
    old = values(before, "prefill_effective_tokens_total")
    new = values(after, "prefill_effective_tokens_total")
    modes = ("device_hit", "host_hit", "storage_hit", "input")
    if not old or old.keys() != new.keys():
        return [None] * 3
    workers = {}
    for key, value in new.items():
        labels = dict(key)
        mode = labels.pop("mode", None)
        if mode not in modes or not math.isfinite(value) or not math.isfinite(old[key]):
            return [None] * 3
        delta = value - old[key]
        if delta < 0:
            return [None] * 3
        workers.setdefault(tuple(sorted(labels.items())), {})[mode] = delta
    if any(set(v) != set(modes) for v in workers.values()):
        return [None] * 3
    deltas = {mode: sum(v[mode] for v in workers.values()) for mode in modes}
    denominator = sum(deltas.values())
    return [
        percent(v, denominator)
        for v in (
            deltas["device_hit"],
            deltas["host_hit"],
            denominator - deltas["input"],
        )
    ]
