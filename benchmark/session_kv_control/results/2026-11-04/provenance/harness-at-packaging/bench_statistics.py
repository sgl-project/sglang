"""Small deterministic statistics and causal, client-observed transfer costs."""
import hashlib
import math
import random
import statistics


def quantile(values, p):
    if not values:
        return None
    items = sorted(values)
    position = (len(items)-1)*p
    lo, hi = math.floor(position), math.ceil(position)
    return items[lo] + (items[hi]-items[lo])*(position-lo)


def bootstrap_mean_ci(values, *, seed=1729, resamples=4000, confidence=.95):
    """Resample complete paired deltas, never independently resample each arm."""
    if not values:
        return {"n_pairs": 0, "mean": None, "ci": None, "resamples": resamples}
    rng = random.Random(seed)
    estimates = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples)]
    alpha = (1-confidence)/2
    return {"n_pairs": len(values), "mean": statistics.mean(values),
            "ci": [quantile(estimates, alpha), quantile(estimates, 1-alpha)],
            "confidence": confidence, "resamples": resamples, "seed": seed,
            "method": "percentile bootstrap of paired mean deltas",
            "small_sample": len(values) < 20}


def stable_seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


class PrefetchCostEMA:
    """Estimate registration-complete cost; this is not isolated device I/O time.

    Late operations are kept separately because on-demand reads may contend.
    The prediction only uses already observed, uncontended completions.
    """
    def __init__(self, alpha=.25, initial_ms=500, safety=1.5):
        if not all(math.isfinite(x) for x in (alpha, initial_ms, safety)) or not 0 < alpha <= 1 or initial_ms <= 0 or safety <= 0:
            raise ValueError("invalid EMA configuration")
        self.alpha, self.initial_ms, self.safety = alpha, initial_ms, safety
        self.states = {}

    def observe(self, key, elapsed_ms, estimated_bytes=None, *, contaminated=False):
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("invalid completion cost")
        name = str(key) + (":contended" if contaminated else ":uncontended")
        state = self.states.setdefault(name, {"n": 0, "completion_ms_ema": None,
                                               "estimated_bytes_ema": None, "estimated_bytes_per_ms_ema": None})
        state["n"] += 1
        def update(field, value):
            if value is not None:
                old = state[field]
                state[field] = value if old is None else self.alpha*value+(1-self.alpha)*old
        update("completion_ms_ema", elapsed_ms)
        update("estimated_bytes_ema", estimated_bytes)
        update("estimated_bytes_per_ms_ema", estimated_bytes/elapsed_ms if estimated_bytes and elapsed_ms else None)
        return dict(state)

    def choose(self, key, available_wait_s, estimated_bytes=None):
        state = self.states.get(str(key)+":uncontended")
        predicted = state["completion_ms_ema"] if state else self.initial_ms
        required = predicted*self.safety
        return {"lead_s": min(available_wait_s, required/1000), "predicted_completion_ms": predicted,
                "requested_safety_lead_ms": required, "available_wait_ms": available_wait_s*1000,
                "window_insufficient": required > available_wait_s*1000,
                "sample_n": state["n"] if state else 0, "estimate_source": "historical_uncontended_ema" if state else "configured_initial_cost",
                "estimated_full_kv_bytes": estimated_bytes,
                "alpha": self.alpha, "safety_factor": self.safety,
                "physical_io_bytes": None}
