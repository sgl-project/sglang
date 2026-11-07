"""KV transfer time per wave: ``alpha + bytes / bandwidth`` plus a sampled excursion.

The coefficients come from explicit ``kv_transfer_config`` keys, or from a named
profile in the ``network_profiles`` JSON file; explicit keys override a profile.
"""

import hashlib
import json
import random
from typing import Optional


def load_network_profiles(source) -> dict:
    """Profiles by name, from a JSON file path or an inline mapping."""
    if source is None:
        return {}
    if isinstance(source, dict):
        profiles = source
    else:
        with open(source, encoding="utf-8") as f:
            profiles = json.load(f)
    if not isinstance(profiles, dict) or not all(
        isinstance(profile, dict) for profile in profiles.values()
    ):
        raise ValueError(
            "kv_transfer_config.network_profiles must map each network name to "
            "a profile object."
        )
    return profiles


def _resolve(config: dict) -> dict:
    network = config.get("network")
    if network is None:
        return config
    profiles = load_network_profiles(config.get("network_profiles"))
    if network not in profiles:
        available = ", ".join(sorted(profiles)) or "none, set network_profiles"
        raise ValueError(
            f"Unknown kv_transfer_config.network {network!r}. Available: {available}."
        )
    return {**profiles[network], **config}


def _cumulative(pmf: dict) -> list:
    """Turn {value: probability} into a sorted, normalised CDF."""
    items = sorted((float(value), float(prob)) for value, prob in pmf.items())
    total = sum(prob for _, prob in items)
    if not items or total <= 0 or any(value < 0 or prob < 0 for value, prob in items):
        raise ValueError(f"Distribution must contain non-negative values: {pmf}")
    cdf, running = [], 0.0
    for value, prob in items:
        running += prob / total
        cdf.append((value, running))
    return cdf


class _ExcursionBand:
    def __init__(self, config: dict):
        if "probability" not in config or "magnitude_s" not in config:
            raise ValueError(
                "Each kv_transfer_config.excursion band needs a `probability` "
                "and a `magnitude_s` distribution."
            )
        self.max_bytes = float(config.get("max_bytes", float("inf")))
        if self.max_bytes <= 0:
            raise ValueError("Excursion max_bytes must be positive")
        self.probability = float(config["probability"])
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Excursion probability must be between 0 and 1")
        self.cdf = _cumulative(config["magnitude_s"])


class _ExcursionModel:
    """Delay above the alpha-beta line, drawn per size band.

    Each request draws from its own stream seeded by ``(seed, req_id)``, so a run
    is reproducible regardless of the order waves are scheduled in.
    """

    def __init__(self, config: dict, seed: int):
        if not config.get("bands"):
            raise ValueError(
                "kv_transfer_config.excursion needs a non-empty `bands` list."
            )
        self.bands = sorted(
            (_ExcursionBand(band) for band in config["bands"]),
            key=lambda band: band.max_bytes,
        )
        self.seed = seed
        self._req_rngs: dict[str, random.Random] = {}
        # Fitted distributions are dominated by first waves; this scales later
        # waves to their measured mean.
        self.later_wave_exc_scale = float(config.get("later_wave_exc_scale", 1.0))
        if self.later_wave_exc_scale < 0:
            raise ValueError("later_wave_exc_scale must be non-negative")

    def sample(self, nbytes: int, req_id: str, wave_index: int) -> float:
        rng = self._req_rngs.get(req_id)
        if rng is None:
            digest = hashlib.sha256(f"{self.seed}:{req_id}".encode()).digest()
            rng = self._req_rngs[req_id] = random.Random(
                int.from_bytes(digest[:8], "big")
            )
        band = next((b for b in self.bands if nbytes < b.max_bytes), self.bands[-1])
        if rng.random() >= band.probability:
            return 0.0
        draw = rng.random()
        magnitude = next(
            (value for value, cumulative in band.cdf if draw <= cumulative),
            band.cdf[-1][0],
        )
        if wave_index > 0:
            magnitude *= self.later_wave_exc_scale
        return magnitude


class KVTransferEstimator:
    """Per-wave service time ``alpha + bytes / bandwidth`` and excursion."""

    def __init__(self, config: dict, accelerator_bandwidth: Optional[float] = None):
        config = _resolve(config)
        self.network = config.get("network")
        self.alpha = float(config.get("alpha_us", 0.0)) * 1e-6
        if self.alpha < 0:
            raise ValueError("alpha_us must be non-negative")
        if config.get("bandwidth_gbps") is not None:
            self.bandwidth = float(config["bandwidth_gbps"]) * 1e9
            self.bandwidth_source = "config"
        elif accelerator_bandwidth is not None:
            # A nominal link speed, not an achieved KV transfer rate.
            self.bandwidth = float(accelerator_bandwidth)
            self.bandwidth_source = "accelerator"
        else:
            raise ValueError(
                "KV transfer bandwidth is not configured: set "
                "kv_transfer_config.bandwidth_gbps or name a `network`."
            )
        if self.bandwidth <= 0:
            raise ValueError("KV transfer bandwidth must be positive")
        excursion = config.get("excursion")
        self.excursion = (
            _ExcursionModel(excursion, seed=int(config.get("seed", 0)))
            if excursion
            else None
        )

    def est_service_dur(self, nbytes: int) -> float:
        return self.alpha + nbytes / self.bandwidth

    def est_excursion(self, nbytes: int, req_id: str, wave_index: int) -> float:
        if self.excursion is None:
            return 0.0
        return self.excursion.sample(nbytes, req_id, wave_index)

    def describe(self) -> str:
        name = f"[{self.network}]" if self.network else ""
        excursion = (
            f" + excursion({len(self.excursion.bands)} band(s))"
            if self.excursion
            else ""
        )
        return (
            f"alpha_beta{name}(alpha={self.alpha * 1e6:.1f}us, "
            f"bandwidth={self.bandwidth / 1e9:.3f} GB/s from "
            f"{self.bandwidth_source}){excursion}"
        )
