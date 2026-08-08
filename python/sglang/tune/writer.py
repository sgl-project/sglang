"""Emit the tuned config — the TWO artifacts, each with its fitting key.

1. Committed community corpus  -> human-readable FILENAME key (coarse: device+sm+family+
   heads+dtype+parallelism). Greppable, PR-reviewable, contributed one device at a time.
   TP/EP/DP are explicit literal fields, not derived into a shape integer.

2. Local first-boot cache      -> opaque sha256 FINGERPRINT over the EXTENDED hardware
   state (device name, sm, CUDA + driver version, max SM clock, PCIe gen/width) plus the
   attention profile and schema version. A throttled/power-capped H20 or a CUDA-bumped box
   (cf. #31310) gets its own re-tune instead of silently reusing a mismatched file.

Both carry ``schema_version`` in the body and are validated on load; both are size-neutral
JSON written with ``json.dump(indent=2)`` — matching the MoE ``save_configs`` house style.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict

from .device import DeviceInfo, _canonical
from .shapes import AttnProfile

SCHEMA_VERSION = "attune/1"


def config_filename(dev: DeviceInfo, profile: AttnProfile) -> str:
    """Coarse, human-readable committed-corpus filename."""
    k = profile.key_fields()
    parts = [
        f"device_name={_canonical(dev.name)}",
        f"sm={dev.sm}",
        f"family={k['family']}",
        f"qo={k['qo']}",
        f"kv={k['kv']}",
        f"hd={k['hd']}",
        f"dtype={k['dtype']}",
        f"kv_dtype={k['kv_dtype']}",
        f"tp={k['tp']}",
        f"ep={k['ep']}",
        f"dp={k['dp']}",
    ]
    return "attn," + ",".join(parts) + ".json"


def fingerprint(dev: DeviceInfo, profile: AttnProfile) -> str:
    """sha256(16 hex) over the extended hardware state + profile + schema — the local
    cache key. Deliberately captures power/clock/PCIe/CUDA so racked units that differ
    only in throttle state or driver do not share a tune."""
    blob = {
        "schema": SCHEMA_VERSION,
        "hw": dev.fingerprint_inputs(),
        "profile": profile.key_fields(),
    }
    payload = json.dumps(blob, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_config(
    dev: DeviceInfo,
    profile: AttnProfile,
    decode_body: Dict[str, dict],
    prefill_body: Dict[str, dict],
    skipped: Dict[str, str],
    provenance: dict,
) -> dict:
    """Assemble the schema-versioned config object (identical for both artifacts;
    only the on-disk key differs)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint(dev, profile),
        "device": dev.fingerprint_inputs(),
        "profile": profile.key_fields(),
        "provenance": provenance,  # git sha, attune version, measured_at (stamped by caller)
        "decode": decode_body,  # "batch:ctxlen" -> {backend, page_size, latency_us}
        "prefill": prefill_body,  # "batch:seqlen" -> {backend, latency_us}
        "skipped": skipped,  # backend -> failure reason (self-documenting)
    }


def save_committed(
    config_dir: str, dev: DeviceInfo, profile: AttnProfile, config: dict
) -> str:
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, config_filename(dev, profile))
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def save_local_cache(cache_dir: str, config: dict) -> str:
    """Local cache keyed by fingerprint under sm<arch>/, mirroring flashinfer_autotune.py.
    Atomic write (write-temp-then-rename) so concurrent boots never see a half file."""
    sm = config["device"]["sm"]
    d = os.path.join(cache_dir, SCHEMA_VERSION, f"sm{sm}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, config["fingerprint"] + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp, path)  # atomic
    return path
