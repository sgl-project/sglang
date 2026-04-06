#!/usr/bin/env python3
"""
fm_sovereign_ops.py — Sovereign connector: ABN/ATO authority heartbeat
Logs operator identity and authorisation state to pulse_log.

Samuel James Hiotis | ABN 56628117363
Vault keys: NAME, ABN (TFN is never logged or stored by this agent)
"""
import os, time, json, logging
from pathlib import Path
from datetime import datetime, timezone

# ── Load vault ───────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SOVEREIGN-OPS] %(message)s",
)
log = logging.getLogger("sovereign_ops")

NAME = os.getenv("NAME", "Samuel James Hiotis")
ABN  = os.getenv("ABN",  "56628117363")
# NOTE: TFN is explicitly NOT read or logged here — vault-only.


def authorise():
    out = {
        "agent":     "fm-sovereign-ops",
        "principal": NAME,
        "abn":       ABN,
        "status":    "AUTHORIZED",
        "ts":        datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(out), flush=True)
    log.info("Principal: %s | ABN: %s | Status: AUTHORIZED", NAME, ABN)


if __name__ == "__main__":
    log.info("fm-sovereign-ops online")
    while True:
        authorise()
        time.sleep(3600)
