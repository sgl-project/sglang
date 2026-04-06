#!/usr/bin/env python3
"""
fm_lba_bridge.py — LBA/TFN shield + workspace integration
Provides identity-linkage heartbeat. TFN is NEVER logged, transmitted,
or stored by this agent — vault-only for compliance.

Samuel James Hiotis | ABN 56628117363
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
    format="%(asctime)s [LBA-BRIDGE] %(message)s",
)
log = logging.getLogger("lba_bridge")

GMAIL_USER = os.getenv("GMAIL_USER", "")
ABN        = os.getenv("ABN", "56628117363")
# TFN: presence is checked (masked), never logged in full
_TFN_SET   = bool(os.getenv("TFN", ""))


def shield():
    out = {
        "agent":        "fm-lba-bridge",
        "identity":     GMAIL_USER or "NOT_SET",
        "abn":          ABN,
        "tfn_shielded": _TFN_SET,
        "lba_status":   "ACTIVE",
        "ts":           datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(out), flush=True)
    log.info(
        "Identity %s linked. TFN shielded: %s. LBA: ACTIVE",
        GMAIL_USER or "NOT_SET",
        _TFN_SET,
    )


if __name__ == "__main__":
    log.info("fm-lba-bridge online")
    while True:
        shield()
        time.sleep(1800)
