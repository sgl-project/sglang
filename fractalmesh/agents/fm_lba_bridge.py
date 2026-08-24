#!/usr/bin/env python3
"""
fm_lba_bridge.py — LBA/TFN shield + ATO identity heartbeat
TFN is NEVER logged, transmitted, or stored — presence checked only.
Samuel James Hiotis | ABN 56628117363
"""
import os
import json
import time
import signal
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LBA-BRIDGE] %(message)s")
log = logging.getLogger("lba_bridge")

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
GMAIL_USER = os.getenv("GMAIL_USER", "")
ABN        = os.getenv("ABN", "56628117363")
INTERVAL   = int(os.getenv("LBA_INTERVAL", "1800"))
_TFN_SET   = bool(os.getenv("TFN", ""))   # presence only — value never used

_running   = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS lba_log (
        id INTEGER PRIMARY KEY, identity TEXT, abn TEXT,
        tfn_shielded INTEGER, lba_status TEXT, ato_compliant INTEGER,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log(identity, abn, tfn_shielded, lba_status, ato_compliant):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO lba_log (identity,abn,tfn_shielded,lba_status,ato_compliant) VALUES (?,?,?,?,?)",
                 (identity, abn, int(tfn_shielded), lba_status, int(ato_compliant)))
    conn.commit(); conn.close()


def shield():
    ts           = datetime.now(timezone.utc).isoformat()
    identity     = GMAIL_USER or "NOT_SET"
    ato_compliant = bool(ABN and _TFN_SET)
    out = {
        "agent":        "fm-lba-bridge",
        "identity":     identity,
        "abn":          ABN,
        "tfn_shielded": _TFN_SET,
        "lba_status":   "ACTIVE",
        "ato_compliant": ato_compliant,
        "ts":           ts,
    }
    print(json.dumps(out), flush=True)
    log.info("Identity=%s | ABN=%s | TFN_shielded=%s | ATO=%s",
             identity, ABN, _TFN_SET, ato_compliant)
    _log(identity, ABN, _TFN_SET, "ACTIVE", ato_compliant)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    log.info("fm-lba-bridge online | interval=%ds | TFN_shielded=%s", INTERVAL, _TFN_SET)
    while _running:
        try:
            shield()
        except Exception as e:
            log.error("ERR %s", e)
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    log.info("fm-lba-bridge stopped.")
