#!/usr/bin/env python3
"""
fm_email_listener.py — Gmail IMAP watcher stub
Polls inbox for incoming compliance or client signals.
Vault keys: GMAIL_USER, GMAIL_APP_PASS

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
    format="%(asctime)s [EMAIL-LISTENER] %(message)s",
)
log = logging.getLogger("email_listener")

GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_APP_PASS", "")
POLL_INTERVAL = int(os.getenv("EMAIL_POLL_INTERVAL", "300"))  # seconds


def poll():
    if not GMAIL_USER or not GMAIL_PASS:
        out = {"agent": "fm-email-listener", "status": "credentials_not_set",
               "ts": datetime.now(timezone.utc).isoformat()}
    else:
        # Full IMAP implementation goes here (imaplib)
        out = {"agent": "fm-email-listener", "status": "polling",
               "account": GMAIL_USER,
               "ts": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    log.info("fm-email-listener online (poll every %ds)", POLL_INTERVAL)
    while True:
        poll()
        time.sleep(POLL_INTERVAL)
