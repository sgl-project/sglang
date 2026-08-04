#!/usr/bin/env python3
"""
fm_devto.py — Dev.to automated publishing agent (dry-run by default)
FractalMesh Omega Titan | Samuel James Hiotis | ABN 56628117363

Vault keys:
  DEVTO_API_KEY       — your Dev.to API token
  ENABLE_DEVTO_PUBLISH — set to "true" to go live (default: dry-run)
"""
import os, json, time, random
from datetime import datetime, timezone

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

API_KEY = os.getenv("DEVTO_API_KEY", "")
ENABLED = os.getenv("ENABLE_DEVTO_PUBLISH", "false").lower() == "true"

TOPICS = [
    ("FractalMesh command hub patterns", ["ai", "devops"]),
    ("Policy-driven agent mesh", ["ai", "architecture"]),
    ("Cost-aware multi-model routing", ["llm", "automation"]),
    ("Audit trails for AI operations", ["security", "ai"]),
    ("Sovereign edge computing in 2026", ["infrastructure", "devops"]),
    ("RF intelligence for local-first systems", ["networking", "iot"]),
    ("Building resilient infrastructure without cloud lock-in", ["devops", "infrastructure"]),
    ("Zero-knowledge proof-of-presence explained", ["security", "blockchain"]),
]

print("[fm-devto] online", flush=True)

while True:
    title_base, tags = random.choice(TOPICS)
    full_title = f"{title_base} | {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    payload = {
        "article": {
            "title": full_title,
            "body_markdown": f"# {full_title}\n\nAutomated placeholder from FractalMesh Omega Titan.\n\n*Samuel James Hiotis | ABN 56628117363*",
            "published": False,
            "tags": tags
        }
    }

    if ENABLED and API_KEY and HAS_REQUESTS:
        mode = "live"
        try:
            r = requests.post(
                "https://dev.to/api/articles",
                headers={"api-key": API_KEY},
                json=payload,
                timeout=30
            )
            out = {"agent": "fm-devto", "mode": mode, "status_code": r.status_code,
                   "title": full_title, "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            out = {"agent": "fm-devto", "mode": mode, "error": str(e),
                   "title": full_title, "ts": datetime.now(timezone.utc).isoformat()}
    else:
        mode = "dry-run"
        out = {"agent": "fm-devto", "mode": mode, "next_title": full_title,
               "ts": datetime.now(timezone.utc).isoformat()}

    print(json.dumps(out), flush=True)
    time.sleep(21600)  # 6-hour cooldown
