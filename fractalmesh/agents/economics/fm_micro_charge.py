#!/usr/bin/env python3
"""
fm_micro_charge.py — Micro-charge monetization agent (dry-run by default)
FractalMesh Omega Titan | Samuel James Hiotis | ABN 56628117363

Vault keys:
  ENABLE_MICROCHARGE — set to "true" to go live (default: dry-run)
  GUMROAD_BASE_URL   — e.g. https://gum.co/yourproduct
  BLOFIN_REFERRAL_URL — affiliate referral URL
"""
import os, json, time, random
from datetime import datetime, timezone

ENABLED = os.getenv("ENABLE_MICROCHARGE", "false").lower() == "true"
GUMROAD_BASE_URL = os.getenv("GUMROAD_BASE_URL", "").strip()
BLOFIN_REFERRAL_URL = os.getenv("BLOFIN_REFERRAL_URL", "").strip()

PRODUCTS = [
    {"name": "Murray River Market Report",        "price": "5.00",  "slug": "murray-report"},
    {"name": "Local SEO Audit",                   "price": "9.00",  "slug": "seo-audit"},
    {"name": "Micro Insight Subscription",        "price": "12.00", "slug": "micro-sub"},
    {"name": "Sovereign Edge Node Blueprint",     "price": "29.00", "slug": "sovereign-blueprint"},
    {"name": "FractalMesh Deployment Starter Kit","price": "49.00", "slug": "deploy-kit"},
]


def link_for(p: dict) -> str:
    if GUMROAD_BASE_URL:
        return GUMROAD_BASE_URL.rstrip("/") + "/" + p["slug"]
    return "https://example.com/offers/" + p["slug"]


print("[fm-micro-charge] online", flush=True)

while True:
    p = random.choice(PRODUCTS)
    print(json.dumps({
        "agent":          "fm-micro-charge",
        "mode":           "live" if ENABLED else "dry-run",
        "product":        p["name"],
        "price":          p["price"],
        "link":           link_for(p),
        "affiliate_link": BLOFIN_REFERRAL_URL or "",
        "ts":             datetime.now(timezone.utc).isoformat()
    }), flush=True)
    time.sleep(240)  # 4-minute heartbeat
