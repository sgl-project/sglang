#!/usr/bin/env python3
"""
FractalMesh Notes IP Registrar — Firebase Notes Integration
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Syncs sovereign.db content to Firebase Notes app,
registers IP/research documents, maintains knowledge base.
Port: 5061 (health only)
"""
import os, json, time, sqlite3, logging, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [NOTES-IP] %(message)s")
log = logging.getLogger("notes_ip")

ROOT    = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT   = os.path.join(ROOT, ".env")
DB_PATH = os.path.join(ROOT, "db", "sovereign.db")

FIREBASE_NOTES_URL = "https://firebase-notes-app-52699481575.us-west1.run.app"

app = Flask(__name__)
CORS(app)

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env"), str(Path.home() / ".secrets/fractal.env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=",1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

# ─── IP Document registry ─────────────────────────────────────────────────────

IP_DOCUMENTS = [
    {
        "title": "FractalMesh Sovereign IP v3.0",
        "category": "whitepaper",
        "content": (
            "Edge-optimized quantized actor-critic RL trading system. "
            "4-bit quantization enables <1ms inference on ARM64. "
            "Fractal signal scoring for BTC/ETH/SOL/XRP/BNB. "
            "Sovereign enclave: phone-side + cloud-side split-brain RL. "
            "Markov LTV model: Prospect→Trial→Active→Upsell→Champion→Churned. "
            "Enochian Gate UCB1 retention engine. "
            "Operator: Samuel James Hiotis, ABN 56628117363, Albury NSW 2640."
        ),
    },
    {
        "title": "WiGLE Geo-Intelligence Product Spec",
        "category": "product",
        "content": (
            "Geo-Intelligence Feed: WiFi network density (WiGLE API) + "
            "NASA EONET earth events + Copernicus satellite imagery. "
            "Delivered as REST API overlay on fractal trading signals. "
            "Price: $349 AUD one-off or $35/mo subscription. "
            "Use case: Regional risk mapping for traders, insurers, infrastructure."
        ),
    },
    {
        "title": "Synthwave Empire NFT Architecture",
        "category": "technical",
        "content": (
            "AI DJ (OpenRouter Mistral 7B) → MIDI JSON → midiutil MIDI → "
            "fluidsynth WAV → mpv playback. "
            "NFT Minter: Pinata IPFS (audio+metadata) → Sugar CLI Solana. "
            "Symbol: FMSW. Royalty: 700 basis points (7%). "
            "Interval: 600s compose, 60s mint poll. "
            "Dev.to auto-publish per mint. Revenue: royalties + $149/$15mo subscription."
        ),
    },
    {
        "title": "FractalMesh Revenue Model v402",
        "category": "business",
        "content": (
            "Revenue streams: (1) One-off products $149-$899 AUD. "
            "(2) Monthly subscriptions $15-$89/mo. "
            "(3) NFT royalties 7% on-chain. "
            "(4) Enterprise white-label deals. "
            "(5) Geo-Intelligence data resale. "
            "Seed MRR: $217/mo from 5 active subscriptions. "
            "Pipeline: 12 Albury-Wodonga leads, top score 88 (Albury City Council). "
            "Zero-capital strategy: all revenue from existing IP + APIs."
        ),
    },
]

def sync_rag_docs():
    """Seed IP documents into rag_docs table."""
    conn    = get_db()
    synced  = 0
    for doc in IP_DOCUMENTS:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO rag_docs(title,content,category) VALUES(?,?,?)",
                (doc["title"], doc["content"], doc["category"])
            )
            synced += 1
        except Exception as e:
            log.warning("RAG doc sync error: %s", e)
    conn.commit(); conn.close()
    log.info("Synced %d IP documents to rag_docs", synced)

def ping_firebase_notes():
    """Health-check Firebase Notes app — log status."""
    try:
        with urllib.request.urlopen(FIREBASE_NOTES_URL, timeout=6) as r:
            status = r.getcode()
            log.info("Firebase Notes: HTTP %d (%s)", status, FIREBASE_NOTES_URL)
            return status == 200
    except Exception as e:
        log.warning("Firebase Notes unreachable: %s", e)
        return False

def register_whitepaper_to_firebase():
    """POST latest whitepaper content to Firebase Notes API if available."""
    api_key = load_env("FIREBASE_API_KEY")
    if not api_key:
        log.info("No FIREBASE_API_KEY — skipping remote registration")
        return
    conn = get_db()
    docs = conn.execute("SELECT * FROM rag_docs ORDER BY id DESC LIMIT 5").fetchall()
    conn.close()
    for doc in docs:
        try:
            body = json.dumps({
                "title":   doc["title"],
                "content": doc["content"],
                "tags":    [doc["category"], "fractalmesh", "sovereign-ip"],
            }).encode()
            req = urllib.request.Request(
                f"{FIREBASE_NOTES_URL}/api/notes",
                data=body, method="POST",
                headers={"Content-Type":"application/json",
                         "Authorization":f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                log.info("Registered to Firebase Notes: %s", doc["title"])
        except Exception as e:
            log.warning("Firebase registration: %s — %s", doc["title"], e)

@app.route("/health")
def health():
    return jsonify({"status":"ok","service":"fm-notes-ip-registrar","port":5061,
                    "firebase_url":FIREBASE_NOTES_URL,
                    "timestamp":datetime.now().isoformat()})

@app.route("/api/rag")
def rag():
    conn = get_db()
    rows = conn.execute("SELECT * FROM rag_docs ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

import threading

def _background_loop():
    while True:
        try:
            sync_rag_docs()
            ping_firebase_notes()
            register_whitepaper_to_firebase()
        except Exception as e:
            log.error("Cycle error: %s", e)
        time.sleep(3600)  # hourly

threading.Thread(target=_background_loop, daemon=True).start()

if __name__ == "__main__":
    sync_rag_docs()
    port = int(os.environ.get("NOTES_PORT", 5061))
    log.info("fm-notes-ip-registrar starting on :%d", port)
    app.run(host="0.0.0.0", port=port, threaded=True)
