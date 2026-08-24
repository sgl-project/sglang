#!/usr/bin/env python3
"""
FractalMesh Tunnel — Public URL Manager
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Manages ngrok / cloudflared / localtunnel public tunnels
so the dashboard and API are accessible without port-forwarding.
Zero-capital: public URL → share with leads → close sales.
"""
import os, json, time, subprocess, logging, urllib.request, threading
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TUNNEL] %(message)s")
log = logging.getLogger("tunnel")

ROOT  = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT = os.path.join(ROOT, ".env")
DB    = os.path.join(ROOT, "db", "sovereign.db")

app = Flask(__name__)
CORS(app)

TUNNEL_STATE = {
    "active": False,
    "provider": None,
    "url": None,
    "dashboard_url": None,
    "started_at": None,
    "error": None,
}
_PROC = None

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

def _which(cmd):
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def start_ngrok():
    """Start ngrok tunnel to dashboard :8090."""
    global _PROC
    token = load_env("NGROK_TOKEN") or load_env("NGROK_AUTH_TOKEN")
    ngrok = _which("ngrok")
    if not ngrok:
        return False, "ngrok not installed"
    if token:
        subprocess.run([ngrok, "config", "add-authtoken", token], capture_output=True, timeout=5)
    try:
        _PROC = subprocess.Popen(
            [ngrok, "http", "8090", "--log=stdout", "--log-format=json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        time.sleep(3)
        # Get tunnel URL from ngrok API
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5) as r:
            data = json.loads(r.read())
        tunnels = data.get("tunnels",[])
        if tunnels:
            url = tunnels[0].get("public_url","")
            TUNNEL_STATE.update({
                "active": True, "provider": "ngrok", "url": url,
                "dashboard_url": url, "started_at": datetime.now().isoformat(), "error": None,
            })
            log.info("ngrok tunnel active: %s → :8090", url)
            return True, url
        return False, "No tunnels found"
    except Exception as e:
        return False, str(e)

def start_cloudflared():
    """Start cloudflared quick tunnel (no auth needed)."""
    global _PROC
    cf = _which("cloudflared")
    if not cf:
        return False, "cloudflared not installed"
    try:
        _PROC = subprocess.Popen(
            [cf, "tunnel", "--url", "http://localhost:8090"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        time.sleep(4)
        # Read URL from output
        for _ in range(20):
            line = _PROC.stdout.readline()
            if "trycloudflare.com" in line or "https://" in line:
                import re
                m = re.search(r"https://[^\s]+", line)
                if m:
                    url = m.group(0)
                    TUNNEL_STATE.update({
                        "active": True, "provider": "cloudflared", "url": url,
                        "dashboard_url": url, "started_at": datetime.now().isoformat(), "error": None,
                    })
                    log.info("cloudflared tunnel active: %s", url)
                    return True, url
        return False, "Could not parse cloudflared URL"
    except Exception as e:
        return False, str(e)

def start_localtunnel():
    """localtunnel via npx — free, no auth."""
    lt = _which("lt")
    if not lt:
        lt = _which("npx")
        cmd = ["npx", "localtunnel", "--port", "8090"]
    else:
        cmd = ["lt", "--port", "8090"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        import re
        m = re.search(r"https://[^\s]+", proc.stdout + proc.stderr)
        if m:
            url = m.group(0)
            TUNNEL_STATE.update({
                "active": True, "provider": "localtunnel", "url": url,
                "dashboard_url": url, "started_at": datetime.now().isoformat(), "error": None,
            })
            log.info("localtunnel active: %s", url)
            return True, url
        return False, "No URL found"
    except Exception as e:
        return False, str(e)

def auto_start_tunnel():
    """Try providers in order: ngrok → cloudflared → localtunnel."""
    for name, fn in [("ngrok", start_ngrok), ("cloudflared", start_cloudflared),
                     ("localtunnel", start_localtunnel)]:
        log.info("Trying %s...", name)
        ok, info = fn()
        if ok:
            log.info("Tunnel established via %s: %s", name, info)
            return True
        log.warning("%s failed: %s", name, info)
    TUNNEL_STATE["error"] = "All tunnel providers failed"
    log.error("No tunnel available — dashboard accessible on local network only (:8090)")
    return False

def save_url_to_db():
    """Save public URL to audit_log for visibility."""
    if not TUNNEL_STATE.get("url"):
        return
    try:
        import sqlite3
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute("INSERT INTO audit_log(event,detail) VALUES('TUNNEL_URL',?)",
                     (TUNNEL_STATE["url"],))
        conn.commit(); conn.close()
    except Exception:
        pass

@app.route("/health")
def health():
    return jsonify({"status":"ok","service":"fm-tunnel","port":5062,
                    "tunnel":TUNNEL_STATE,"timestamp":datetime.now().isoformat()})

@app.route("/api/tunnel/url")
def tunnel_url():
    return jsonify(TUNNEL_STATE)

def _keep_alive():
    """Monitor tunnel health, restart if dead."""
    while True:
        time.sleep(300)  # check every 5 min
        if TUNNEL_STATE.get("active"):
            url = TUNNEL_STATE.get("url","")
            if url:
                try:
                    urllib.request.urlopen(url, timeout=8)
                except Exception:
                    log.warning("Tunnel down — restarting...")
                    TUNNEL_STATE["active"] = False
                    auto_start_tunnel()
                    save_url_to_db()

if __name__ == "__main__":
    log.info("fm-tunnel starting — auto-detecting tunnel provider")
    ok = auto_start_tunnel()
    if ok:
        save_url_to_db()
        # Print URL prominently
        print(f"\n{'='*60}")
        print(f"  PUBLIC URL: {TUNNEL_STATE['url']}")
        print(f"  Share this with leads for instant demos!")
        print(f"{'='*60}\n")
    threading.Thread(target=_keep_alive, daemon=True).start()
    port = int(os.environ.get("TUNNEL_PORT", 5062))
    app.run(host="0.0.0.0", port=port, threaded=True)
