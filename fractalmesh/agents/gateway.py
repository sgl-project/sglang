#!/usr/bin/env python3
"""
FractalMesh Nexus Gateway — Universal API v600
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Port: 8000
Universal proxy + AI generation + system aggregation.
All 14 agents accessible through a single endpoint.
Integrates: OpenRouter AI, all FractalMesh sub-APIs.
"""
import os, json, time, urllib.request, threading
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOT  = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT = os.path.join(ROOT, ".env")

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

# Agent registry
AGENTS = {
    "fm-pod":               {"port":5058,"health":"/health","desc":"Master API"},
    "fm-geosignal":         {"port":5057,"health":"/health","desc":"Live signals + NASA"},
    "fm-analytics":         {"port":5060,"health":"/health","desc":"MRR/ARR/LTV"},
    "fm-notes-ip-registrar":{"port":5061,"health":"/health","desc":"Firebase Notes IP"},
    "fm-tunnel":            {"port":5062,"health":"/health","desc":"Public tunnel"},
}

# Health cache
_HEALTH_CACHE   = {}
_HEALTH_UPDATED = None

def _check_agent(name, cfg):
    try:
        url = f"http://localhost:{cfg['port']}{cfg['health']}"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
            return {"name":name,"status":"online","port":cfg["port"],
                    "desc":cfg["desc"],"data":data}
    except Exception as e:
        return {"name":name,"status":"offline","port":cfg["port"],
                "desc":cfg["desc"],"error":str(e)[:80]}

def _refresh_health():
    global _HEALTH_CACHE, _HEALTH_UPDATED
    results = {}
    for name, cfg in AGENTS.items():
        results[name] = _check_agent(name, cfg)
    _HEALTH_CACHE   = results
    _HEALTH_UPDATED = datetime.now().isoformat()

def _background_health():
    while True:
        time.sleep(30)
        try:
            _refresh_health()
        except Exception:
            pass

threading.Thread(target=_background_health, daemon=True).start()
_refresh_health()  # initial

def _proxy_get(port, path, timeout=8):
    """Proxy GET request to a sub-agent."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=timeout) as r:
            return json.loads(r.read()), r.getcode()
    except Exception as e:
        return {"error": str(e)}, 503

def _proxy_post(port, path, body, timeout=10):
    """Proxy POST request to a sub-agent."""
    try:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"http://localhost:{port}{path}", data=data, method="POST",
            headers={"Content-Type":"application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), r.getcode()
    except Exception as e:
        return {"error": str(e)}, 503

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":    "online",
        "service":   "fm-nexus-gateway",
        "version":   "v600",
        "port":      8000,
        "operator":  "Samuel James Hiotis | ABN 56628117363",
        "timestamp": datetime.now().isoformat(),
    })

@app.route("/system/status")
def system_status():
    """Aggregate status from all agents."""
    results = _HEALTH_CACHE or {}
    online  = sum(1 for a in results.values() if a.get("status")=="online")
    total   = len(AGENTS)
    # Also pull revenue summary
    rev_data, _ = _proxy_get(5058, "/api/revenue/mrr")
    sig_data, _ = _proxy_get(5057, "/api/signals")
    return jsonify({
        "gateway":      "online",
        "version":      "v600",
        "agents_online":online,
        "agents_total": total,
        "agents":       results,
        "revenue": {
            "mrr_aud":   rev_data.get("mrr_aud",0),
            "arr_aud":   rev_data.get("arr_aud",0),
            "active_subs":rev_data.get("active_subs",0),
        },
        "top_signal":   max(sig_data.get("signals",[{}]), key=lambda s: s.get("fractal_score",0), default={}),
        "health_updated":_HEALTH_UPDATED,
        "timestamp":    datetime.now().isoformat(),
    })

@app.route("/agents/list")
def agents_list():
    return jsonify({"agents": [
        {"name":k,"port":v["port"],"desc":v["desc"],
         "status":_HEALTH_CACHE.get(k,{}).get("status","unknown")}
        for k,v in AGENTS.items()
    ],"timestamp":datetime.now().isoformat()})

@app.route("/ai/generate", methods=["POST"])
def ai_generate():
    """Universal AI generation via OpenRouter."""
    data      = request.get_json(force=True) or {}
    prompt    = data.get("prompt","")
    model     = data.get("model","meta-llama/llama-3.1-8b-instruct:free")
    max_tok   = int(data.get("max_tokens", 300))
    system_p  = data.get("system",
        "You are the FractalMesh Nexus AI. Samuel James Hiotis (ABN 56628117363, Albury NSW). "
        "Help with trading, products, leads, code, and system operations. Be concise and accurate."
    )
    if not prompt:
        return jsonify({"error":"prompt required"}), 400
    api_key = load_env("OPENROUTER_API_KEY") or load_env("ANTHROPIC_API_KEY") or load_env("OPENAI_API_KEY")
    if api_key:
        try:
            body = json.dumps({
                "model":    model,
                "messages": [
                    {"role":"system","content":system_p},
                    {"role":"user","content":prompt},
                ],
                "max_tokens": max_tok,
            }).encode()
            base = "https://openrouter.ai/api/v1" if load_env("OPENROUTER_API_KEY") else "https://api.openai.com/v1"
            req  = urllib.request.Request(
                f"{base}/chat/completions", data=body, method="POST",
                headers={"Content-Type":"application/json",
                         "Authorization":f"Bearer {api_key}",
                         "HTTP-Referer":"https://fractalmesh.io",
                         "X-Title":"FractalMesh Nexus"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                result = json.loads(r.read())
            text   = result["choices"][0]["message"]["content"]
            usage  = result.get("usage",{})
            return jsonify({
                "text":       text,
                "model":      model,
                "tokens_in":  usage.get("prompt_tokens",0),
                "tokens_out": usage.get("completion_tokens",0),
                "timestamp":  datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"error":f"AI generation failed: {e}",
                            "note":"Set OPENROUTER_API_KEY in vault"}), 503
    return jsonify({
        "text":  "AI generation unavailable — set OPENROUTER_API_KEY in vault",
        "mock":  True,
        "prompt":prompt,
    })

@app.route("/api/<path:subpath>", methods=["GET","POST"])
def api_proxy(subpath):
    """Intelligent proxy: route to correct sub-agent by path prefix."""
    routes = {
        "analytics": 5060,
        "revenue":   5058,
        "signals":   5057,
        "leads":     5058,
        "orders":    5058,
        "products":  5058,
        "checkout":  5058,
        "chat":      5058,
        "nft":       5058,
        "tunnel":    5062,
        "rag":       5061,
    }
    first = subpath.split("/")[0]
    port  = routes.get(first, 5058)
    full_path = f"/api/{subpath}"
    if request.method == "POST":
        resp, code = _proxy_post(port, full_path, request.get_json(force=True) or {})
    else:
        resp, code = _proxy_get(port, full_path)
    return jsonify(resp), code

@app.route("/metrics")
def metrics():
    """Prometheus-style metrics summary."""
    rev, _  = _proxy_get(5058, "/api/revenue/mrr")
    ana, _  = _proxy_get(5060, "/api/analytics/summary")
    online  = sum(1 for a in _HEALTH_CACHE.values() if a.get("status")=="online")
    return jsonify({
        "fm_agents_online":  online,
        "fm_agents_total":   len(AGENTS),
        "fm_mrr_aud":        rev.get("mrr_aud",0),
        "fm_arr_aud":        rev.get("arr_aud",0),
        "fm_active_subs":    rev.get("active_subs",0),
        "fm_churn_rate_pct": rev.get("churn_rate_pct",0),
        "fm_total_revenue":  ana.get("revenue",{}).get("total_aud",0),
        "fm_nfts_minted":    ana.get("nfts",{}).get("minted",0),
        "timestamp":         datetime.now().isoformat(),
    })

if __name__ == "__main__":
    port = int(os.environ.get("GATEWAY_PORT", 8000))
    print(f"[nexus-gateway] FractalMesh Nexus Gateway v600 starting on :{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
