#!/usr/bin/env python3
"""
omni_dashboard.py — FractalMesh Omega Titan Omni-Dashboard
Hot-upgrade, agent control, log viewer, DB stats, vault status, terminal proxy
Port: APP_PORT env var (default 8090)
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sys
import json
import time
import signal
import sqlite3
import hashlib
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template_string, Response

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
VAULT    = os.path.expanduser("~/.secrets/fractal.env")
PORT     = int(os.getenv("APP_PORT", "8090"))
TOKEN    = os.getenv("DASHBOARD_TOKEN", "")   # optional bearer auth
ABN      = "56628117363"

app = Flask("omni_dashboard")

# ── Vault loading ─────────────────────────────────────────────────────────────
_vault_path = Path(VAULT)
if _vault_path.exists():
    for _line in _vault_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── Auth middleware ────────────────────────────────────────────────────────────
def _auth_ok() -> bool:
    if not TOKEN:
        return True
    hdr = request.headers.get("Authorization", "")
    return hdr == f"Bearer {TOKEN}"

# ── PM2 helpers ───────────────────────────────────────────────────────────────
def _pm2_list() -> list:
    try:
        r     = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=10)
        procs = json.loads(r.stdout)
        return [{
            "id":       p.get("pm_id"),
            "name":     p.get("name"),
            "status":   p.get("pm2_env", {}).get("status", "?"),
            "cpu":      p.get("monit", {}).get("cpu", 0),
            "mem_mb":   round(p.get("monit", {}).get("memory", 0) / 1048576, 1),
            "restarts": p.get("pm2_env", {}).get("restart_time", 0),
            "pid":      p.get("pid"),
        } for p in procs]
    except Exception as e:
        return [{"error": str(e)}]


def _pm2_logs(name: str, lines: int = 50) -> str:
    try:
        r = subprocess.run(
            ["pm2", "logs", name, "--lines", str(lines), "--nostream"],
            capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr)[-8000:]
    except Exception as e:
        return f"Error: {e}"


def _pm2_action(name: str, action: str) -> dict:
    if action not in ("restart", "stop", "start", "reload"):
        return {"error": "invalid action"}
    try:
        r = subprocess.run(["pm2", action, name], capture_output=True, text=True, timeout=15)
        return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr)[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── DB helpers ─────────────────────────────────────────────────────────────────
def _db_stats() -> dict:
    tables = ["pulse_log", "leads", "revenue", "affiliate_log", "domain_log",
              "device_health", "enochian_log", "contract_log", "bounty_log",
              "wigle_telemetry_ledger", "tokenomics_state", "healer_log",
              "immortality_log", "oversight_log", "lba_log", "email_log",
              "android_health", "toolkit_log", "advert_log"]
    counts = {}
    if not os.path.exists(DB):
        return {"error": "DB not found", "path": DB}
    try:
        conn = sqlite3.connect(DB, timeout=5)
        for t in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                counts[t] = row[0] if row else 0
            except Exception:
                pass
        db_size = os.path.getsize(DB)
        conn.close()
        return {"tables": counts, "total_rows": sum(counts.values()),
                "size_bytes": db_size, "path": DB}
    except Exception as e:
        return {"error": str(e)}

# ── Vault status (presence only, no values) ────────────────────────────────────
def _vault_status() -> dict:
    keys = ["BUS_SECRET", "GMAIL_USER", "GMAIL_APP_PASS", "STRIPE_SECRET_KEY",
            "STRIPE_WEBHOOK_SECRET", "CF_TUNNEL_TOKEN", "EXTERNAL_WEBHOOK_TOKEN",
            "BLOFIN_API_KEY", "KUCOIN_API_KEY", "PIONEX_API_KEY",
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SUPABASE_URL",
            "WIGLE_API_NAME", "WIGLE_API_TOKEN", "PINATA_JWT",
            "GOOGLE_CSE_API_KEY", "DASHBOARD_TOKEN"]
    vault_exists = os.path.exists(VAULT)
    vault_perms  = oct(os.stat(VAULT).st_mode)[-3:] if vault_exists else "---"
    return {
        "vault_path":    VAULT,
        "vault_exists":  vault_exists,
        "vault_perms":   vault_perms,
        "perms_ok":      vault_perms == "600",
        "keys": {k: bool(os.getenv(k)) for k in keys},
        "keys_set":      sum(1 for k in keys if os.getenv(k)),
        "keys_total":    len(keys),
    }

# ── Disk / system ──────────────────────────────────────────────────────────────
def _sys_info() -> dict:
    import shutil
    try:
        total, used, free = shutil.disk_usage(ROOT)
        return {
            "disk_used_gb":  round(used / 1e9, 2),
            "disk_free_gb":  round(free / 1e9, 2),
            "disk_pct":      round(used / total * 100, 1),
            "root":          ROOT,
            "python":        sys.version.split()[0],
            "pid":           os.getpid(),
            "ts_utc":        datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "omni-dashboard",
                    "ts": datetime.now(timezone.utc).isoformat()})


@app.get("/api/agents")
def api_agents():
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"agents": _pm2_list(), "ts": datetime.now(timezone.utc).isoformat()})


@app.get("/api/logs/<name>")
def api_logs(name: str):
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    lines = int(request.args.get("lines", 50))
    return jsonify({"name": name, "log": _pm2_logs(name, min(lines, 500))})


@app.post("/api/agents/<name>/<action>")
def api_agent_action(name: str, action: str):
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    result = _pm2_action(name, action)
    return jsonify(result)


@app.get("/api/db")
def api_db():
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_db_stats())


@app.get("/api/vault")
def api_vault():
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_vault_status())


@app.get("/api/system")
def api_system():
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_sys_info())


@app.post("/api/upgrade")
def api_upgrade():
    """Hot-upgrade: git pull + pm2 restart specified agent or all."""
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    data      = request.get_json(silent=True) or {}
    agent     = data.get("agent", "all")
    do_pull   = data.get("git_pull", False)
    output    = []
    if do_pull:
        try:
            r = subprocess.run(
                ["git", "-C", os.path.expanduser("~/sglang"), "pull", "--ff-only"],
                capture_output=True, text=True, timeout=30)
            output.append(f"git pull: {(r.stdout + r.stderr).strip()[:300]}")
        except Exception as e:
            output.append(f"git pull error: {e}")
    result = _pm2_action(agent, "restart")
    output.append(f"pm2 restart {agent}: {result}")
    return jsonify({"ok": True, "steps": output})


@app.post("/api/exec")
def api_exec():
    """Safe command execution — allowlist only."""
    if not _auth_ok(): return jsonify({"error": "Unauthorized"}), 401
    data    = request.get_json(silent=True) or {}
    cmd_key = data.get("cmd", "")
    ALLOWED = {
        "pm2_list":       ["pm2", "list"],
        "pm2_save":       ["pm2", "save"],
        "git_status":     ["git", "-C", os.path.expanduser("~/sglang"), "status", "--short"],
        "git_log":        ["git", "-C", os.path.expanduser("~/sglang"), "log", "--oneline", "-10"],
        "disk":           ["df", "-h", ROOT],
        "processes":      ["ps", "aux"],
    }
    if cmd_key not in ALLOWED:
        return jsonify({"error": f"Command not in allowlist. Allowed: {list(ALLOWED.keys())}"}), 400
    try:
        r = subprocess.run(ALLOWED[cmd_key], capture_output=True, text=True, timeout=15)
        return jsonify({"cmd": cmd_key, "output": (r.stdout + r.stderr)[:2000]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── Dashboard UI ───────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FractalMesh Omega Titan — Omni-Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body{background:#0a0a0f;color:#e2e8f0;font-family:monospace}
  .card{background:#111827;border:1px solid #1f2937;border-radius:8px;padding:16px}
  .online{color:#22d3ee} .stopped{color:#ef4444} .errored{color:#f59e0b}
  pre{background:#0d1117;padding:12px;border-radius:6px;overflow-x:auto;font-size:11px;max-height:300px;overflow-y:auto}
  .btn{padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;border:none}
  .btn-cyan{background:#0891b2;color:#fff} .btn-red{background:#dc2626;color:#fff}
  .btn-amber{background:#d97706;color:#fff} .badge{padding:2px 8px;border-radius:9999px;font-size:11px}
</style>
</head>
<body class="p-4">
<div class="mb-6">
  <h1 class="text-2xl font-bold" style="color:#22d3ee">⬡ FractalMesh Omega Titan</h1>
  <p class="text-sm text-gray-400">Omni-Dashboard | ABN 56 628 117 363 | Samuel James Hiotis</p>
  <p id="ts" class="text-xs text-gray-600 mt-1"></p>
</div>

<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">AGENTS ONLINE</div>
    <div id="online_count" class="text-3xl font-bold online">—</div>
  </div>
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">DB ROWS</div>
    <div id="db_rows" class="text-3xl font-bold" style="color:#fbbf24">—</div>
  </div>
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">VAULT KEYS SET</div>
    <div id="vault_keys" class="text-3xl font-bold" style="color:#a78bfa">—</div>
  </div>
  <div class="card">
    <div class="text-xs text-gray-400 mb-1">DISK USED</div>
    <div id="disk_pct" class="text-3xl font-bold text-gray-300">—</div>
  </div>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
  <div class="card">
    <div class="flex justify-between items-center mb-3">
      <h2 class="text-sm font-bold" style="color:#22d3ee">AGENT SWARM</h2>
      <button class="btn btn-cyan" onclick="loadAgents()">Refresh</button>
    </div>
    <div id="agent_list" class="text-xs space-y-1"></div>
  </div>
  <div class="card">
    <div class="flex justify-between items-center mb-3">
      <h2 class="text-sm font-bold" style="color:#fbbf24">DATABASE TABLES</h2>
    </div>
    <div id="db_tables" class="text-xs space-y-1"></div>
  </div>
</div>

<div class="card mb-4">
  <div class="flex justify-between items-center mb-3">
    <h2 class="text-sm font-bold" style="color:#a78bfa">HOT UPGRADE</h2>
  </div>
  <div class="flex gap-2 items-center flex-wrap">
    <input id="upgrade_agent" type="text" value="all" placeholder="agent name or 'all'"
      class="bg-gray-900 border border-gray-700 rounded px-3 py-1 text-xs w-48">
    <label class="flex items-center gap-1 text-xs">
      <input type="checkbox" id="do_pull"> git pull first
    </label>
    <button class="btn btn-amber" onclick="hotUpgrade()">Hot Upgrade</button>
    <span id="upgrade_result" class="text-xs text-gray-400"></span>
  </div>
</div>

<div class="card mb-4">
  <div class="flex justify-between items-center mb-3">
    <h2 class="text-sm font-bold" style="color:#22d3ee">LOG VIEWER</h2>
  </div>
  <div class="flex gap-2 mb-2">
    <input id="log_agent" type="text" value="all" placeholder="agent name"
      class="bg-gray-900 border border-gray-700 rounded px-3 py-1 text-xs w-48">
    <input id="log_lines" type="number" value="50" min="10" max="500"
      class="bg-gray-900 border border-gray-700 rounded px-3 py-1 text-xs w-20">
    <button class="btn btn-cyan" onclick="loadLogs()">Load Logs</button>
  </div>
  <pre id="log_output" class="text-gray-300">— select agent and press Load Logs —</pre>
</div>

<div class="card mb-4">
  <div class="flex justify-between items-center mb-3">
    <h2 class="text-sm font-bold" style="color:#a78bfa">VAULT STATUS</h2>
  </div>
  <div id="vault_status" class="text-xs grid grid-cols-2 md:grid-cols-4 gap-1"></div>
</div>

<script>
const api = (path, opts={}) => fetch(path, opts).then(r=>r.json()).catch(e=>({error:e.message}));

async function loadAgents(){
  const d = await api('/api/agents');
  document.getElementById('online_count').textContent = (d.agents||[]).filter(a=>a.status==='online').length;
  document.getElementById('agent_list').innerHTML = (d.agents||[]).map(a=>`
    <div class="flex justify-between items-center py-1 border-b border-gray-800">
      <span class="${a.status==='online'?'online':a.status==='stopped'?'stopped':'errored'}">${a.name||'?'}</span>
      <span class="text-gray-400">${a.status} | ${a.mem_mb}MB | ↺${a.restarts}</span>
      <div class="flex gap-1">
        <button class="btn btn-cyan" onclick="agentAction('${a.name}','restart')">↺</button>
        <button class="btn btn-red" onclick="agentAction('${a.name}','stop')">■</button>
      </div>
    </div>`).join('') || '<span class="text-gray-500">No agents</span>';
}

async function agentAction(name, action){
  const d = await api(`/api/agents/${name}/${action}`, {method:'POST'});
  setTimeout(loadAgents, 1500);
}

async function loadDB(){
  const d = await api('/api/db');
  document.getElementById('db_rows').textContent = (d.total_rows||0).toLocaleString();
  document.getElementById('db_tables').innerHTML = Object.entries(d.tables||{}).map(([t,c])=>`
    <div class="flex justify-between py-0.5 border-b border-gray-800">
      <span class="text-gray-300">${t}</span>
      <span style="color:#fbbf24">${c.toLocaleString()}</span>
    </div>`).join('');
}

async function loadVault(){
  const d = await api('/api/vault');
  document.getElementById('vault_keys').textContent = `${d.keys_set}/${d.keys_total}`;
  document.getElementById('vault_status').innerHTML = Object.entries(d.keys||{}).map(([k,v])=>`
    <div class="flex items-center gap-1">
      <span class="${v?'online':'stopped'}">${v?'✓':'✗'}</span>
      <span class="text-gray-400 truncate">${k}</span>
    </div>`).join('');
}

async function loadSys(){
  const d = await api('/api/system');
  document.getElementById('disk_pct').textContent = `${d.disk_pct||'?'}%`;
  document.getElementById('ts').textContent = `System: ${d.ts_utc||''} | Python ${d.python||''}`;
}

async function loadLogs(){
  const name  = document.getElementById('log_agent').value.trim() || 'all';
  const lines = document.getElementById('log_lines').value || 50;
  const d     = await api(`/api/logs/${name}?lines=${lines}`);
  document.getElementById('log_output').textContent = d.log || d.error || '(empty)';
}

async function hotUpgrade(){
  const agent   = document.getElementById('upgrade_agent').value.trim() || 'all';
  const do_pull = document.getElementById('do_pull').checked;
  const el      = document.getElementById('upgrade_result');
  el.textContent = 'upgrading…';
  const d = await api('/api/upgrade', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({agent, git_pull: do_pull})
  });
  el.textContent = JSON.stringify(d.steps||d).slice(0,200);
}

async function refresh(){
  await Promise.all([loadAgents(), loadDB(), loadVault(), loadSys()]);
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


if __name__ == "__main__":
    print(f"[omni-dashboard] Starting on :{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
