"""
FractalMesh Omni Nexus v2.1.0
FastAPI terminal-style dashboard on :8095.
Endpoints: /api/health /api/stats /api/agents /api/affiliates
           /api/fdi /api/guardrails /api/ip /api/compliance
           /api/log/recent
Auto-refreshes every 30s in browser.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import sqlite3
import subprocess
from datetime import datetime, timezone

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    raise SystemExit(
        "[fm-omni-nexus] fastapi/uvicorn not installed. "
        "Run: pip install fastapi uvicorn"
    )

ROOT = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB   = os.path.join(ROOT, "database", "sovereign.db")
PORT = int(os.getenv("NEXUS_PORT", "8095"))

PHI      = 1.6180339887
OPERATOR = "Samuel James Hiotis"
ABN      = "56 628 117 363"
SITE     = "https://fractalmesh.net"

app = FastAPI(title="FractalMesh Omni Nexus", version="2.1.0")


# ── helpers ───────────────────────────────────────────────────────────────────

def _db(sql: str, params=(), default=None):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return default if default is not None else []


def _scalar(sql: str, params=(), default=0):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        row  = conn.execute(sql, params).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else default
    except Exception:
        return default


def _pm2_list() -> list:
    try:
        out = subprocess.check_output(
            ["pm2", "jlist"], timeout=10, stderr=subprocess.DEVNULL)
        return json.loads(out)
    except Exception:
        return []


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":    "ok",
        "operator":  OPERATOR,
        "abn":       ABN,
        "ts":        datetime.now(tz=timezone.utc).isoformat(),
        "db_exists": os.path.exists(DB),
    }


@app.get("/api/stats")
def stats():
    total_rev    = _scalar("SELECT COALESCE(SUM(amount_aud),0) FROM revenue")
    total_orders = _scalar("SELECT COUNT(*) FROM orders WHERE status='paid'")
    active_prods = _scalar("SELECT COUNT(*) FROM products WHERE active=1")
    leads        = _scalar("SELECT COUNT(*) FROM leads")
    aff_programs = _scalar("SELECT COUNT(*) FROM affiliates WHERE status='active'")
    aff_clicks   = _scalar(
        "SELECT COUNT(*) FROM affiliate_clicks WHERE ts>datetime('now','-1 day')")
    aff_earned   = _scalar(
        "SELECT COALESCE(SUM(amount),0) FROM affiliate_conversions "
        "WHERE status!='rejected'")
    aff_pending  = _scalar(
        "SELECT COALESCE(SUM(amount),0) FROM affiliate_conversions "
        "WHERE status='pending'")
    methane_anom = _scalar(
        "SELECT COUNT(*) FROM methane_readings WHERE is_anomaly=1")
    ais_alerts   = _scalar(
        "SELECT COUNT(*) FROM ais_alerts WHERE resolved=0")
    ip_value     = _scalar(
        "SELECT COALESCE(SUM(value_estimate_aud),0) FROM ip_registry")
    content      = _scalar("SELECT COUNT(*) FROM content_pieces")
    drip_active  = _scalar(
        "SELECT COUNT(*) FROM drip_sequences WHERE status='active'")
    conv_rate    = round((total_orders / max(leads, 1)) * 100, 2)
    phi          = round(float(total_rev) * PHI / max(float(total_rev), 1), 4)

    return {
        "revenue_aud":         round(float(total_rev), 2),
        "orders":              total_orders,
        "products_active":     active_prods,
        "leads":               leads,
        "conversion_pct":      conv_rate,
        "affiliate_programs":  aff_programs,
        "affiliate_clicks_24h": aff_clicks,
        "affiliate_earned_aud": round(float(aff_earned), 2),
        "affiliate_pending_aud": round(float(aff_pending), 2),
        "methane_anomalies":   methane_anom,
        "ais_open_alerts":     ais_alerts,
        "ip_portfolio_aud":    round(float(ip_value), 2),
        "content_pieces":      content,
        "drip_active":         drip_active,
        "phi_score":           phi,
        "ts":                  datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/agents")
def agents():
    procs = _pm2_list()
    return {
        "count":    len(procs),
        "agents":   [
            {
                "name":   p.get("name"),
                "status": p.get("pm2_env", {}).get("status"),
                "uptime": p.get("pm2_env", {}).get("pm_uptime"),
                "restarts": p.get("pm2_env", {}).get("restart_time", 0),
                "cpu":    p.get("monit", {}).get("cpu", 0),
                "mem_mb": round(p.get("monit", {}).get("memory", 0) / 1048576, 1),
            }
            for p in procs
        ],
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/affiliates")
def affiliates():
    rows = _db(
        "SELECT program, network, commission_type, commission_value, "
        "cookie_days, payout_threshold, status, notes "
        "FROM affiliates ORDER BY program")
    clicks = _db(
        "SELECT program, COUNT(*) as clicks FROM affiliate_clicks "
        "WHERE ts>datetime('now','-7 days') GROUP BY program")
    click_map = {r["program"]: r["clicks"] for r in clicks}
    for r in rows:
        r["clicks_7d"] = click_map.get(r["program"], 0)
    return {"programs": rows, "count": len(rows),
            "ts": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/api/fdi")
def fdi():
    """Satellite intelligence: methane anomalies + AIS alerts."""
    methane = _db(
        "SELECT id, source, lat, lon, ch4_ppb, ch4_enhancement, "
        "estimated_flux_kt, sensing_date, phi_score "
        "FROM methane_readings WHERE is_anomaly=1 "
        "ORDER BY phi_score DESC LIMIT 20")
    ais = _db(
        "SELECT id, mmsi, vessel_name, alert_type, lat, lon, "
        "dark_hours, ts FROM ais_alerts WHERE resolved=0 "
        "ORDER BY ts DESC LIMIT 20")
    reports = _db(
        "SELECT report_ref, tier, status, amount_aud, phi_score, ts "
        "FROM methane_report_log ORDER BY ts DESC LIMIT 10")
    return {
        "methane_anomalies": methane,
        "ais_alerts":        ais,
        "recent_reports":    reports,
        "ts":                datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/guardrails")
def guardrails():
    """Vault key presence check — never exposes values."""
    keys = [
        "GMAIL_USER", "GMAIL_APP_PASS", "OPENAI_API_KEY",
        "STRIPE_SECRET_KEY", "DEVTO_API_KEY",
        "CDSE_USER", "AISHUB_USER", "ADMOB_PUBLISHER_ID",
        "ENABLE_CONTENT", "ENABLE_DRIP", "ENABLE_NEGOTIATOR",
        "ENABLE_METHANE_REPORTS", "ENABLE_ADMOB", "ENABLE_CAMPAIGNS",
        "ENABLE_SMART_CONTRACTS", "ENABLE_WATERMARK", "ENABLE_GEO_VALIDATOR",
        "ENABLE_CARBON_CREDITS", "ENABLE_LICENSING",
    ]
    presence = {k: bool(os.getenv(k, "")) for k in keys}
    enabled  = [k for k, v in presence.items()
                if k.startswith("ENABLE_") and v and
                os.getenv(k, "false").lower() == "true"]
    return {
        "vault_keys":    presence,
        "live_features": enabled,
        "ts":            datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/ip")
def ip_registry():
    rows = _db(
        "SELECT title, category, value_estimate_aud, license_type, status, ts "
        "FROM ip_registry ORDER BY value_estimate_aud DESC LIMIT 50")
    total = _scalar(
        "SELECT COALESCE(SUM(value_estimate_aud),0) FROM ip_registry")
    return {"portfolio": rows, "total_aud": round(float(total), 2),
            "count": len(rows), "ts": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/api/compliance")
def compliance():
    proposals = _db(
        "SELECT prospect, tier, ask_aud, status, ts "
        "FROM negotiation_log ORDER BY ts DESC LIMIT 20")
    pipeline  = _db(
        "SELECT prospect, tier, ask_aud, sent, ts "
        "FROM proposals ORDER BY ts DESC LIMIT 10")
    return {
        "negotiation_log": proposals,
        "proposal_pipeline": pipeline,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/log/recent")
def log_recent():
    rows = _db(
        "SELECT source, event, priority, ts "
        "FROM pulse_log ORDER BY ts DESC LIMIT 100")
    return {"logs": rows, "count": len(rows),
            "ts": datetime.now(tz=timezone.utc).isoformat()}


# ── HTML dashboard ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FractalMesh Omni Nexus</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#00ff88;font-family:'Courier New',monospace;
     font-size:13px;padding:16px}
h1{color:#00ffcc;font-size:18px;margin-bottom:4px}
.sub{color:#666;font-size:11px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{background:#111;border:1px solid #1a3a2a;border-radius:6px;padding:12px}
.card h2{color:#00ff88;font-size:12px;text-transform:uppercase;
         letter-spacing:2px;margin-bottom:8px;border-bottom:1px solid #1a3a2a;
         padding-bottom:4px}
.stat{display:flex;justify-content:space-between;margin:4px 0}
.val{color:#fff;font-weight:bold}
.ok{color:#00ff88}.warn{color:#ffaa00}.err{color:#ff4444}
pre{font-size:11px;overflow:auto;max-height:200px;color:#aaa;
    background:#0d0d0d;padding:8px;border-radius:4px;white-space:pre-wrap}
.refresh{position:fixed;top:12px;right:16px;color:#333;font-size:11px}
table{width:100%;border-collapse:collapse;font-size:11px}
th{color:#00ffcc;text-align:left;padding:2px 6px;border-bottom:1px solid #1a3a2a}
td{padding:2px 6px;border-bottom:1px solid #0d1a12;color:#ccc}
a{color:#00ff88;text-decoration:none}
</style>
</head>
<body>
<h1>⬡ FractalMesh Omni Nexus v2.1.0</h1>
<div class="sub">Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader &nbsp;|&nbsp;
  <a href="https://fractalmesh.net" target="_blank">fractalmesh.net</a></div>
<div class="refresh" id="countdown">next refresh in 30s</div>
<div class="grid" id="root">Loading…</div>

<script>
const API = ["stats","agents","affiliates","fdi","guardrails","ip","compliance","log/recent"];
let countdown = 30;

async function load() {
  const results = await Promise.allSettled(
    API.map(e => fetch("/api/" + e).then(r => r.json()))
  );
  const [stats, agents, affiliates, fdi, guardrails, ip, compliance, logs] =
    results.map(r => r.status === "fulfilled" ? r.value : {});
  render(stats, agents, affiliates, fdi, guardrails, ip, compliance, logs);
}

function stat(label, value, cls="") {
  return `<div class="stat"><span>${label}</span><span class="val ${cls}">${value}</span></div>`;
}

function render(st, ag, af, fd, gr, ip, co, lg) {
  const ts = new Date().toLocaleTimeString();
  let html = "";

  // Business stats
  html += `<div class="card"><h2>Business Metrics</h2>
    ${stat("Revenue", "A$" + (st.revenue_aud||0).toFixed(2))}
    ${stat("Orders", st.orders||0)}
    ${stat("Leads", st.leads||0)}
    ${stat("Conversion", (st.conversion_pct||0) + "%")}
    ${stat("Products active", st.products_active||0)}
    ${stat("IP portfolio", "A$" + (st.ip_portfolio_aud||0).toLocaleString())}
    ${stat("φ score", st.phi_score||0, "ok")}
  </div>`;

  // Affiliates
  html += `<div class="card"><h2>Affiliates</h2>
    ${stat("Programs", st.affiliate_programs||0)}
    ${stat("Clicks / 24h", st.affiliate_clicks_24h||0)}
    ${stat("Earned", "A$" + (st.affiliate_earned_aud||0).toFixed(2), "ok")}
    ${stat("Pending", "A$" + (st.affiliate_pending_aud||0).toFixed(2), "warn")}
    ${stat("Content pieces", st.content_pieces||0)}
    ${stat("Drip sequences", st.drip_active||0)}
  </div>`;

  // Agents
  const agList = (ag.agents||[]).slice(0,12);
  html += `<div class="card"><h2>PM2 Agents (${ag.count||0})</h2>
    <table><tr><th>Name</th><th>Status</th><th>CPU</th><th>MB</th></tr>
    ${agList.map(a => `<tr>
      <td>${a.name}</td>
      <td class="${a.status==="online"?"ok":"err"}">${a.status}</td>
      <td>${a.cpu}%</td><td>${a.mem_mb}</td></tr>`).join("")}
    </table></div>`;

  // FDI (satellite intelligence)
  html += `<div class="card"><h2>Satellite Intelligence</h2>
    ${stat("Methane anomalies", st.methane_anomalies||0, (st.methane_anomalies||0)>0?"warn":"")}
    ${stat("AIS open alerts", st.ais_open_alerts||0, (st.ais_open_alerts||0)>0?"warn":"")}
    ${stat("Reports staged", (fd.recent_reports||[]).length)}
    <div style="margin-top:8px;font-size:11px;color:#666">
      ${(fd.recent_reports||[]).slice(0,3).map(r =>
        `<div>${r.report_ref||""} — A$${(r.amount_aud||0).toLocaleString()}</div>`
      ).join("")}
    </div>
  </div>`;

  // Vault / guardrails
  const live = gr.live_features||[];
  const present = Object.entries(gr.vault_keys||{}).filter(([,v])=>v).length;
  const total   = Object.keys(gr.vault_keys||{}).length;
  html += `<div class="card"><h2>Vault &amp; Guardrails</h2>
    ${stat("Keys present", present + "/" + total, present > 5 ? "ok" : "warn")}
    ${stat("Live features", live.length)}
    <div style="margin-top:6px;font-size:10px;color:#555">
      ${live.map(f=>`<span class="ok">${f}</span> `).join("")}
    </div>
  </div>`;

  // IP registry
  html += `<div class="card"><h2>IP Registry</h2>
    ${stat("Assets", ip.count||0)}
    ${stat("Total value", "A$" + (ip.total_aud||0).toLocaleString())}
    <div style="margin-top:8px">
    ${(ip.portfolio||[]).slice(0,5).map(a =>
      `<div class="stat"><span style="color:#aaa">${(a.title||"").slice(0,28)}</span>
       <span class="val">A$${(a.value_estimate_aud||0).toLocaleString()}</span></div>`
    ).join("")}
    </div>
  </div>`;

  // Logs
  html += `<div class="card"><h2>Recent Log</h2>
    <pre>${(lg.logs||[]).slice(0,20).map(l =>
      `[${(l.ts||"").slice(11,19)}] ${l.source||""}: ${l.event||""}`
    ).join("\\n") || "no logs"}</pre>
  </div>`;

  document.getElementById("root").innerHTML = html;
}

function tick() {
  countdown--;
  document.getElementById("countdown").textContent = "next refresh in " + countdown + "s";
  if (countdown <= 0) { countdown = 30; load(); }
}

load();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=HTML)


if __name__ == "__main__":
    print(f"[fm-omni-nexus] Starting on :{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
