#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Business Reporting Engine
Port: 7873

Automated business reporting system. Generates daily/weekly/monthly reports
by querying data from across sovereign.db, formats them as JSON and HTML,
schedules delivery via email, and stores report history.

Author : Samuel James Hiotis | ABN 56 628 117 363
System : FractalMesh SaaS Platform
"""

# ---------------------------------------------------------------------------
# Vault loading — MUST run before any os.getenv calls
# ---------------------------------------------------------------------------
import os
from pathlib import Path

_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import base64
import collections
import gzip
import hashlib
import io
import json
import math
import sqlite3
import statistics
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT               = int(os.getenv("REPORTING_ENGINE_PORT", "7873"))
SENDGRID_API_KEY   = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "")
ADMIN_SECRET       = os.getenv("ADMIN_SECRET", "")

ROOT    = Path.home() / "fmsaas"
DB_PATH = ROOT / "database" / "sovereign.db"
LOG_PATH = ROOT / "logs" / "reporting_engine.log"

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()
_last_report_at: float = 0.0
_last_report_name: str = ""
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %Human:%M:%S", time.gmtime())
    entry = f"[{ts}] [{level}] {msg}\n"
    try:
        with open(str(LOG_PATH), "a") as fh:
            fh.write(entry)
    except OSError:
        pass


def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = f"[{ts}] [{level}] {msg}\n"
    try:
        with open(str(LOG_PATH), "a") as fh:
            fh.write(entry)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS report_templates (
                id           INTEGER PRIMARY KEY,
                name         TEXT UNIQUE NOT NULL,
                report_type  TEXT NOT NULL,
                schedule     TEXT NOT NULL,
                recipients   TEXT,
                last_run_at  REAL,
                next_run_at  REAL,
                enabled      INTEGER DEFAULT 1,
                created_at   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS report_runs (
                id           INTEGER PRIMARY KEY,
                template_id  INTEGER,
                report_name  TEXT NOT NULL,
                period_start REAL,
                period_end   REAL,
                status       TEXT DEFAULT 'pending',
                output_json  TEXT,
                output_html  TEXT,
                sent_to      TEXT,
                generated_at REAL,
                sent_at      REAL
            );

            CREATE INDEX IF NOT EXISTS idx_report_runs_template
                ON report_runs(template_id);
            CREATE INDEX IF NOT EXISTS idx_report_runs_generated
                ON report_runs(generated_at);
            CREATE INDEX IF NOT EXISTS idx_templates_next_run
                ON report_templates(next_run_at, enabled);
        """)
    _seed_templates()


def _seed_templates() -> None:
    now = time.time()
    seeds = [
        {
            "name": "daily_revenue",
            "report_type": "revenue_daily",
            "schedule": "daily",
            "recipients": "",
        },
        {
            "name": "weekly_kpi",
            "report_type": "kpi_dashboard",
            "schedule": "weekly",
            "recipients": "",
        },
        {
            "name": "agent_health_daily",
            "report_type": "agent_health",
            "schedule": "daily",
            "recipients": "",
        },
    ]
    with get_db() as conn:
        existing = {r["name"] for r in conn.execute("SELECT name FROM report_templates")}
        for s in seeds:
            if s["name"] not in existing:
                next_run = _compute_next_run(s["schedule"], now)
                conn.execute(
                    """INSERT INTO report_templates
                       (name, report_type, schedule, recipients, last_run_at,
                        next_run_at, enabled, created_at)
                       VALUES (?,?,?,?,NULL,?,1,?)""",
                    (s["name"], s["report_type"], s["schedule"],
                     s["recipients"], next_run, now),
                )

# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
def _compute_next_run(schedule: str, from_ts: float) -> float:
    """Return the next run timestamp after from_ts for a given schedule."""
    if schedule == "daily":
        return from_ts + 86400
    elif schedule == "weekly":
        return from_ts + 7 * 86400
    elif schedule == "monthly":
        return from_ts + 30 * 86400
    return from_ts + 86400


def _compute_period(schedule: str, last_run_at: float, now: float):
    """Return (period_start, period_end) for a report run."""
    if schedule == "daily":
        period_end = now
        period_start = now - 86400
    elif schedule == "weekly":
        period_end = now
        period_start = now - 7 * 86400
    elif schedule == "monthly":
        period_end = now
        period_start = now - 30 * 86400
    else:
        period_end = now
        period_start = now - 86400
    return period_start, period_end

# ---------------------------------------------------------------------------
# Query functions — one per report type
# ---------------------------------------------------------------------------

def _query_revenue_daily(conn: sqlite3.Connection, period_start: float, period_end: float) -> dict:
    data: dict = {}

    # Today's payment stats from `payments` table
    try:
        rows = conn.execute(
            """SELECT currency,
                      COUNT(*) as count,
                      SUM(amount) as total,
                      AVG(amount) as avg_amount
               FROM payments
               WHERE created_at BETWEEN ? AND ?
               GROUP BY currency""",
            (period_start, period_end),
        ).fetchall()
        by_currency = []
        total_all = 0.0
        total_count = 0
        for r in rows:
            amt = float(r["total"] or 0)
            cnt = int(r["count"] or 0)
            total_all += amt
            total_count += cnt
            by_currency.append({
                "currency": r["currency"],
                "count": cnt,
                "total": round(amt, 2),
                "avg_order_value": round(float(r["avg_amount"] or 0), 2),
            })
        data["payments"] = {
            "total_revenue": round(total_all, 2),
            "total_count": total_count,
            "avg_order_value": round(total_all / total_count, 2) if total_count else 0,
            "by_currency": by_currency,
        }
    except Exception as exc:
        data["payments"] = {"error": str(exc)}

    # Active subscriptions
    try:
        active = conn.execute(
            "SELECT COUNT(*) as cnt FROM subscriptions WHERE status='active'"
        ).fetchone()
        data["active_subscriptions"] = int(active["cnt"]) if active else 0
    except Exception as exc:
        data["active_subscriptions"] = {"error": str(exc)}

    # Latest MRR snapshot
    try:
        mrr_row = conn.execute(
            "SELECT mrr, recorded_at FROM mrr_snapshots ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
        data["mrr"] = {
            "value": float(mrr_row["mrr"]) if mrr_row else 0,
            "recorded_at": float(mrr_row["recorded_at"]) if mrr_row else None,
        }
    except Exception as exc:
        data["mrr"] = {"error": str(exc)}

    return data


def _query_revenue_weekly(conn: sqlite3.Connection, period_start: float, period_end: float) -> dict:
    data: dict = {}

    # 7-day total and daily breakdown
    try:
        rows = conn.execute(
            """SELECT DATE(created_at, 'unixepoch') as day,
                      SUM(amount) as daily_total
               FROM payments
               WHERE created_at BETWEEN ? AND ?
               GROUP BY day
               ORDER BY day""",
            (period_start, period_end),
        ).fetchall()
        daily_breakdown = [{"date": r["day"], "amount": round(float(r["daily_total"] or 0), 2)} for r in rows]
        week_total = sum(d["amount"] for d in daily_breakdown)
        data["weekly_total"] = round(week_total, 2)
        data["daily_breakdown"] = daily_breakdown
    except Exception as exc:
        data["weekly_total"] = {"error": str(exc)}
        data["daily_breakdown"] = []

    # Top products by revenue (from order_items)
    try:
        top = conn.execute(
            """SELECT product_name, SUM(amount) as rev
               FROM order_items oi
               JOIN payments p ON p.id = oi.payment_id
               WHERE p.created_at BETWEEN ? AND ?
               GROUP BY product_name
               ORDER BY rev DESC
               LIMIT 10""",
            (period_start, period_end),
        ).fetchall()
        data["top_products"] = [
            {"product": r["product_name"], "revenue": round(float(r["rev"] or 0), 2)}
            for r in top
        ]
    except Exception as exc:
        data["top_products"] = {"error": str(exc)}

    # Week-over-week growth %
    try:
        prev_start = period_start - 7 * 86400
        prev_end = period_start
        prev_row = conn.execute(
            "SELECT SUM(amount) as total FROM payments WHERE created_at BETWEEN ? AND ?",
            (prev_start, prev_end),
        ).fetchone()
        prev_total = float(prev_row["total"] or 0) if prev_row else 0
        curr_total = float(data.get("weekly_total", 0)) if isinstance(data.get("weekly_total"), (int, float)) else 0
        if prev_total > 0:
            wow_growth = round((curr_total - prev_total) / prev_total * 100, 2)
        else:
            wow_growth = None
        data["wow_growth_pct"] = wow_growth
    except Exception as exc:
        data["wow_growth_pct"] = {"error": str(exc)}

    # New subscriptions this week
    try:
        new_subs = conn.execute(
            "SELECT COUNT(*) as cnt FROM subscriptions WHERE created_at BETWEEN ? AND ?",
            (period_start, period_end),
        ).fetchone()
        data["new_subscriptions"] = int(new_subs["cnt"]) if new_subs else 0
    except Exception as exc:
        data["new_subscriptions"] = {"error": str(exc)}

    return data


def _query_kpi_dashboard(conn: sqlite3.Connection, period_start: float, period_end: float) -> dict:
    data: dict = {}

    # Latest KPI snapshot values per metric
    try:
        rows = conn.execute(
            """SELECT metric_name, value, recorded_at
               FROM kpi_snapshots k1
               WHERE recorded_at = (
                   SELECT MAX(recorded_at) FROM kpi_snapshots k2
                   WHERE k2.metric_name = k1.metric_name
               )"""
        ).fetchall()
        data["kpis"] = [
            {"metric": r["metric_name"], "value": r["value"], "recorded_at": r["recorded_at"]}
            for r in rows
        ]
    except Exception as exc:
        data["kpis"] = {"error": str(exc)}

    # Support tickets: open/resolved today
    try:
        today_start = period_end - 86400
        open_t = conn.execute(
            "SELECT COUNT(*) as cnt FROM tickets WHERE status='open' AND created_at >= ?",
            (today_start,),
        ).fetchone()
        resolved_t = conn.execute(
            "SELECT COUNT(*) as cnt FROM tickets WHERE status='resolved' AND updated_at >= ?",
            (today_start,),
        ).fetchone()
        data["support_tickets"] = {
            "open_today": int(open_t["cnt"]) if open_t else 0,
            "resolved_today": int(resolved_t["cnt"]) if resolved_t else 0,
        }
    except Exception as exc:
        data["support_tickets"] = {"error": str(exc)}

    # Email campaigns: emails sent this week, open rate
    try:
        sent = conn.execute(
            "SELECT SUM(sent_count) as total_sent, AVG(open_rate) as avg_open FROM campaigns WHERE sent_at BETWEEN ? AND ?",
            (period_start, period_end),
        ).fetchone()
        data["email_campaigns"] = {
            "emails_sent_this_week": int(sent["total_sent"] or 0) if sent else 0,
            "avg_open_rate": round(float(sent["avg_open"] or 0), 4) if sent else 0,
        }
    except Exception as exc:
        data["email_campaigns"] = {"error": str(exc)}

    # CRM deals: pipeline value by stage
    try:
        deal_rows = conn.execute(
            "SELECT stage, SUM(value) as pipeline_value, COUNT(*) as deal_count FROM deals GROUP BY stage"
        ).fetchall()
        data["crm_pipeline"] = [
            {"stage": r["stage"], "pipeline_value": round(float(r["pipeline_value"] or 0), 2), "deal_count": int(r["deal_count"])}
            for r in deal_rows
        ]
    except Exception as exc:
        data["crm_pipeline"] = {"error": str(exc)}

    return data


def _query_agent_health(conn: sqlite3.Connection, period_start: float, period_end: float) -> dict:
    data: dict = {}

    # Agent uptime % for last 24h per agent from watchdog_checks
    try:
        rows = conn.execute(
            """SELECT agent_name,
                      COUNT(*) as total_checks,
                      SUM(CASE WHEN status='up' THEN 1 ELSE 0 END) as up_checks
               FROM watchdog_checks
               WHERE checked_at BETWEEN ? AND ?
               GROUP BY agent_name
               ORDER BY agent_name""",
            (period_start, period_end),
        ).fetchall()
        uptime_list = []
        for r in rows:
            total = int(r["total_checks"])
            up = int(r["up_checks"])
            uptime_pct = round(up / total * 100, 2) if total > 0 else 0
            uptime_list.append({
                "agent": r["agent_name"],
                "uptime_pct": uptime_pct,
                "total_checks": total,
                "up_checks": up,
            })
        data["agent_uptime"] = uptime_list
    except Exception as exc:
        data["agent_uptime"] = {"error": str(exc)}

    # Incidents in last 24h
    try:
        incidents = conn.execute(
            """SELECT agent_name, incident_type, description, started_at, resolved_at
               FROM watchdog_incidents
               WHERE started_at BETWEEN ? AND ?
               ORDER BY started_at DESC""",
            (period_start, period_end),
        ).fetchall()
        data["incidents"] = [
            {
                "agent": r["agent_name"],
                "type": r["incident_type"],
                "description": r["description"],
                "started_at": r["started_at"],
                "resolved_at": r["resolved_at"],
            }
            for r in incidents
        ]
    except Exception as exc:
        data["incidents"] = {"error": str(exc)}

    return data


def _query_custom(conn: sqlite3.Connection, sql: str) -> dict:
    """Run an admin-provided SELECT query."""
    sql_clean = sql.strip()
    # Validate: must start with SELECT and contain no DDL/DML
    upper = sql_clean.upper()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                 "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA"]
    if not upper.startswith("SELECT"):
        raise ValueError("Custom SQL must start with SELECT")
    for kw in forbidden:
        if kw in upper:
            raise ValueError(f"Forbidden keyword in custom SQL: {kw}")
    rows = conn.execute(sql_clean).fetchall()
    cols = [desc[0] for desc in conn.execute(sql_clean).description] if rows else []
    # Re-run to get column names properly
    cursor = conn.execute(sql_clean)
    cols = [d[0] for d in cursor.description]
    result_rows = cursor.fetchall()
    return {
        "columns": cols,
        "rows": [dict(zip(cols, row)) for row in result_rows],
        "row_count": len(result_rows),
    }

# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------
_HTML_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f4f6f9; color: #333; }
  .header { background: #1a1a2e; color: #fff; padding: 28px 40px; }
  .header h1 { font-size: 1.8rem; letter-spacing: 1px; }
  .header .subtitle { font-size: 0.9rem; color: #a0aec0; margin-top: 4px; }
  .header .badge { display: inline-block; background: #e94560; color: #fff;
                   font-size: 0.7rem; padding: 2px 8px; border-radius: 12px;
                   margin-left: 10px; vertical-align: middle; }
  .container { max-width: 960px; margin: 0 auto; padding: 32px 20px; }
  .section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
             margin-bottom: 24px; overflow: hidden; }
  .section-title { background: #16213e; color: #e2e8f0; padding: 14px 20px;
                   font-size: 0.95rem; font-weight: 600; letter-spacing: 0.5px; }
  .section-body { padding: 20px; }
  .metric-grid { display: flex; flex-wrap: wrap; gap: 16px; }
  .metric-card { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px;
                 padding: 16px 20px; min-width: 160px; flex: 1; }
  .metric-card .label { font-size: 0.75rem; text-transform: uppercase; color: #718096;
                        letter-spacing: 0.5px; margin-bottom: 6px; }
  .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
  .metric-card .value.good { color: #38a169; }
  .metric-card .value.warn { color: #d69e2e; }
  .metric-card .value.bad  { color: #e53e3e; }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  th { background: #edf2f7; color: #4a5568; font-weight: 600; text-align: left;
       padding: 10px 14px; border-bottom: 2px solid #e2e8f0; }
  td { padding: 9px 14px; border-bottom: 1px solid #f0f4f8; }
  tr:nth-child(even) td { background: #f7fafc; }
  tr:last-child td { border-bottom: none; }
  .footer { text-align: center; padding: 24px; color: #a0aec0; font-size: 0.8rem; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
          font-size: 0.75rem; font-weight: 600; }
  .pill-green { background: #c6f6d5; color: #276749; }
  .pill-red   { background: #fed7d7; color: #9b2c2c; }
  .pill-blue  { background: #bee3f8; color: #2c5282; }
  .bar-wrap { background: #e2e8f0; border-radius: 4px; height: 8px; width: 100%; }
  .bar-fill { background: #4299e1; border-radius: 4px; height: 8px; }
</style>
"""


def _fmt_currency(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return str(v)


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1f}%"
    except Exception:
        return str(v)


def _html_escape(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _metric_card(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="metric-card">'
            f'<div class="label">{_html_escape(label)}</div>'
            f'<div class="value {cls}">{_html_escape(value)}</div>'
            f'</div>')


def _table_from_list(rows: list, columns: list = None) -> str:
    if not rows:
        return "<p style='color:#718096;font-size:0.85rem;'>No data available.</p>"
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    html = "<table><thead><tr>"
    for col in columns:
        html += f"<th>{_html_escape(str(col))}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"
        for col in columns:
            html += f"<td>{_html_escape(str(row.get(col, '')))}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def _build_html(report_name: str, report_type: str, period_start: float,
                period_end: float, data: dict, generated_at: float) -> str:
    period_s = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(period_start))
    period_e = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(period_end))
    gen_ts   = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(generated_at))

    sections_html = ""

    if report_type == "revenue_daily":
        payments = data.get("payments", {})
        if isinstance(payments, dict) and "error" not in payments:
            cards = _metric_card("Total Revenue", _fmt_currency(payments.get("total_revenue", 0)), "good")
            cards += _metric_card("Transaction Count", str(payments.get("total_count", 0)))
            cards += _metric_card("Avg Order Value", _fmt_currency(payments.get("avg_order_value", 0)))
            cards += _metric_card("Active Subscriptions", str(data.get("active_subscriptions", 0)), "good")
            mrr = data.get("mrr", {})
            mrr_val = mrr.get("value", 0) if isinstance(mrr, dict) else 0
            cards += _metric_card("MRR", _fmt_currency(mrr_val))
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Revenue Summary</div>'
                f'<div class="section-body"><div class="metric-grid">{cards}</div></div>'
                f'</div>'
            )
            by_curr = payments.get("by_currency", [])
            if by_curr:
                sections_html += (
                    f'<div class="section">'
                    f'<div class="section-title">Revenue by Currency</div>'
                    f'<div class="section-body">{_table_from_list(by_curr, ["currency","count","total","avg_order_value"])}</div>'
                    f'</div>'
                )

    elif report_type == "revenue_weekly":
        total = data.get("weekly_total", 0)
        wow   = data.get("wow_growth_pct")
        new_s = data.get("new_subscriptions", 0)
        wow_str = _fmt_pct(wow) if wow is not None else "N/A"
        wow_cls = "good" if isinstance(wow, (int, float)) and wow >= 0 else "bad"
        cards = _metric_card("7-Day Revenue", _fmt_currency(total), "good")
        cards += _metric_card("WoW Growth", wow_str, wow_cls)
        cards += _metric_card("New Subscriptions", str(new_s))
        sections_html += (
            f'<div class="section">'
            f'<div class="section-title">Weekly Revenue Summary</div>'
            f'<div class="section-body"><div class="metric-grid">{cards}</div></div>'
            f'</div>'
        )
        daily = data.get("daily_breakdown", [])
        if daily:
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Daily Breakdown</div>'
                f'<div class="section-body">{_table_from_list(daily, ["date","amount"])}</div>'
                f'</div>'
            )
        top_prod = data.get("top_products", [])
        if top_prod and not isinstance(top_prod, dict):
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Top Products by Revenue</div>'
                f'<div class="section-body">{_table_from_list(top_prod, ["product","revenue"])}</div>'
                f'</div>'
            )

    elif report_type == "kpi_dashboard":
        kpis = data.get("kpis", [])
        if kpis and isinstance(kpis, list):
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">KPI Snapshot</div>'
                f'<div class="section-body">{_table_from_list(kpis, ["metric","value","recorded_at"])}</div>'
                f'</div>'
            )
        support = data.get("support_tickets", {})
        if isinstance(support, dict) and "error" not in support:
            cards = _metric_card("Tickets Open Today", str(support.get("open_today", 0)), "warn")
            cards += _metric_card("Tickets Resolved Today", str(support.get("resolved_today", 0)), "good")
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Support Desk</div>'
                f'<div class="section-body"><div class="metric-grid">{cards}</div></div>'
                f'</div>'
            )
        ec = data.get("email_campaigns", {})
        if isinstance(ec, dict) and "error" not in ec:
            cards = _metric_card("Emails Sent This Week", str(ec.get("emails_sent_this_week", 0)))
            cards += _metric_card("Avg Open Rate", _fmt_pct(float(ec.get("avg_open_rate", 0)) * 100))
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Email Campaigns</div>'
                f'<div class="section-body"><div class="metric-grid">{cards}</div></div>'
                f'</div>'
            )
        pipeline = data.get("crm_pipeline", [])
        if pipeline and isinstance(pipeline, list):
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">CRM Pipeline by Stage</div>'
                f'<div class="section-body">{_table_from_list(pipeline, ["stage","deal_count","pipeline_value"])}</div>'
                f'</div>'
            )

    elif report_type == "agent_health":
        uptime = data.get("agent_uptime", [])
        if uptime and isinstance(uptime, list):
            # Compute summary stats
            pcts = [u["uptime_pct"] for u in uptime if isinstance(u.get("uptime_pct"), (int, float))]
            avg_uptime = round(statistics.mean(pcts), 2) if pcts else 0
            min_uptime = round(min(pcts), 2) if pcts else 0
            agents_down = sum(1 for p in pcts if p < 90)
            cards = _metric_card("Agents Monitored", str(len(uptime)))
            cards += _metric_card("Avg Uptime", _fmt_pct(avg_uptime),
                                  "good" if avg_uptime >= 99 else "warn" if avg_uptime >= 90 else "bad")
            cards += _metric_card("Min Uptime", _fmt_pct(min_uptime),
                                  "good" if min_uptime >= 99 else "warn" if min_uptime >= 90 else "bad")
            cards += _metric_card("Agents Below 90%", str(agents_down), "bad" if agents_down else "good")
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Uptime Summary</div>'
                f'<div class="section-body"><div class="metric-grid">{cards}</div></div>'
                f'</div>'
            )
            # Uptime table with bar
            table_rows_html = ""
            for u in uptime:
                pct = float(u.get("uptime_pct", 0))
                color = "#38a169" if pct >= 99 else "#d69e2e" if pct >= 90 else "#e53e3e"
                bar = (f'<div class="bar-wrap"><div class="bar-fill" '
                       f'style="width:{pct}%;background:{color};"></div></div>')
                table_rows_html += (
                    f"<tr><td>{_html_escape(str(u.get('agent','')))}</td>"
                    f"<td>{_html_escape(str(u.get('total_checks','')))}</td>"
                    f"<td>{_html_escape(str(u.get('up_checks','')))}</td>"
                    f"<td>{_html_escape(_fmt_pct(pct))}&nbsp;{bar}</td></tr>"
                )
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Per-Agent Uptime (Last 24h)</div>'
                f'<div class="section-body">'
                f'<table><thead><tr><th>Agent</th><th>Checks</th><th>Up</th><th>Uptime %</th></tr></thead>'
                f'<tbody>{table_rows_html}</tbody></table>'
                f'</div></div>'
            )
        incidents = data.get("incidents", [])
        if incidents and isinstance(incidents, list):
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Incidents (Last 24h)</div>'
                f'<div class="section-body">'
                f'{_table_from_list(incidents, ["agent","type","description","started_at","resolved_at"])}'
                f'</div></div>'
            )
        elif isinstance(incidents, list):
            sections_html += (
                f'<div class="section">'
                f'<div class="section-title">Incidents (Last 24h)</div>'
                f'<div class="section-body"><p style="color:#38a169;font-weight:600;">No incidents in the last 24 hours.</p>'
                f'</div></div>'
            )

    elif report_type == "custom":
        cols = data.get("columns", [])
        rows_data = data.get("rows", [])
        row_count = data.get("row_count", 0)
        sections_html += (
            f'<div class="section">'
            f'<div class="section-title">Custom Query Results ({row_count} rows)</div>'
            f'<div class="section-body">{_table_from_list(rows_data, cols)}</div>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FractalMesh Report — {_html_escape(report_name)}</title>
{_HTML_CSS}
</head>
<body>
<div class="header">
  <h1>FractalMesh OMEGA Titan <span class="badge">REPORT</span></h1>
  <div class="subtitle">{_html_escape(report_name)} &bull; {_html_escape(report_type.replace("_"," ").title())}</div>
</div>
<div class="container">
  <div class="section">
    <div class="section-title">Report Period</div>
    <div class="section-body">
      <div class="metric-grid">
        {_metric_card("Period Start", period_s)}
        {_metric_card("Period End", period_e)}
        {_metric_card("Generated At", gen_ts)}
        {_metric_card("Report Type", report_type)}
      </div>
    </div>
  </div>
  {sections_html}
</div>
<div class="footer">
  Generated by FractalMesh OMEGA Titan &mdash; Business Reporting Engine &bull; Port 7873<br>
  &copy; {time.strftime("%Y", time.gmtime())} Samuel James Hiotis | ABN 56 628 117 363
</div>
</body>
</html>"""
    return html

# ---------------------------------------------------------------------------
# Email delivery via SendGrid
# ---------------------------------------------------------------------------
def _send_report_email(recipients: list, subject: str, html_body: str) -> bool:
    """Send HTML report via SendGrid. Returns True on success."""
    api_key = SENDGRID_API_KEY
    from_email = SENDGRID_FROM_EMAIL
    if not api_key or not from_email:
        _log("WARN", "SendGrid not configured — skipping email delivery")
        return False
    if not recipients:
        return False

    payload = {
        "personalizations": [{"to": [{"email": r} for r in recipients]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 202)
    except urllib.error.HTTPError as exc:
        _log("ERROR", f"SendGrid HTTP {exc.code}: {exc.read()[:200]}")
        return False
    except Exception as exc:
        _log("ERROR", f"SendGrid error: {exc}")
        return False

# ---------------------------------------------------------------------------
# Core report generation
# ---------------------------------------------------------------------------
def _generate_report(template: dict) -> int:
    """Generate a report run for a template. Returns run ID."""
    global _last_report_at, _last_report_name

    schedule     = template["schedule"]
    report_type  = template["report_type"]
    template_id  = template["id"]
    report_name  = template["name"]
    recipients_raw = template.get("recipients") or ""
    recipients   = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    last_run_at  = template.get("last_run_at")
    now = time.time()
    period_start, period_end = _compute_period(schedule, last_run_at, now)

    # Insert pending run
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO report_runs
               (template_id, report_name, period_start, period_end, status, generated_at)
               VALUES (?,?,?,?,'pending',?)""",
            (template_id, report_name, period_start, period_end, now),
        )
        run_id = cur.lastrowid

    try:
        with get_db() as conn:
            if report_type == "revenue_daily":
                data = _query_revenue_daily(conn, period_start, period_end)
            elif report_type == "revenue_weekly":
                data = _query_revenue_weekly(conn, period_start, period_end)
            elif report_type == "kpi_dashboard":
                data = _query_kpi_dashboard(conn, period_start, period_end)
            elif report_type == "agent_health":
                data = _query_agent_health(conn, period_start, period_end)
            else:
                raise ValueError(f"Unknown report_type: {report_type}")

        output_json = json.dumps(data, default=str)
        output_html = _build_html(report_name, report_type, period_start,
                                  period_end, data, now)

        sent_to = None
        sent_at = None
        if recipients:
            subject = (f"FractalMesh Report: {report_name} — "
                       f"{time.strftime('%Y-%m-%d', time.gmtime(now))}")
            ok = _send_report_email(recipients, subject, output_html)
            if ok:
                sent_to = ",".join(recipients)
                sent_at = time.time()

        with get_db() as conn:
            conn.execute(
                """UPDATE report_runs
                   SET status='completed', output_json=?, output_html=?,
                       sent_to=?, sent_at=?
                   WHERE id=?""",
                (output_json, output_html, sent_to, sent_at, run_id),
            )
            conn.execute(
                """UPDATE report_templates
                   SET last_run_at=?, next_run_at=?
                   WHERE id=?""",
                (now, _compute_next_run(schedule, now), template_id),
            )

        with _lock:
            _last_report_at = now
            _last_report_name = report_name

        _log("INFO", f"Report generated: {report_name} (run #{run_id})")

    except Exception as exc:
        _log("ERROR", f"Report generation failed for {report_name}: {exc}")
        with get_db() as conn:
            conn.execute(
                "UPDATE report_runs SET status='failed' WHERE id=?",
                (run_id,),
            )

    return run_id

# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------
def _scheduler_loop() -> None:
    while True:
        try:
            now = time.time()
            with get_db() as conn:
                due = conn.execute(
                    """SELECT id, name, report_type, schedule, recipients,
                              last_run_at, next_run_at, enabled
                       FROM report_templates
                       WHERE enabled=1 AND next_run_at <= ?""",
                    (now,),
                ).fetchall()
            for row in due:
                tmpl = dict(row)
                _log("INFO", f"Scheduler triggering: {tmpl['name']}")
                _generate_report(tmpl)
        except Exception as exc:
            _log("ERROR", f"Scheduler loop error: {exc}")
        time.sleep(300)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _is_admin(headers) -> bool:
    secret = ADMIN_SECRET
    if not secret:
        return False
    auth = headers.get("X-Admin-Secret") or headers.get("Authorization", "")
    return auth == secret or auth == f"Bearer {secret}"

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class ReportingHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        _log("HTTP", fmt % args)

    def _send_json(self, code: int, payload) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        query_str = self.path.split("?")[1] if "?" in self.path else ""
        params: dict = {}
        if query_str:
            for part in query_str.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v

        # GET /health
        if path == "/health":
            with _lock:
                lr_at   = _last_report_at
                lr_name = _last_report_name
            with get_db() as conn:
                tmpl_count = conn.execute("SELECT COUNT(*) FROM report_templates WHERE enabled=1").fetchone()[0]
            self._send_json(200, {
                "status": "ok",
                "service": "fm_reporting_engine",
                "port": PORT,
                "uptime_seconds": round(time.time() - START_TIME, 2),
                "enabled_templates": tmpl_count,
                "last_report_name": lr_name or None,
                "last_report_at": lr_at or None,
            })
            return

        # GET /templates
        if path == "/templates":
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM report_templates ORDER BY created_at"
                ).fetchall()
            self._send_json(200, {"templates": [dict(r) for r in rows]})
            return

        # GET /templates/{name}
        if path.startswith("/templates/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                name = parts[2]
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                if not row:
                    self._send_json(404, {"error": "Template not found"})
                    return
                self._send_json(200, dict(row))
                return

        # GET /reports
        if path == "/reports":
            template_id = params.get("template_id")
            limit = int(params.get("limit", "50"))
            limit = min(limit, 200)
            with get_db() as conn:
                if template_id:
                    rows = conn.execute(
                        """SELECT id, template_id, report_name, period_start, period_end,
                                  status, sent_to, generated_at, sent_at
                           FROM report_runs WHERE template_id=?
                           ORDER BY generated_at DESC LIMIT ?""",
                        (template_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT id, template_id, report_name, period_start, period_end,
                                  status, sent_to, generated_at, sent_at
                           FROM report_runs
                           ORDER BY generated_at DESC LIMIT ?""",
                        (limit,),
                    ).fetchall()
            self._send_json(200, {"runs": [dict(r) for r in rows]})
            return

        # GET /reports/{id}/html  or  GET /reports/{id}
        if path.startswith("/reports/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2].isdigit():
                run_id = int(parts[2])
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT * FROM report_runs WHERE id=?", (run_id,)
                    ).fetchone()
                if not row:
                    self._send_json(404, {"error": "Report run not found"})
                    return
                r = dict(row)
                # Return JSON without HTML blob for detail view
                r_out = {k: v for k, v in r.items() if k != "output_html"}
                if r.get("output_json"):
                    try:
                        r_out["output_json"] = json.loads(r["output_json"])
                    except Exception:
                        r_out["output_json"] = r["output_json"]
                self._send_json(200, r_out)
                return

            if len(parts) == 4 and parts[2].isdigit() and parts[3] == "html":
                run_id = int(parts[2])
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT output_html FROM report_runs WHERE id=?", (run_id,)
                    ).fetchone()
                if not row or not row["output_html"]:
                    self._send_json(404, {"error": "HTML report not available"})
                    return
                self._send_html(200, row["output_html"])
                return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        # POST /templates — create template (admin)
        if path == "/templates":
            if not _is_admin(self.headers):
                self._send_json(403, {"error": "Forbidden"})
                return
            body = self._read_body()
            name = (body.get("name") or "").strip()
            report_type = (body.get("report_type") or "").strip()
            schedule = (body.get("schedule") or "daily").strip()
            recipients = (body.get("recipients") or "").strip()
            enabled = int(body.get("enabled", 1))
            if not name or not report_type:
                self._send_json(400, {"error": "name and report_type required"})
                return
            valid_types = {"revenue_daily", "revenue_weekly", "kpi_dashboard", "agent_health", "custom"}
            if report_type not in valid_types:
                self._send_json(400, {"error": f"Invalid report_type. Must be one of: {', '.join(valid_types)}"})
                return
            valid_schedules = {"daily", "weekly", "monthly"}
            if schedule not in valid_schedules:
                self._send_json(400, {"error": f"Invalid schedule. Must be one of: {', '.join(valid_schedules)}"})
                return
            now = time.time()
            try:
                with get_db() as conn:
                    conn.execute(
                        """INSERT INTO report_templates
                           (name, report_type, schedule, recipients,
                            next_run_at, enabled, created_at)
                           VALUES (?,?,?,?,?,?,?)""",
                        (name, report_type, schedule, recipients,
                         _compute_next_run(schedule, now), enabled, now),
                    )
                    row = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                self._send_json(201, dict(row))
            except sqlite3.IntegrityError:
                self._send_json(409, {"error": "Template name already exists"})
            return

        # POST /templates/{name}/run — trigger immediately (admin)
        if path.startswith("/templates/") and path.endswith("/run"):
            if not _is_admin(self.headers):
                self._send_json(403, {"error": "Forbidden"})
                return
            parts = path.split("/")
            if len(parts) == 4:
                name = parts[2]
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                if not row:
                    self._send_json(404, {"error": "Template not found"})
                    return
                tmpl = dict(row)
                # Check if custom and sql provided
                body = self._read_body()
                if tmpl["report_type"] == "custom":
                    sql = body.get("sql", "")
                    if not sql:
                        self._send_json(400, {"error": "sql required for custom report_type"})
                        return
                    # Run custom inline
                    now = time.time()
                    period_start, period_end = _compute_period(tmpl["schedule"], tmpl.get("last_run_at"), now)
                    with get_db() as conn:
                        cur = conn.execute(
                            """INSERT INTO report_runs
                               (template_id, report_name, period_start, period_end, status, generated_at)
                               VALUES (?,?,?,?,'pending',?)""",
                            (tmpl["id"], tmpl["name"], period_start, period_end, now),
                        )
                        run_id = cur.lastrowid
                    try:
                        with get_db() as conn:
                            data = _query_custom(conn, sql)
                        output_json = json.dumps(data, default=str)
                        output_html = _build_html(
                            tmpl["name"], "custom", period_start, period_end, data, now
                        )
                        with get_db() as conn:
                            conn.execute(
                                """UPDATE report_runs
                                   SET status='completed', output_json=?, output_html=?
                                   WHERE id=?""",
                                (output_json, output_html, run_id),
                            )
                        self._send_json(200, {"run_id": run_id, "status": "completed"})
                    except ValueError as exc:
                        with get_db() as conn:
                            conn.execute(
                                "UPDATE report_runs SET status='failed' WHERE id=?",
                                (run_id,),
                            )
                        self._send_json(400, {"error": str(exc)})
                    except Exception as exc:
                        with get_db() as conn:
                            conn.execute(
                                "UPDATE report_runs SET status='failed' WHERE id=?",
                                (run_id,),
                            )
                        self._send_json(500, {"error": str(exc)})
                    return

                # Non-custom: run in background thread, return immediately
                def _run_bg():
                    _generate_report(tmpl)

                t = threading.Thread(target=_run_bg, daemon=True)
                t.start()
                self._send_json(202, {
                    "message": f"Report generation started for '{name}'",
                    "template": name,
                })
                return

        # POST /templates/{name}/schedule — update schedule/recipients (admin)
        if path.startswith("/templates/") and path.endswith("/schedule"):
            if not _is_admin(self.headers):
                self._send_json(403, {"error": "Forbidden"})
                return
            parts = path.split("/")
            if len(parts) == 4:
                name = parts[2]
                body = self._read_body()
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                if not row:
                    self._send_json(404, {"error": "Template not found"})
                    return
                schedule = body.get("schedule", row["schedule"])
                recipients = body.get("recipients", row["recipients"] or "")
                enabled = body.get("enabled", row["enabled"])
                valid_schedules = {"daily", "weekly", "monthly"}
                if schedule not in valid_schedules:
                    self._send_json(400, {"error": "Invalid schedule"})
                    return
                now = time.time()
                with get_db() as conn:
                    conn.execute(
                        """UPDATE report_templates
                           SET schedule=?, recipients=?, enabled=?, next_run_at=?
                           WHERE name=?""",
                        (schedule, recipients, int(enabled),
                         _compute_next_run(schedule, now), name),
                    )
                    updated = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                self._send_json(200, dict(updated))
                return

        self._send_json(404, {"error": "Not found"})

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")

        # PUT /templates/{name} — update template (admin)
        if path.startswith("/templates/"):
            if not _is_admin(self.headers):
                self._send_json(403, {"error": "Forbidden"})
                return
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                name = parts[2]
                body = self._read_body()
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                if not row:
                    self._send_json(404, {"error": "Template not found"})
                    return
                report_type = body.get("report_type", row["report_type"])
                schedule = body.get("schedule", row["schedule"])
                recipients = body.get("recipients", row["recipients"] or "")
                enabled = body.get("enabled", row["enabled"])
                valid_types = {"revenue_daily", "revenue_weekly", "kpi_dashboard", "agent_health", "custom"}
                valid_schedules = {"daily", "weekly", "monthly"}
                if report_type not in valid_types:
                    self._send_json(400, {"error": "Invalid report_type"})
                    return
                if schedule not in valid_schedules:
                    self._send_json(400, {"error": "Invalid schedule"})
                    return
                now = time.time()
                with get_db() as conn:
                    conn.execute(
                        """UPDATE report_templates
                           SET report_type=?, schedule=?, recipients=?,
                               enabled=?, next_run_at=?
                           WHERE name=?""",
                        (report_type, schedule, recipients, int(enabled),
                         _compute_next_run(schedule, now), name),
                    )
                    updated = conn.execute(
                        "SELECT * FROM report_templates WHERE name=?", (name,)
                    ).fetchone()
                self._send_json(200, dict(updated))
                return

        self._send_json(404, {"error": "Not found"})

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    init_db()
    _log("INFO", f"FractalMesh Reporting Engine starting on port {PORT}")

    # Start background scheduler
    sched = threading.Thread(target=_scheduler_loop, daemon=True, name="report-scheduler")
    sched.start()

    server = HTTPServer(("0.0.0.0", PORT), ReportingHandler)
    _log("INFO", f"Reporting Engine listening on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("INFO", "Reporting Engine shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
