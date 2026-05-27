#!/usr/bin/env python3
"""
fm_analytics_hub.py — Business Analytics Hub (Port 7856)
FractalMesh OMEGA Titan | Central BI: KPIs, reports, funnels, cohorts, alerts, anomaly detection.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import math
import hashlib
import sqlite3
import threading
import statistics
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict

# ── vault ──────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("ANALYTICS_HUB_PORT", "7856"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
MCP_SECRET   = os.getenv("MCP_SECRET", "")
MCP_PORT     = int(os.getenv("MCP_PORT", "7785"))

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
MEM_DB = Path.home() / "fmsaas" / "database" / "sovereign_memory.db"

ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ── database ───────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kpi_snapshots (
            id              INTEGER PRIMARY KEY,
            period          TEXT,
            metric_name     TEXT,
            metric_value    REAL,
            dimension       TEXT,
            dimension_value TEXT,
            snapshot_at     REAL
        );
        CREATE INDEX IF NOT EXISTS idx_kpi_metric_at
            ON kpi_snapshots(metric_name, snapshot_at);

        CREATE TABLE IF NOT EXISTS reports (
            id           INTEGER PRIMARY KEY,
            report_type  TEXT,
            period       TEXT,
            content      TEXT,
            generated_at REAL
        );

        CREATE TABLE IF NOT EXISTS funnels (
            id              INTEGER PRIMARY KEY,
            funnel_name     TEXT,
            step_name       TEXT,
            step_order      INTEGER,
            count           INTEGER,
            conversion_rate REAL,
            recorded_at     REAL
        );
        CREATE INDEX IF NOT EXISTS idx_funnel_name
            ON funnels(funnel_name, step_order);

        CREATE TABLE IF NOT EXISTS cohorts (
            id             INTEGER PRIMARY KEY,
            cohort_date    TEXT,
            cohort_size    INTEGER,
            retention_day  INTEGER,
            retained       INTEGER,
            retention_rate REAL
        );
        CREATE INDEX IF NOT EXISTS idx_cohort_date
            ON cohorts(cohort_date, retention_day);

        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY,
            metric_name   TEXT,
            threshold     REAL,
            operator      TEXT,
            current_value REAL,
            triggered_at  REAL,
            resolved_at   REAL,
            message       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_resolved
            ON alerts(resolved_at);
    """)
    conn.commit()
    conn.close()


# ── helpers ────────────────────────────────────────────────────────────────────
def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _today_epoch_range():
    """Return (start_of_day, now) as epoch floats for today UTC."""
    now = time.time()
    # midnight UTC today
    import time as _t
    t = _t.gmtime(now)
    midnight = _t.mktime(_t.strptime(
        f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}", "%Y-%m-%d"
    ))
    # mktime uses local time; adjust for UTC
    local_offset = _t.mktime(_t.gmtime(0)) - _t.mktime(_t.localtime(0))
    # simpler: use calendar
    import calendar
    midnight_utc = calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, 0))
    return float(midnight_utc), now


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


def _rows_to_list(rows) -> list:
    return [dict(r) for r in rows]


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _check_alert_threshold(metric_name: str, current_value: float):
    """Check all alert rules for this metric and trigger if condition met."""
    conn = get_db()
    try:
        rules = conn.execute(
            "SELECT id, threshold, operator, message FROM alerts "
            "WHERE metric_name=? AND resolved_at IS NULL",
            (metric_name,)
        ).fetchall()
        for rule in rules:
            op = rule["operator"]
            thr = rule["threshold"]
            triggered = False
            if op == "gt"  and current_value >  thr: triggered = True
            elif op == "lt"  and current_value <  thr: triggered = True
            elif op == "gte" and current_value >= thr: triggered = True
            elif op == "lte" and current_value <= thr: triggered = True
            if triggered:
                conn.execute(
                    "UPDATE alerts SET current_value=?, triggered_at=? WHERE id=?",
                    (current_value, time.time(), rule["id"])
                )
        conn.commit()
    finally:
        conn.close()


def _anomaly_check(metric_name: str, current_value: float):
    """If current_value > 2 std-devs from 30-day rolling mean, create an alert."""
    conn = get_db()
    try:
        cutoff = time.time() - 30 * 86400
        rows = conn.execute(
            "SELECT metric_value FROM kpi_snapshots "
            "WHERE metric_name=? AND snapshot_at >= ? ORDER BY snapshot_at",
            (metric_name, cutoff)
        ).fetchall()
        values = [r["metric_value"] for r in rows if r["metric_value"] is not None]
        if len(values) < 10:
            return  # not enough data
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        if stdev == 0:
            return
        z = abs(current_value - mean) / stdev
        if z > 2.0:
            msg = (
                f"Anomaly detected: {metric_name}={current_value:.4f} "
                f"is {z:.2f} std-devs from 30d mean {mean:.4f}"
            )
            # only insert if no open anomaly alert for this metric
            existing = conn.execute(
                "SELECT id FROM alerts WHERE metric_name=? AND resolved_at IS NULL "
                "AND message LIKE 'Anomaly%'",
                (metric_name,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO alerts (metric_name, threshold, operator, current_value, "
                    "triggered_at, resolved_at, message) VALUES (?,?,?,?,?,NULL,?)",
                    (metric_name, mean + 2 * stdev, "gt", current_value, time.time(), msg)
                )
                conn.commit()
    finally:
        conn.close()


# ── background KPI thread ──────────────────────────────────────────────────────
def _compute_and_store_kpis():
    conn = get_db()
    now = time.time()
    start_of_day, _ = _today_epoch_range()
    kpis = {}

    try:
        # daily_revenue — payments table
        if _table_exists(conn, "payments"):
            row = conn.execute(
                "SELECT COALESCE(SUM(amount),0) FROM payments "
                "WHERE status='succeeded' AND created_at >= ?",
                (start_of_day,)
            ).fetchone()
            kpis["daily_revenue"] = _safe_float(row[0]) / 100.0
        else:
            kpis["daily_revenue"] = 0.0

        # active_subscriptions
        if _table_exists(conn, "subscriptions"):
            row = conn.execute(
                "SELECT COUNT(*) FROM subscriptions WHERE status='active'"
            ).fetchone()
            kpis["active_subscriptions"] = _safe_float(row[0])
        else:
            kpis["active_subscriptions"] = 0.0

        # total_customers
        if _table_exists(conn, "customers"):
            row = conn.execute("SELECT COUNT(*) FROM customers").fetchone()
            kpis["total_customers"] = _safe_float(row[0])
        else:
            kpis["total_customers"] = 0.0

        # orders_today
        if _table_exists(conn, "orders"):
            row = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE created_at >= ?", (start_of_day,)
            ).fetchone()
            kpis["orders_today"] = _safe_float(row[0])
        else:
            kpis["orders_today"] = 0.0

        # signals_today
        if _table_exists(conn, "signals"):
            row = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE created_at >= ?", (start_of_day,)
            ).fetchone()
            kpis["signals_today"] = _safe_float(row[0])
        else:
            kpis["signals_today"] = 0.0

        # memory_entries — sovereign_memory.db
        kpis["memory_entries"] = 0.0
        if MEM_DB.exists():
            try:
                mconn = sqlite3.connect(str(MEM_DB), timeout=5)
                mconn.execute("PRAGMA journal_mode=WAL")
                if _table_exists(mconn, "memories"):
                    row = mconn.execute("SELECT COUNT(*) FROM memories").fetchone()
                    kpis["memory_entries"] = _safe_float(row[0])
                mconn.close()
            except Exception:
                pass

        # store all KPIs
        for metric_name, metric_value in kpis.items():
            conn.execute(
                "INSERT INTO kpi_snapshots "
                "(period, metric_name, metric_value, dimension, dimension_value, snapshot_at) "
                "VALUES (?,?,?,?,?,?)",
                ("5min", metric_name, metric_value, "all", "all", now)
            )
            _check_alert_threshold(metric_name, metric_value)
            _anomaly_check(metric_name, metric_value)

        conn.commit()
    except Exception as exc:
        print(f"[analytics_hub] KPI compute error: {exc}")
    finally:
        conn.close()


def _generate_daily_report():
    """Generate a daily summary report over the last 24h of kpi_snapshots."""
    conn = get_db()
    now = time.time()
    cutoff = now - 86400
    try:
        rows = conn.execute(
            "SELECT metric_name, metric_value FROM kpi_snapshots "
            "WHERE snapshot_at >= ? AND snapshot_at <= ?",
            (cutoff, now)
        ).fetchall()
        by_metric = defaultdict(list)
        for r in rows:
            by_metric[r["metric_name"]].append(r["metric_value"])

        summary = {}
        for metric, values in by_metric.items():
            if not values:
                continue
            summary[metric] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "avg": round(statistics.mean(values), 4),
                "samples": len(values),
            }

        import time as _t
        t = _t.gmtime(now)
        period_label = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"

        content = json.dumps({"period": period_label, "metrics": summary}, indent=2)
        conn.execute(
            "INSERT INTO reports (report_type, period, content, generated_at) VALUES (?,?,?,?)",
            ("daily", period_label, content, now)
        )
        conn.commit()
    except Exception as exc:
        print(f"[analytics_hub] daily report error: {exc}")
    finally:
        conn.close()


def _kpi_loop():
    """Daemon thread: compute KPIs every 300s, generate daily report at midnight."""
    last_report_day = -1
    while True:
        try:
            _compute_and_store_kpis()
        except Exception as exc:
            print(f"[analytics_hub] kpi_loop error: {exc}")

        # check if midnight crossed
        t = time.gmtime()
        if t.tm_hour == 0 and t.tm_mday != last_report_day:
            try:
                _generate_daily_report()
                last_report_day = t.tm_mday
            except Exception as exc:
                print(f"[analytics_hub] midnight report error: {exc}")

        time.sleep(300)


# ── report generation ──────────────────────────────────────────────────────────
def _generate_report_on_demand(report_type: str, period: str) -> dict:
    conn = get_db()
    now = time.time()
    try:
        if report_type == "weekly":
            window = 7 * 86400
            cutoff_curr = now - window
            cutoff_prev = now - 2 * window

            curr_rows = conn.execute(
                "SELECT metric_name, metric_value FROM kpi_snapshots "
                "WHERE snapshot_at >= ? AND snapshot_at <= ?",
                (cutoff_curr, now)
            ).fetchall()
            prev_rows = conn.execute(
                "SELECT metric_name, metric_value FROM kpi_snapshots "
                "WHERE snapshot_at >= ? AND snapshot_at < ?",
                (cutoff_prev, cutoff_curr)
            ).fetchall()

            curr_by = defaultdict(list)
            prev_by = defaultdict(list)
            for r in curr_rows:
                curr_by[r["metric_name"]].append(r["metric_value"])
            for r in prev_rows:
                prev_by[r["metric_name"]].append(r["metric_value"])

            metrics_summary = {}
            all_metrics = set(curr_by.keys()) | set(prev_by.keys())
            for m in all_metrics:
                curr_avg = statistics.mean(curr_by[m]) if curr_by[m] else 0.0
                prev_avg = statistics.mean(prev_by[m]) if prev_by[m] else 0.0
                if prev_avg != 0:
                    growth = round((curr_avg - prev_avg) / abs(prev_avg) * 100, 2)
                else:
                    growth = 0.0
                metrics_summary[m] = {
                    "current_avg": round(curr_avg, 4),
                    "prior_avg": round(prev_avg, 4),
                    "growth_pct": growth,
                    "samples_current": len(curr_by[m]),
                    "samples_prior": len(prev_by[m]),
                }

            content_dict = {
                "report_type": "weekly",
                "period": period,
                "generated_at": now,
                "metrics": metrics_summary,
            }

        elif report_type == "daily":
            cutoff = now - 86400
            rows = conn.execute(
                "SELECT metric_name, metric_value FROM kpi_snapshots "
                "WHERE snapshot_at >= ?", (cutoff,)
            ).fetchall()
            by_metric = defaultdict(list)
            for r in rows:
                by_metric[r["metric_name"]].append(r["metric_value"])

            summary = {}
            for metric, values in by_metric.items():
                if not values:
                    continue
                summary[metric] = {
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "avg": round(statistics.mean(values), 4),
                    "samples": len(values),
                }
            content_dict = {
                "report_type": "daily",
                "period": period,
                "generated_at": now,
                "metrics": summary,
            }

        else:
            # generic: last 30d
            cutoff = now - 30 * 86400
            rows = conn.execute(
                "SELECT metric_name, metric_value FROM kpi_snapshots "
                "WHERE snapshot_at >= ?", (cutoff,)
            ).fetchall()
            by_metric = defaultdict(list)
            for r in rows:
                by_metric[r["metric_name"]].append(r["metric_value"])
            summary = {}
            for metric, values in by_metric.items():
                if not values:
                    continue
                summary[metric] = {
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "avg": round(statistics.mean(values), 4),
                    "samples": len(values),
                }
            content_dict = {
                "report_type": report_type,
                "period": period,
                "generated_at": now,
                "metrics": summary,
            }

        content_str = json.dumps(content_dict, indent=2)
        cursor = conn.execute(
            "INSERT INTO reports (report_type, period, content, generated_at) VALUES (?,?,?,?)",
            (report_type, period, content_str, now)
        )
        conn.commit()
        report_id = cursor.lastrowid
        return {
            "id": report_id,
            "report_type": report_type,
            "period": period,
            "generated_at": now,
            "content": content_dict,
        }
    finally:
        conn.close()


# ── dashboard helpers ──────────────────────────────────────────────────────────
def _dashboard_data() -> dict:
    conn = get_db()
    now = time.time()
    try:
        # latest KPI per metric
        rows = conn.execute(
            "SELECT metric_name, metric_value, snapshot_at "
            "FROM kpi_snapshots "
            "ORDER BY snapshot_at DESC"
        ).fetchall()
        seen = {}
        for r in rows:
            if r["metric_name"] not in seen:
                seen[r["metric_name"]] = {
                    "metric_name": r["metric_name"],
                    "metric_value": r["metric_value"],
                    "snapshot_at": r["snapshot_at"],
                }

        # revenue_7d
        cutoff_7d = now - 7 * 86400
        cutoff_14d = now - 14 * 86400
        r7 = conn.execute(
            "SELECT COALESCE(SUM(metric_value),0) FROM kpi_snapshots "
            "WHERE metric_name='daily_revenue' AND snapshot_at >= ?",
            (cutoff_7d,)
        ).fetchone()
        revenue_7d = _safe_float(r7[0] if r7 else 0)

        # prior 7d
        rp = conn.execute(
            "SELECT COALESCE(SUM(metric_value),0) FROM kpi_snapshots "
            "WHERE metric_name='daily_revenue' AND snapshot_at >= ? AND snapshot_at < ?",
            (cutoff_14d, cutoff_7d)
        ).fetchone()
        revenue_prior_7d = _safe_float(rp[0] if rp else 0)
        if revenue_prior_7d > 0:
            revenue_growth = round((revenue_7d - revenue_prior_7d) / revenue_prior_7d * 100, 2)
        else:
            revenue_growth = 0.0

        # churn_rate: subscriptions cancelled this month / total active
        churn_rate = 0.0
        if _table_exists(conn, "subscriptions"):
            import calendar as _cal
            t = time.gmtime(now)
            month_start = _cal.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0))
            cancelled = conn.execute(
                "SELECT COUNT(*) FROM subscriptions WHERE status='cancelled' AND updated_at >= ?",
                (month_start,)
            ).fetchone()
            active = conn.execute(
                "SELECT COUNT(*) FROM subscriptions WHERE status='active'"
            ).fetchone()
            n_cancelled = _safe_float(cancelled[0] if cancelled else 0)
            n_active = _safe_float(active[0] if active else 0)
            total = n_cancelled + n_active
            churn_rate = round(n_cancelled / total * 100, 2) if total > 0 else 0.0

        # avg_order_value
        avg_order_value = 0.0
        if _table_exists(conn, "orders"):
            aov = conn.execute(
                "SELECT AVG(amount) FROM orders WHERE status='completed'"
            ).fetchone()
            avg_order_value = round(_safe_float(aov[0] if aov else 0) / 100.0, 2)

        # active alerts
        alert_rows = conn.execute(
            "SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY triggered_at DESC LIMIT 20"
        ).fetchall()
        active_alerts = _rows_to_list(alert_rows)

        # recent anomalies (alerts with message starting "Anomaly")
        anomaly_rows = conn.execute(
            "SELECT * FROM alerts WHERE message LIKE 'Anomaly%' "
            "ORDER BY triggered_at DESC LIMIT 10"
        ).fetchall()
        recent_anomalies = _rows_to_list(anomaly_rows)

        # top metrics by latest value desc
        top_metrics = sorted(seen.values(), key=lambda x: x["metric_value"], reverse=True)[:10]

        return {
            "current_kpis": list(seen.values()),
            "revenue_7d": round(revenue_7d, 2),
            "revenue_growth_pct": revenue_growth,
            "churn_rate_pct": churn_rate,
            "avg_order_value": avg_order_value,
            "active_alerts": active_alerts,
            "recent_anomalies": recent_anomalies,
            "top_metrics": top_metrics,
            "generated_at": now,
        }
    finally:
        conn.close()


# ── request handler ────────────────────────────────────────────────────────────
def _parse_path(path: str):
    """Split path into parts and query string dict."""
    if "?" in path:
        base, qs = path.split("?", 1)
    else:
        base, qs = path, ""
    parts = [p for p in base.strip("/").split("/") if p]
    params = {}
    if qs:
        for pair in qs.split("&"):
            if "=" in pair:
                pk, pv = pair.split("=", 1)
                params[pk] = pv
    return parts, params


class AnalyticsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _send(self, code: int, body: dict):
        data = json.dumps(body, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _ok(self, body: dict):
        self._send(200, body)

    def _created(self, body: dict):
        self._send(201, body)

    def _bad(self, msg: str):
        self._send(400, {"error": msg})

    def _not_found(self, msg: str = "not found"):
        self._send(404, {"error": msg})

    def _forbidden(self):
        self._send(403, {"error": "forbidden"})

    def _server_error(self, msg: str):
        self._send(500, {"error": msg})

    def _require_admin(self) -> bool:
        auth = self.headers.get("X-Admin-Secret", "") or self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            auth = auth[7:]
        if not ADMIN_SECRET or auth != ADMIN_SECRET:
            self._forbidden()
            return False
        return True

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    # ── GET ────────────────────────────────────────────────────────────────────
    def do_GET(self):
        parts, params = _parse_path(self.path)

        # /health
        if parts == ["health"] or parts == []:
            conn = get_db()
            snap_count = conn.execute("SELECT COUNT(*) FROM kpi_snapshots").fetchone()[0]
            alert_count = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE resolved_at IS NULL"
            ).fetchone()[0]
            conn.close()
            self._ok({
                "status": "ok",
                "service": "fm_analytics_hub",
                "port": PORT,
                "uptime_seconds": round(time.time() - START_TIME, 1),
                "snapshot_count": snap_count,
                "active_alert_count": alert_count,
            })
            return

        # /dashboard
        if parts == ["dashboard"]:
            try:
                self._ok(_dashboard_data())
            except Exception as exc:
                self._server_error(str(exc))
            return

        # /kpis or /kpis/{metric_name}
        if parts and parts[0] == "kpis":
            if len(parts) == 1:
                # latest value per metric
                conn = get_db()
                try:
                    rows = conn.execute(
                        "SELECT metric_name, metric_value, dimension, dimension_value, snapshot_at "
                        "FROM kpi_snapshots ORDER BY snapshot_at DESC"
                    ).fetchall()
                    seen = {}
                    for r in rows:
                        if r["metric_name"] not in seen:
                            seen[r["metric_name"]] = _row_to_dict(r)
                    self._ok({"kpis": list(seen.values())})
                finally:
                    conn.close()
                return

            metric_name = parts[1]
            period_filter = params.get("period")
            limit = int(params.get("limit", "100"))
            conn = get_db()
            try:
                if period_filter:
                    rows = conn.execute(
                        "SELECT * FROM kpi_snapshots WHERE metric_name=? AND period=? "
                        "ORDER BY snapshot_at DESC LIMIT ?",
                        (metric_name, period_filter, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM kpi_snapshots WHERE metric_name=? "
                        "ORDER BY snapshot_at DESC LIMIT ?",
                        (metric_name, limit)
                    ).fetchall()
                self._ok({"metric_name": metric_name, "series": _rows_to_list(rows)})
            finally:
                conn.close()
            return

        # /reports or /reports/{id}
        if parts and parts[0] == "reports":
            if len(parts) == 1:
                conn = get_db()
                try:
                    rows = conn.execute(
                        "SELECT id, report_type, period, generated_at FROM reports "
                        "ORDER BY generated_at DESC LIMIT 50"
                    ).fetchall()
                    self._ok({"reports": _rows_to_list(rows)})
                finally:
                    conn.close()
                return

            if len(parts) == 2 and parts[1] != "generate":
                try:
                    report_id = int(parts[1])
                except ValueError:
                    self._bad("invalid report id")
                    return
                conn = get_db()
                try:
                    row = conn.execute(
                        "SELECT * FROM reports WHERE id=?", (report_id,)
                    ).fetchone()
                    if not row:
                        self._not_found("report not found")
                        return
                    d = _row_to_dict(row)
                    try:
                        d["content"] = json.loads(d["content"])
                    except Exception:
                        pass
                    self._ok(d)
                finally:
                    conn.close()
                return

        # /funnels or /funnels/{funnel_name}
        if parts and parts[0] == "funnels":
            if len(parts) == 1:
                conn = get_db()
                try:
                    rows = conn.execute(
                        "SELECT * FROM funnels ORDER BY funnel_name, step_order"
                    ).fetchall()
                    self._ok({"funnels": _rows_to_list(rows)})
                finally:
                    conn.close()
                return

            funnel_name = parts[1]
            conn = get_db()
            try:
                rows = conn.execute(
                    "SELECT * FROM funnels WHERE funnel_name=? ORDER BY step_order",
                    (funnel_name,)
                ).fetchall()
                self._ok({"funnel_name": funnel_name, "steps": _rows_to_list(rows)})
            finally:
                conn.close()
            return

        # /cohorts
        if parts == ["cohorts"]:
            conn = get_db()
            try:
                rows = conn.execute(
                    "SELECT * FROM cohorts ORDER BY cohort_date, retention_day"
                ).fetchall()
                self._ok({"cohorts": _rows_to_list(rows)})
            finally:
                conn.close()
            return

        # /alerts
        if parts == ["alerts"]:
            conn = get_db()
            try:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY triggered_at DESC"
                ).fetchall()
                self._ok({"alerts": _rows_to_list(rows)})
            finally:
                conn.close()
            return

        self._not_found()

    # ── POST ───────────────────────────────────────────────────────────────────
    def do_POST(self):
        parts, _ = _parse_path(self.path)
        body = self._read_body()

        # /kpis — manual KPI record
        if parts == ["kpis"]:
            metric_name = body.get("metric_name", "").strip()
            if not metric_name:
                self._bad("metric_name required")
                return
            metric_value = _safe_float(body.get("metric_value", 0))
            dimension = body.get("dimension", "all")
            dimension_value = body.get("dimension_value", "all")
            period = body.get("period", "manual")
            now = time.time()
            conn = get_db()
            try:
                cursor = conn.execute(
                    "INSERT INTO kpi_snapshots "
                    "(period, metric_name, metric_value, dimension, dimension_value, snapshot_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (period, metric_name, metric_value, dimension, dimension_value, now)
                )
                conn.commit()
                snap_id = cursor.lastrowid
            finally:
                conn.close()
            _check_alert_threshold(metric_name, metric_value)
            _anomaly_check(metric_name, metric_value)
            self._created({
                "id": snap_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "snapshot_at": now,
            })
            return

        # /reports/generate
        if parts == ["reports", "generate"]:
            report_type = body.get("report_type", "daily")
            period = body.get("period", "")
            if not period:
                t = time.gmtime()
                period = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
            try:
                result = _generate_report_on_demand(report_type, period)
                self._created(result)
            except Exception as exc:
                self._server_error(str(exc))
            return

        # /funnels — record funnel step
        if parts == ["funnels"]:
            funnel_name = body.get("funnel_name", "").strip()
            step_name = body.get("step_name", "").strip()
            if not funnel_name or not step_name:
                self._bad("funnel_name and step_name required")
                return
            step_order = int(body.get("step_order", 0))
            count = int(body.get("count", 0))
            now = time.time()

            # compute conversion_rate vs previous step
            conn = get_db()
            try:
                prev = conn.execute(
                    "SELECT count FROM funnels WHERE funnel_name=? AND step_order=? "
                    "ORDER BY recorded_at DESC LIMIT 1",
                    (funnel_name, step_order - 1)
                ).fetchone()
                if prev and prev["count"] > 0:
                    conversion_rate = round(count / prev["count"] * 100, 2)
                else:
                    conversion_rate = 100.0 if step_order == 0 else 0.0

                cursor = conn.execute(
                    "INSERT INTO funnels "
                    "(funnel_name, step_name, step_order, count, conversion_rate, recorded_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (funnel_name, step_name, step_order, count, conversion_rate, now)
                )
                conn.commit()
                row_id = cursor.lastrowid
                self._created({
                    "id": row_id,
                    "funnel_name": funnel_name,
                    "step_name": step_name,
                    "step_order": step_order,
                    "count": count,
                    "conversion_rate": conversion_rate,
                    "recorded_at": now,
                })
            finally:
                conn.close()
            return

        # /cohorts — record cohort retention
        if parts == ["cohorts"]:
            cohort_date = body.get("cohort_date", "").strip()
            cohort_size = int(body.get("cohort_size", 0))
            retention_day = int(body.get("retention_day", 0))
            retained = int(body.get("retained", 0))
            if not cohort_date:
                self._bad("cohort_date required")
                return
            retention_rate = round(retained / cohort_size * 100, 2) if cohort_size > 0 else 0.0
            conn = get_db()
            try:
                cursor = conn.execute(
                    "INSERT INTO cohorts "
                    "(cohort_date, cohort_size, retention_day, retained, retention_rate) "
                    "VALUES (?,?,?,?,?)",
                    (cohort_date, cohort_size, retention_day, retained, retention_rate)
                )
                conn.commit()
                row_id = cursor.lastrowid
                self._created({
                    "id": row_id,
                    "cohort_date": cohort_date,
                    "cohort_size": cohort_size,
                    "retention_day": retention_day,
                    "retained": retained,
                    "retention_rate": retention_rate,
                })
            finally:
                conn.close()
            return

        # /alerts — create alert threshold
        if parts == ["alerts"]:
            metric_name = body.get("metric_name", "").strip()
            if not metric_name:
                self._bad("metric_name required")
                return
            threshold = _safe_float(body.get("threshold", 0))
            operator = body.get("operator", "gt").strip()
            if operator not in ("gt", "lt", "gte", "lte"):
                self._bad("operator must be gt, lt, gte, or lte")
                return
            message = body.get("message", f"Alert: {metric_name} {operator} {threshold}")
            now = time.time()
            conn = get_db()
            try:
                cursor = conn.execute(
                    "INSERT INTO alerts "
                    "(metric_name, threshold, operator, current_value, triggered_at, resolved_at, message) "
                    "VALUES (?,?,?,NULL,NULL,NULL,?)",
                    (metric_name, threshold, operator, message)
                )
                conn.commit()
                row_id = cursor.lastrowid
                self._created({
                    "id": row_id,
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "operator": operator,
                    "message": message,
                })
            finally:
                conn.close()
            return

        self._not_found()

    # ── PUT ────────────────────────────────────────────────────────────────────
    def do_PUT(self):
        parts, _ = _parse_path(self.path)

        # /alerts/{id}/resolve
        if len(parts) == 3 and parts[0] == "alerts" and parts[2] == "resolve":
            try:
                alert_id = int(parts[1])
            except ValueError:
                self._bad("invalid alert id")
                return
            conn = get_db()
            try:
                row = conn.execute("SELECT id FROM alerts WHERE id=?", (alert_id,)).fetchone()
                if not row:
                    self._not_found("alert not found")
                    return
                conn.execute(
                    "UPDATE alerts SET resolved_at=? WHERE id=?",
                    (time.time(), alert_id)
                )
                conn.commit()
                self._ok({"id": alert_id, "resolved": True, "resolved_at": time.time()})
            finally:
                conn.close()
            return

        self._not_found()

    # ── DELETE ─────────────────────────────────────────────────────────────────
    def do_DELETE(self):
        parts, _ = _parse_path(self.path)

        # /alerts/{id} — admin gated
        if len(parts) == 2 and parts[0] == "alerts":
            if not self._require_admin():
                return
            try:
                alert_id = int(parts[1])
            except ValueError:
                self._bad("invalid alert id")
                return
            conn = get_db()
            try:
                row = conn.execute("SELECT id FROM alerts WHERE id=?", (alert_id,)).fetchone()
                if not row:
                    self._not_found("alert not found")
                    return
                conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
                conn.commit()
                self._ok({"id": alert_id, "deleted": True})
            finally:
                conn.close()
            return

        self._not_found()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    init_db()

    # start background KPI thread
    t = threading.Thread(target=_kpi_loop, daemon=True, name="kpi-loop")
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), AnalyticsHandler)
    print(f"[analytics_hub] listening on port {PORT} | db={DB}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[analytics_hub] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
