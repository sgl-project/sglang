#!/usr/bin/env python3
"""
fm_revenue_forecast.py — Revenue Forecasting + Financial Dashboard Agent (Port 7850)
Provides forecasting models (SMA, Linear Regression, Exponential Smoothing),
financial dashboards, goal tracking, and multi-source revenue aggregation.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import math
import time
import sqlite3
import threading
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── vault ──────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("REVENUE_FORECAST_PORT", "7850"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB           = ROOT / "database" / "sovereign.db"

ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL = 300  # seconds between background syncs


# ── database helpers ───────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _safe_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    """Execute a query, returning rows or [] on OperationalError."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []


def _init_db() -> None:
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS revenue_entries (
                id          INTEGER PRIMARY KEY,
                source      TEXT,
                category    TEXT,
                amount_usd  REAL,
                currency    TEXT DEFAULT 'USD',
                description TEXT,
                date        TEXT,
                recorded_at REAL
            );
            CREATE TABLE IF NOT EXISTS revenue_goals (
                id         INTEGER PRIMARY KEY,
                period     TEXT UNIQUE,
                target_usd REAL,
                actual_usd REAL DEFAULT 0,
                status     TEXT DEFAULT 'active',
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS forecast_models (
                id           INTEGER PRIMARY KEY,
                name         TEXT UNIQUE,
                model_type   TEXT,
                parameters   TEXT,
                mape         REAL,
                last_trained REAL,
                created_at   REAL
            );
            CREATE INDEX IF NOT EXISTS idx_revenue_date   ON revenue_entries(date);
            CREATE INDEX IF NOT EXISTS idx_revenue_source ON revenue_entries(source);
        """)
        conn.commit()
    finally:
        conn.close()


def _seed_goals() -> None:
    now = time.time()
    today = datetime.utcnow()
    cur_month  = today.strftime("%Y-%m")
    next_month = (today.replace(day=28) + timedelta(days=4)).strftime("%Y-%m")
    q3_period  = f"{today.year}-Q3"

    seeds = [
        (cur_month,  2000.0),
        (next_month, 5000.0),
        (q3_period, 15000.0),
    ]
    conn = _get_conn()
    try:
        for period, target in seeds:
            conn.execute(
                "INSERT OR IGNORE INTO revenue_goals (period, target_usd, created_at) VALUES (?,?,?)",
                (period, target, now),
            )
        conn.commit()
    finally:
        conn.close()


# ── revenue sync helpers ───────────────────────────────────────────────────────

def _sync_sources(conn: sqlite3.Connection) -> dict:
    """Pull revenue from all known sovereign.db tables into revenue_entries."""
    counts = {}
    now = time.time()

    # Helper: generate a stable dedup key
    def _dedup_key(source: str, date: str, amount: float) -> str:
        raw = f"{source}:{date}:{amount:.6f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _insert(source: str, category: str, amount: float, description: str, date: str) -> bool:
        if amount <= 0:
            return False
        dedup = _dedup_key(source, date, amount)
        existing = _safe_query(conn, "SELECT id FROM revenue_entries WHERE description LIKE ?",
                               (f"%{dedup}%",))
        if existing:
            return False
        conn.execute(
            "INSERT INTO revenue_entries (source,category,amount_usd,currency,description,date,recorded_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (source, category, amount, "USD", f"synced:{dedup}", date, now),
        )
        return True

    # gumroad_sales → sum(price)
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, SUM(price) as total "
        "FROM gumroad_sales GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["total"]:
            if _insert("gumroad", "digital_product", float(r["total"]), "Gumroad sale", r["d"]):
                n += 1
    counts["gumroad"] = n

    # coinbase_orders → sum(total_value_usd) where side='BUY'
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, SUM(total_value_usd) as total "
        "FROM coinbase_orders WHERE side='BUY' GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["total"]:
            if _insert("coinbase", "trading", float(r["total"]), "Coinbase trading", r["d"]):
                n += 1
    counts["coinbase"] = n

    # aiaas_requests → sum(cost_usd)
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, SUM(cost_usd) as total "
        "FROM aiaas_requests GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["total"]:
            if _insert("aiaas", "api_service", float(r["total"]), "AIaaS markup", r["d"]):
                n += 1
    counts["aiaas"] = n

    # api_requests → count * $0.001
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, COUNT(*) as cnt "
        "FROM api_requests GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["cnt"]:
            amount = float(r["cnt"]) * 0.001
            if _insert("data_api", "api_service", amount, "Data API estimated", r["d"]):
                n += 1
    counts["data_api"] = n

    # nft_sales → sum(price_eth) * eth_price_usd
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, SUM(price_eth) as total_eth, "
        "AVG(eth_price_usd) as eth_px FROM nft_sales GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["total_eth"] and r["eth_px"]:
            amount = float(r["total_eth"]) * float(r["eth_px"])
            if _insert("nft", "nft_sale", amount, "NFT sale", r["d"]):
                n += 1
    counts["nft"] = n

    # license_events issued → join products for price
    rows = _safe_query(conn,
        "SELECT date(le.created_at,'unixepoch') as d, SUM(p.price) as total "
        "FROM license_events le JOIN products p ON le.product_id=p.id "
        "WHERE le.event_type='issued' GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["total"]:
            if _insert("licensing", "license_sale", float(r["total"]), "License issued", r["d"]):
                n += 1
    counts["licensing"] = n

    # bugcrowd_submissions resolved → sum(reward)
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, SUM(reward) as total "
        "FROM bugcrowd_submissions WHERE status='resolved' GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["total"]:
            if _insert("bugcrowd", "bounty", float(r["total"]), "Bug bounty reward", r["d"]):
                n += 1
    counts["bugcrowd"] = n

    # affiliate_clicks → $0.05 per click
    rows = _safe_query(conn,
        "SELECT date(created_at,'unixepoch') as d, COUNT(*) as cnt "
        "FROM affiliate_clicks GROUP BY d")
    n = 0
    for r in rows:
        if r["d"] and r["cnt"]:
            amount = float(r["cnt"]) * 0.05
            if _insert("affiliate", "affiliate", amount, "Affiliate click estimated", r["d"]):
                n += 1
    counts["affiliate"] = n

    conn.commit()
    return counts


# ── forecasting helpers ────────────────────────────────────────────────────────

def _get_daily_series(conn: sqlite3.Connection, days: int) -> list:
    """Return [(date_str, amount)] sorted by date for the last N days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = _safe_query(conn,
        "SELECT date, SUM(amount_usd) as total FROM revenue_entries "
        "WHERE date >= ? GROUP BY date ORDER BY date ASC",
        (cutoff,))
    return [(r["date"], float(r["total"])) for r in rows if r["total"] is not None]


def _sma_forecast(series: list, days_ahead: int, window: int = 7) -> list:
    """Simple Moving Average forecast."""
    if not series:
        return []
    values = [v for _, v in series]
    recent = values[-window:] if len(values) >= window else values
    avg = sum(recent) / len(recent) if recent else 0.0
    last_date = datetime.strptime(series[-1][0], "%Y-%m-%d")
    result = []
    for i in range(1, days_ahead + 1):
        d = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        result.append((d, avg))
    return result


def _linear_forecast(series: list, days_ahead: int) -> tuple:
    """Linear regression forecast. Returns (forecast_list, slope, intercept)."""
    if not series:
        return [], 0.0, 0.0
    x = list(range(len(series)))
    y = [v for _, v in series]
    n = len(x)
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_xx = sum(xi ** 2 for xi in x)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2 + 1e-10)
    b = (sum_y - a * sum_x) / n
    last_date = datetime.strptime(series[-1][0], "%Y-%m-%d")
    result = []
    for i in range(1, days_ahead + 1):
        xi = len(series) - 1 + i
        predicted = max(0.0, a * xi + b)
        d = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        result.append((d, predicted))
    return result, a, b


def _exp_smooth_forecast(series: list, days_ahead: int, alpha: float = 0.3) -> tuple:
    """Exponential Smoothing forecast. Returns (forecast_list, last_smoothed)."""
    if not series:
        return [], 0.0
    values = [v for _, v in series]
    s = values[0]
    for y in values[1:]:
        s = alpha * y + (1 - alpha) * s
    last_date = datetime.strptime(series[-1][0], "%Y-%m-%d")
    result = []
    current_s = s
    for i in range(1, days_ahead + 1):
        d = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        result.append((d, max(0.0, current_s)))
        # smoothed value stays constant unless we have new observations
    return result, s


def _confidence_interval(forecast: list, series: list) -> dict:
    """Estimate 80% CI based on historical volatility."""
    if not series:
        return {"lower": 0.0, "upper": 0.0}
    values = [v for _, v in series]
    if len(values) < 2:
        avg = values[0] if values else 0.0
        return {"lower": avg * 0.8, "upper": avg * 1.2}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    # 1.28 std = ~80% CI
    forecast_values = [v for _, v in forecast]
    avg_f = sum(forecast_values) / len(forecast_values) if forecast_values else 0.0
    return {
        "lower": round(max(0.0, avg_f - 1.28 * std), 2),
        "upper": round(avg_f + 1.28 * std, 2),
    }


def _trend_label(series: list) -> str:
    if len(series) < 2:
        return "flat"
    _, slope, _ = _linear_forecast(series, 1)
    if slope > 0.5:
        return "growing"
    if slope < -0.5:
        return "declining"
    return "flat"


# ── auth helper ────────────────────────────────────────────────────────────────

def _check_auth(handler) -> bool:
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True  # no secret configured → open
    return secret == ADMIN_SECRET


# ── background sync thread ─────────────────────────────────────────────────────

def _background_sync() -> None:
    while True:
        time.sleep(SYNC_INTERVAL)
        try:
            conn = _get_conn()
            try:
                _sync_sources(conn)
            finally:
                conn.close()
        except Exception:
            pass


# ── HTTP handler ───────────────────────────────────────────────────────────────

class RevenueHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            return {}

    def _parse_qs(self) -> dict:
        path = self.path
        if "?" not in path:
            return {}
        qs = path.split("?", 1)[1]
        params = {}
        for part in qs.split("&"):
            if "=" in part:
                k, _, v = part.partition("=")
                params[k] = v
        return params

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self._handle_health()
        elif path == "/revenue/daily":
            self._handle_revenue_daily()
        elif path == "/revenue/monthly":
            self._handle_revenue_monthly()
        elif path == "/revenue/by_source":
            self._handle_revenue_by_source()
        elif path == "/forecast":
            self._handle_forecast()
        elif path == "/forecast/all":
            self._handle_forecast_all()
        elif path == "/goals":
            self._handle_goals_get()
        elif path == "/dashboard":
            self._handle_dashboard()
        elif path == "/analytics":
            self._handle_analytics()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/revenue/record":
            self._handle_revenue_record()
        elif path == "/revenue/sync":
            self._handle_revenue_sync()
        elif path == "/goals/set":
            self._handle_goals_set()
        else:
            self._send_json({"error": "not found"}, 404)

    # ── GET /health ────────────────────────────────────────────────────────────

    def _handle_health(self):
        self._send_json({"status": "ok", "service": "fm-revenue-forecast", "port": PORT})

    # ── POST /revenue/record ───────────────────────────────────────────────────

    def _handle_revenue_record(self):
        if not _check_auth(self):
            self._send_json({"error": "unauthorized"}, 401)
            return
        body = self._read_body()
        source      = body.get("source", "manual")
        category    = body.get("category", "other")
        amount      = float(body.get("amount_usd", 0))
        description = body.get("description", "")
        date        = body.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
        currency    = body.get("currency", "USD")

        conn = _get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO revenue_entries (source,category,amount_usd,currency,description,date,recorded_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (source, category, amount, currency, description, date, time.time()),
            )
            conn.commit()
            self._send_json({"entry_id": cur.lastrowid})
        finally:
            conn.close()

    # ── POST /revenue/sync ─────────────────────────────────────────────────────

    def _handle_revenue_sync(self):
        conn = _get_conn()
        try:
            sources = _sync_sources(conn)
            total = sum(sources.values())
            self._send_json({"synced": total, "sources": sources})
        finally:
            conn.close()

    # ── GET /revenue/daily ─────────────────────────────────────────────────────

    def _handle_revenue_daily(self):
        params = self._parse_qs()
        days = int(params.get("days", 30))
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        conn = _get_conn()
        try:
            rows = _safe_query(conn,
                "SELECT date, source, SUM(amount_usd) as total "
                "FROM revenue_entries WHERE date >= ? GROUP BY date, source ORDER BY date ASC",
                (cutoff,))

            # aggregate by date
            by_date = {}
            for r in rows:
                d = r["date"]
                if d not in by_date:
                    by_date[d] = {"date": d, "total": 0.0, "by_source": {}}
                src = r["source"]
                amt = float(r["total"] or 0)
                by_date[d]["total"] = round(by_date[d]["total"] + amt, 4)
                by_date[d]["by_source"][src] = round(
                    by_date[d]["by_source"].get(src, 0.0) + amt, 4)

            result = sorted(by_date.values(), key=lambda x: x["date"])
            self._send_json(result)
        finally:
            conn.close()

    # ── GET /revenue/monthly ───────────────────────────────────────────────────

    def _handle_revenue_monthly(self):
        conn = _get_conn()
        try:
            rows = _safe_query(conn,
                "SELECT substr(date,1,7) as month, source, SUM(amount_usd) as total "
                "FROM revenue_entries GROUP BY month, source ORDER BY month ASC")

            by_month = {}
            for r in rows:
                m = r["month"]
                if m not in by_month:
                    by_month[m] = {"month": m, "total": 0.0, "sources": {}}
                src = r["source"]
                amt = float(r["total"] or 0)
                by_month[m]["total"] = round(by_month[m]["total"] + amt, 4)
                by_month[m]["sources"][src] = round(
                    by_month[m]["sources"].get(src, 0.0) + amt, 4)

            result = sorted(by_month.values(), key=lambda x: x["month"])
            self._send_json(result)
        finally:
            conn.close()

    # ── GET /revenue/by_source ─────────────────────────────────────────────────

    def _handle_revenue_by_source(self):
        conn = _get_conn()
        try:
            rows = _safe_query(conn,
                "SELECT source, SUM(amount_usd) as total FROM revenue_entries GROUP BY source")
            result = {}
            grand_total = 0.0
            for r in rows:
                amt = round(float(r["total"] or 0), 4)
                result[r["source"]] = amt
                grand_total += amt
            result["total"] = round(grand_total, 4)
            self._send_json(result)
        finally:
            conn.close()

    # ── GET /forecast ──────────────────────────────────────────────────────────

    def _handle_forecast(self):
        params      = self._parse_qs()
        model_name  = params.get("model", "linear")
        days_ahead  = int(params.get("days_ahead", 30))
        history_days = int(params.get("history_days", 90))

        conn = _get_conn()
        try:
            series = _get_daily_series(conn, history_days)

            if model_name == "sma":
                forecast = _sma_forecast(series, days_ahead)
                extra = {}
            elif model_name == "exp_smooth":
                forecast, last_val = _exp_smooth_forecast(series, days_ahead)
                extra = {"last_smoothed": round(last_val, 4)}
            else:
                model_name = "linear"
                forecast, slope, intercept = _linear_forecast(series, days_ahead)
                extra = {"slope": round(slope, 6), "intercept": round(intercept, 4)}

            ci = _confidence_interval(forecast, series)
            trend = _trend_label(series)
            forecast_list = [{"date": d, "predicted": round(v, 4)} for d, v in forecast]

            result = {
                "model": model_name,
                "forecast": forecast_list,
                "confidence_interval": ci,
                "trend": trend,
            }
            result.update(extra)
            self._send_json(result)
        finally:
            conn.close()

    # ── GET /forecast/all ──────────────────────────────────────────────────────

    def _handle_forecast_all(self):
        conn = _get_conn()
        try:
            series = _get_daily_series(conn, 90)
            days_ahead = 30

            sma_fc = _sma_forecast(series, days_ahead)
            lin_fc, slope, _ = _linear_forecast(series, days_ahead)
            exp_fc, last_val = _exp_smooth_forecast(series, days_ahead)

            def _sum_fc(fc):
                return round(sum(v for _, v in fc), 4)

            def _avg_fc(fc):
                return round(sum(v for _, v in fc) / len(fc), 4) if fc else 0.0

            sma_total = _sum_fc(sma_fc)
            lin_total = _sum_fc(lin_fc)
            exp_total = _sum_fc(exp_fc)

            consensus = round((sma_total + lin_total + exp_total) / 3, 4)

            self._send_json({
                "sma": {
                    "30d_total": sma_total,
                    "daily_avg": _avg_fc(sma_fc),
                },
                "linear": {
                    "30d_total": lin_total,
                    "slope": round(slope, 6),
                },
                "exp_smooth": {
                    "30d_total": exp_total,
                    "last_value": round(last_val, 4),
                },
                "consensus_30d": consensus,
            })
        finally:
            conn.close()

    # ── POST /goals/set ────────────────────────────────────────────────────────

    def _handle_goals_set(self):
        if not _check_auth(self):
            self._send_json({"error": "unauthorized"}, 401)
            return
        body   = self._read_body()
        period = body.get("period", "")
        target = float(body.get("target_usd", 0))
        if not period:
            self._send_json({"error": "period required"}, 400)
            return

        conn = _get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO revenue_goals (period, target_usd, created_at) VALUES (?,?,?) "
                "ON CONFLICT(period) DO UPDATE SET target_usd=excluded.target_usd",
                (period, target, time.time()),
            )
            conn.commit()
            row = _safe_query(conn, "SELECT id FROM revenue_goals WHERE period=?", (period,))
            goal_id = row[0]["id"] if row else cur.lastrowid
            self._send_json({"goal_id": goal_id})
        finally:
            conn.close()

    # ── GET /goals ─────────────────────────────────────────────────────────────

    def _handle_goals_get(self):
        conn = _get_conn()
        try:
            goals = _safe_query(conn,
                "SELECT id, period, target_usd, actual_usd, status FROM revenue_goals "
                "WHERE status='active' ORDER BY period")

            result = []
            today = datetime.utcnow()

            for g in goals:
                period  = g["period"]
                target  = float(g["target_usd"] or 0)
                actual  = float(g["actual_usd"] or 0)

                # recompute actual from revenue_entries
                if "-Q" in period:
                    year, q = period.split("-Q")
                    q = int(q)
                    start_month = (q - 1) * 3 + 1
                    end_month   = start_month + 2
                    start = f"{year}-{start_month:02d}-01"
                    end   = f"{year}-{end_month:02d}-31"
                    rows = _safe_query(conn,
                        "SELECT SUM(amount_usd) as s FROM revenue_entries WHERE date>=? AND date<=?",
                        (start, end))
                else:
                    rows = _safe_query(conn,
                        "SELECT SUM(amount_usd) as s FROM revenue_entries WHERE substr(date,1,7)=?",
                        (period,))
                actual = float(rows[0]["s"] or 0) if rows else 0.0

                pct = round((actual / target * 100), 2) if target > 0 else 0.0

                # days remaining
                try:
                    if "-Q" in period:
                        year, q = period.split("-Q")
                        q = int(q)
                        end_month = q * 3
                        import calendar
                        last_day  = calendar.monthrange(int(year), end_month)[1]
                        period_end = datetime(int(year), end_month, last_day)
                    else:
                        year, month = period.split("-")
                        import calendar
                        last_day = calendar.monthrange(int(year), int(month))[1]
                        period_end = datetime(int(year), int(month), last_day)
                    days_remaining = max(0, (period_end - today).days)
                except Exception:
                    days_remaining = 0

                # update actual_usd in db
                conn.execute("UPDATE revenue_goals SET actual_usd=? WHERE period=?", (actual, period))

                needed_per_day = 0.0
                if days_remaining > 0 and target > actual:
                    needed_per_day = (target - actual) / days_remaining

                on_track = False
                if days_remaining > 0:
                    series = _get_daily_series(conn, 14)
                    if series:
                        daily_avg = sum(v for _, v in series) / len(series)
                        on_track  = daily_avg >= needed_per_day
                    else:
                        on_track = False
                elif actual >= target:
                    on_track = True

                result.append({
                    "id": g["id"],
                    "period": period,
                    "target": target,
                    "actual": round(actual, 4),
                    "pct_achieved": pct,
                    "days_remaining": days_remaining,
                    "on_track": on_track,
                })

            conn.commit()
            self._send_json(result)
        finally:
            conn.close()

    # ── GET /dashboard ─────────────────────────────────────────────────────────

    def _handle_dashboard(self):
        conn = _get_conn()
        try:
            now   = datetime.utcnow()
            today = now.strftime("%Y-%m-%d")
            this_month = now.strftime("%Y-%m")

            week_start      = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
            last_week_start = (now - timedelta(days=now.weekday() + 7)).strftime("%Y-%m-%d")
            last_week_end   = (now - timedelta(days=now.weekday() + 1)).strftime("%Y-%m-%d")

            def _sum(sql, params=()):
                rows = _safe_query(conn, sql, params)
                return float(rows[0][0] or 0) if rows else 0.0

            today_rev  = _sum("SELECT SUM(amount_usd) FROM revenue_entries WHERE date=?", (today,))
            today_txns = len(_safe_query(conn, "SELECT id FROM revenue_entries WHERE date=?", (today,)))

            week_rev      = _sum("SELECT SUM(amount_usd) FROM revenue_entries WHERE date>=?", (week_start,))
            last_week_rev = _sum("SELECT SUM(amount_usd) FROM revenue_entries WHERE date>=? AND date<=?",
                                 (last_week_start, last_week_end))
            vs_last_week = 0.0
            if last_week_rev > 0:
                vs_last_week = round((week_rev - last_week_rev) / last_week_rev * 100, 2)

            month_rev = _sum(
                "SELECT SUM(amount_usd) FROM revenue_entries WHERE substr(date,1,7)=?", (this_month,))

            # current month target
            target_rows = _safe_query(conn,
                "SELECT target_usd FROM revenue_goals WHERE period=? AND status='active'",
                (this_month,))
            month_target = float(target_rows[0]["target_usd"]) if target_rows else 0.0
            pct_of_target = round(month_rev / month_target * 100, 2) if month_target > 0 else 0.0

            # forecast 30d
            series    = _get_daily_series(conn, 90)
            lin_fc, _, _ = _linear_forecast(series, 30)
            forecast_30d = round(sum(v for _, v in lin_fc), 4)

            # top source
            src_rows = _safe_query(conn,
                "SELECT source, SUM(amount_usd) as total FROM revenue_entries GROUP BY source ORDER BY total DESC LIMIT 1")
            top_source = src_rows[0]["source"] if src_rows else "none"

            # active goals count
            active_goals = len(_safe_query(conn,
                "SELECT id FROM revenue_goals WHERE status='active'"))

            # MRR/ARR estimate (last 30 days avg * 30 / 12)
            series_30 = _get_daily_series(conn, 30)
            monthly_run_rate = sum(v for _, v in series_30)
            mrr_estimate = round(monthly_run_rate, 4)
            arr_estimate = round(mrr_estimate * 12, 4)

            self._send_json({
                "today": {
                    "revenue": round(today_rev, 4),
                    "transactions": today_txns,
                },
                "this_week": {
                    "revenue": round(week_rev, 4),
                    "vs_last_week_pct": vs_last_week,
                },
                "this_month": {
                    "revenue": round(month_rev, 4),
                    "target": month_target,
                    "pct_of_target": pct_of_target,
                },
                "forecast_30d": forecast_30d,
                "top_source": top_source,
                "active_goals": active_goals,
                "mrr_estimate": mrr_estimate,
                "arr_estimate": arr_estimate,
            })
        finally:
            conn.close()

    # ── GET /analytics ─────────────────────────────────────────────────────────

    def _handle_analytics(self):
        conn = _get_conn()
        try:
            rows = _safe_query(conn,
                "SELECT date, SUM(amount_usd) as total FROM revenue_entries GROUP BY date ORDER BY date")
            if not rows:
                self._send_json({
                    "cagr_estimate": 0.0,
                    "avg_daily_revenue": 0.0,
                    "best_day": None,
                    "worst_day": None,
                    "revenue_volatility": 0.0,
                    "total_days": 0,
                    "total_revenue": 0.0,
                })
                return

            data = [(r["date"], float(r["total"] or 0)) for r in rows]
            values = [v for _, v in data]
            total_rev = sum(values)
            avg_daily = total_rev / len(values) if values else 0.0

            best_day_idx  = values.index(max(values))
            worst_day_idx = values.index(min(values))
            best_day  = {"date": data[best_day_idx][0],  "revenue": round(data[best_day_idx][1], 4)}
            worst_day = {"date": data[worst_day_idx][0], "revenue": round(data[worst_day_idx][1], 4)}

            # volatility (population std dev)
            mean = avg_daily
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance)

            # CAGR estimate: if data spans at least 365 days
            cagr = 0.0
            if len(data) >= 2:
                first_date = datetime.strptime(data[0][0], "%Y-%m-%d")
                last_date  = datetime.strptime(data[-1][0], "%Y-%m-%d")
                years = (last_date - first_date).days / 365.0
                if years > 0:
                    # Use first vs last 30-day window avg as growth proxy
                    first_30 = sum(values[:30]) / min(30, len(values))
                    last_30  = sum(values[-30:]) / min(30, len(values))
                    if first_30 > 0:
                        try:
                            cagr = round(((last_30 / first_30) ** (1 / max(years, 0.08)) - 1) * 100, 2)
                        except (ValueError, ZeroDivisionError):
                            cagr = 0.0

            self._send_json({
                "cagr_estimate": cagr,
                "avg_daily_revenue": round(avg_daily, 4),
                "best_day": best_day,
                "worst_day": worst_day,
                "revenue_volatility": round(std_dev, 4),
                "total_days": len(data),
                "total_revenue": round(total_rev, 4),
            })
        finally:
            conn.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    _init_db()
    _seed_goals()

    sync_thread = threading.Thread(target=_background_sync, daemon=True)
    sync_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), RevenueHandler)
    print(f"[fm-revenue-forecast] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[fm-revenue-forecast] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
