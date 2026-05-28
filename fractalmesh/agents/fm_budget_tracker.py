#!/usr/bin/env python3
"""
fm_budget_tracker.py — Budget & Financial Tracker (Port 7898)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT         = int(os.getenv("BUDGET_TRACKER_PORT", "7898"))
SG_KEY       = os.getenv("SENDGRID_API_KEY", "")
SG_FROM      = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.ai")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ALERT_EMAIL  = os.getenv("ALERT_EMAIL", SG_FROM)

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS budgets (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            budget_id    TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            category     TEXT NOT NULL,
            period       TEXT NOT NULL DEFAULT 'monthly',
            amount       REAL NOT NULL,
            spent        REAL NOT NULL DEFAULT 0,
            currency     TEXT NOT NULL DEFAULT 'AUD',
            alert_at_pct REAL NOT NULL DEFAULT 0.8,
            alerted      INTEGER NOT NULL DEFAULT 0,
            start_date   REAL,
            end_date     REAL,
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id       TEXT UNIQUE NOT NULL,
            budget_id    TEXT,
            category     TEXT NOT NULL,
            description  TEXT NOT NULL,
            amount       REAL NOT NULL,
            type         TEXT NOT NULL DEFAULT 'expense',
            currency     TEXT NOT NULL DEFAULT 'AUD',
            date         REAL NOT NULL,
            reference    TEXT,
            tags         TEXT DEFAULT '[]',
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS accounts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id   TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            type         TEXT NOT NULL DEFAULT 'checking',
            balance      REAL NOT NULL DEFAULT 0,
            currency     TEXT NOT NULL DEFAULT 'AUD',
            institution  TEXT,
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recurring (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            rec_id       TEXT UNIQUE NOT NULL,
            description  TEXT NOT NULL,
            amount       REAL NOT NULL,
            type         TEXT NOT NULL DEFAULT 'expense',
            category     TEXT NOT NULL,
            frequency    TEXT NOT NULL DEFAULT 'monthly',
            next_date    REAL NOT NULL,
            budget_id    TEXT,
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions(date);
        CREATE INDEX IF NOT EXISTS idx_txn_cat      ON transactions(category);
        CREATE INDEX IF NOT EXISTS idx_txn_budget   ON transactions(budget_id);
    """)
    con.commit()
    _seed_categories(con)
    con.close()

def _seed_categories(con):
    # No seeding needed — categories are free-form text
    pass

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

def _send_alert(subject, body):
    if not SG_KEY or not ALERT_EMAIL:
        return
    payload = json.dumps({
        "personalizations": [{"to": [{"email": ALERT_EMAIL}]}],
        "from": {"email": SG_FROM},
        "subject": subject,
        "content": [{"type": "text/html", "value": body}],
    }).encode()
    req = urllib.request.Request("https://api.sendgrid.com/v3/mail/send", data=payload)
    req.add_header("Authorization", f"Bearer {SG_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

def _check_budget_alerts(con):
    rows = con.execute(
        "SELECT * FROM budgets WHERE active=1 AND alerted=0 AND amount > 0"
    ).fetchall()
    for b in rows:
        pct = b["spent"] / b["amount"]
        if pct >= b["alert_at_pct"]:
            con.execute("UPDATE budgets SET alerted=1, updated_at=? WHERE budget_id=?",
                        (time.time(), b["budget_id"]))
            threading.Thread(target=_send_alert, args=(
                f"Budget Alert: {b['name']} at {int(pct*100)}%",
                f"<p>Budget <strong>{b['name']}</strong> has reached {int(pct*100)}% "
                f"(${b['spent']:.2f} of ${b['amount']:.2f} {b['currency']})</p>"
            ), daemon=True).start()

def _budget_daemon():
    while True:
        time.sleep(3600)
        try:
            con = _db()
            _check_budget_alerts(con)
            # process recurring transactions
            now = time.time()
            recs = con.execute(
                "SELECT * FROM recurring WHERE active=1 AND next_date <= ?", (now,)
            ).fetchall()
            for r in recs:
                tid = "txn_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO transactions(txn_id,budget_id,category,description,amount,type,date,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (tid, r["budget_id"], r["category"], r["description"], r["amount"], r["type"], now, now)
                )
                if r["budget_id"]:
                    con.execute(
                        "UPDATE budgets SET spent=spent+?, updated_at=? WHERE budget_id=? AND type='expense'",
                        (r["amount"] if r["type"] == "expense" else 0, now, r["budget_id"])
                    )
                # advance next_date
                freq_secs = {"daily": 86400, "weekly": 604800, "monthly": 2592000, "yearly": 31536000}
                next_d = r["next_date"] + freq_secs.get(r["frequency"], 2592000)
                con.execute("UPDATE recurring SET next_date=? WHERE rec_id=?", (next_d, r["rec_id"]))
            con.commit()
            con.close()
        except Exception:
            pass

threading.Thread(target=_budget_daemon, daemon=True).start()

class BudgetHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path.strip("/").split("/")
        qs = parse_qs(parsed.query)
        try:
            code, body = self._get(p, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p == ["health"]:
                return _j({"status": "ok", "port": PORT, "agent": "fm_budget_tracker"})

            if p == ["budgets"]:
                rows = con.execute(
                    "SELECT *, ROUND(spent/amount*100,1) as pct_used FROM budgets WHERE active=1 ORDER BY name"
                ).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "budgets":
                row = con.execute("SELECT * FROM budgets WHERE budget_id=?", (p[1],)).fetchone()
                if not row:
                    return _err("Budget not found", 404)
                b = dict(row)
                txns = con.execute(
                    "SELECT * FROM transactions WHERE budget_id=? ORDER BY date DESC LIMIT 20", (p[1],)
                ).fetchall()
                b["recent_transactions"] = [dict(t) for t in txns]
                return _j(b)

            if p == ["transactions"]:
                cat = qs.get("category", [None])[0]
                typ = qs.get("type", [None])[0]
                limit = int(qs.get("limit", ["100"])[0])
                q = "SELECT * FROM transactions WHERE 1=1"
                vals = []
                if cat:
                    q += " AND category=?"; vals.append(cat)
                if typ:
                    q += " AND type=?"; vals.append(typ)
                q += " ORDER BY date DESC LIMIT ?"
                vals.append(limit)
                rows = con.execute(q, vals).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["accounts"]:
                rows = con.execute("SELECT * FROM accounts WHERE active=1 ORDER BY name").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["recurring"]:
                rows = con.execute("SELECT * FROM recurring WHERE active=1 ORDER BY next_date").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["summary"]:
                now = time.time()
                month_start = now - 30 * 86400
                income = con.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM transactions WHERE type='income' AND date >= ?",
                    (month_start,)
                ).fetchone()[0]
                expense = con.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM transactions WHERE type='expense' AND date >= ?",
                    (month_start,)
                ).fetchone()[0]
                by_cat = con.execute(
                    "SELECT category, SUM(amount) as total FROM transactions "
                    "WHERE type='expense' AND date >= ? GROUP BY category ORDER BY total DESC",
                    (month_start,)
                ).fetchall()
                total_balance = con.execute(
                    "SELECT COALESCE(SUM(balance),0) FROM accounts WHERE active=1"
                ).fetchone()[0]
                over_budget = con.execute(
                    "SELECT COUNT(*) FROM budgets WHERE active=1 AND spent > amount"
                ).fetchone()[0]
                return _j({
                    "income_30d": round(income, 2),
                    "expense_30d": round(expense, 2),
                    "net_30d": round(income - expense, 2),
                    "total_account_balance": round(total_balance, 2),
                    "over_budget_count": over_budget,
                    "top_expense_categories": [{"category": r["category"], "total": round(r["total"],2)} for r in by_cat[:5]],
                })

            return _err("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["budgets"]:
                bid = "bud_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO budgets(budget_id,name,category,period,amount,currency,alert_at_pct,start_date,end_date,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (bid, data.get("name",""), data.get("category",""), data.get("period","monthly"),
                     data.get("amount", 0), data.get("currency","AUD"),
                     data.get("alert_at_pct", 0.8), data.get("start_date"), data.get("end_date"), now, now)
                )
                con.commit()
                return _j({"budget_id": bid}, 201)

            if p == ["transactions"]:
                tid = "txn_" + secrets.token_hex(8)
                amount = data.get("amount", 0)
                txn_type = data.get("type", "expense")
                bid = data.get("budget_id")
                con.execute(
                    "INSERT INTO transactions(txn_id,budget_id,category,description,amount,type,currency,date,reference,tags,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, bid, data.get("category",""), data.get("description",""),
                     amount, txn_type, data.get("currency","AUD"),
                     data.get("date", now), data.get("reference"),
                     json.dumps(data.get("tags",[])), now)
                )
                if bid and txn_type == "expense":
                    con.execute(
                        "UPDATE budgets SET spent=spent+?, updated_at=? WHERE budget_id=?",
                        (amount, now, bid)
                    )
                _check_budget_alerts(con)
                con.commit()
                return _j({"txn_id": tid}, 201)

            if p == ["accounts"]:
                aid = "acc_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO accounts(account_id,name,type,balance,currency,institution,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (aid, data.get("name",""), data.get("type","checking"),
                     data.get("balance",0), data.get("currency","AUD"),
                     data.get("institution"), now, now)
                )
                con.commit()
                return _j({"account_id": aid}, 201)

            if len(p) == 3 and p[0] == "accounts" and p[2] == "adjust":
                con.execute(
                    "UPDATE accounts SET balance=?, updated_at=? WHERE account_id=?",
                    (data.get("balance", 0), now, p[1])
                )
                con.commit()
                return _j({"account_id": p[1], "balance": data.get("balance", 0)})

            if p == ["recurring"]:
                rid = "rec_" + secrets.token_hex(8)
                freq_secs = {"daily": 86400, "weekly": 604800, "monthly": 2592000, "yearly": 31536000}
                freq = data.get("frequency", "monthly")
                next_d = now + freq_secs.get(freq, 2592000)
                con.execute(
                    "INSERT INTO recurring(rec_id,description,amount,type,category,frequency,next_date,budget_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, data.get("description",""), data.get("amount",0),
                     data.get("type","expense"), data.get("category",""),
                     freq, next_d, data.get("budget_id"), now)
                )
                con.commit()
                return _j({"rec_id": rid, "next_date": next_d}, 201)

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), BudgetHandler)
    print(f"[fm_budget_tracker] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
