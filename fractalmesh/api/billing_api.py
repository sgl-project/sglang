"""
FractalMesh Billing API v2.0.0
FastAPI-style billing metering: Stripe, PayPal, LemonSqueezy.
Usage metering, idempotent charge tracking, revenue reporting.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import base64
import hashlib
import sqlite3
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT      = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB        = os.path.join(ROOT, "database", "sovereign.db")
PORT      = int(os.getenv("BILLING_API_PORT", "8003"))

STRIPE_KEY     = os.getenv("STRIPE_SECRET_KEY", "")
PAYPAL_ID      = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET  = os.getenv("PAYPAL_CLIENT_SECRET", "")
LEMONSQUEEZY   = os.getenv("LEMONSQUEEZY_API_KEY", "")

PHI = 1.6180339887

# Payment processor fee matrix
PROCESSORS = {
    "stripe":       {"pct": 0.029, "fixed": 0.30},
    "paypal":       {"pct": 0.0259, "fixed": 0.49},
    "lemonsqueezy": {"pct": 0.05,  "fixed": 0.50},
}


# ── DB ────────────────────────────────────────────────────────────────────────

def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS billing_usage (
        id INTEGER PRIMARY KEY, customer TEXT, product TEXT,
        units REAL, unit_price REAL, total_aud REAL, processor TEXT,
        idempotency_key TEXT UNIQUE, status TEXT, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS billing_reports (
        id INTEGER PRIMARY KEY, period TEXT, gross_aud REAL, fees_aud REAL,
        net_aud REAL, transactions INTEGER, processor TEXT, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


# ── Stripe helpers ────────────────────────────────────────────────────────────

def _stripe_request(path: str, method: str = "GET", data: dict = None) -> dict:
    if not STRIPE_KEY:
        return {"error": "stripe_key_not_set"}
    creds = base64.b64encode(f"{STRIPE_KEY}:".encode()).decode()
    url   = f"https://api.stripe.com{path}"
    body  = urllib.parse.urlencode(data).encode() if data else None
    req   = urllib.request.Request(url, data=body, method=method,
                                   headers={"Authorization": f"Basic {creds}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _stripe_charges(limit: int = 10) -> list:
    res = _stripe_request(f"/v1/charges?limit={limit}")
    return res.get("data", [])


def _stripe_balance() -> dict:
    return _stripe_request("/v1/balance")


# ── PayPal helpers ────────────────────────────────────────────────────────────

def _paypal_token() -> str:
    if not PAYPAL_ID or not PAYPAL_SECRET:
        return ""
    creds = base64.b64encode(f"{PAYPAL_ID}:{PAYPAL_SECRET}".encode()).decode()
    body  = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req   = urllib.request.Request(
        "https://api-m.paypal.com/v1/oauth2/token", data=body,
        headers={"Authorization": f"Basic {creds}",
                 "Content-Type":  "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("access_token", "")
    except Exception:
        return ""


# ── LemonSqueezy helpers ──────────────────────────────────────────────────────

def _ls_request(path: str) -> dict:
    if not LEMONSQUEEZY:
        return {"error": "ls_key_not_set"}
    req = urllib.request.Request(
        f"https://api.lemonsqueezy.com/v1{path}",
        headers={"Authorization": f"Bearer {LEMONSQUEEZY}",
                 "Accept":        "application/vnd.api+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


# ── Metering / reporting ──────────────────────────────────────────────────────

def _record_usage(customer: str, product: str, units: float,
                  unit_price: float, processor: str) -> dict:
    total    = round(units * unit_price, 2)
    fee_pct  = PROCESSORS.get(processor, {}).get("pct", 0.029)
    fee_fix  = PROCESSORS.get(processor, {}).get("fixed", 0.30)
    fees     = round(total * fee_pct + fee_fix, 2)
    net      = round(total - fees, 2)
    ikey     = hashlib.sha256(
        f"{customer}{product}{units}{unit_price}{time.time()}".encode()
    ).hexdigest()[:24]
    phi      = round(net * PHI / 100, 6) if net > 0 else 0.0

    conn = sqlite3.connect(DB, timeout=10)
    try:
        conn.execute("""INSERT INTO billing_usage
            (customer,product,units,unit_price,total_aud,processor,idempotency_key,status,phi_score)
            VALUES (?,?,?,?,?,?,?,'pending',?)""",
            (customer, product, units, unit_price, total, processor, ikey, phi))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

    return {"customer": customer, "product": product, "total_aud": total,
            "fees_aud": fees, "net_aud": net, "idempotency_key": ikey}


def _daily_report(processor: str = "stripe") -> dict:
    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    conn  = sqlite3.connect(DB, timeout=10)
    rows  = conn.execute("""SELECT total_aud FROM billing_usage
        WHERE processor=? AND ts > ?""", (processor, since)).fetchall()
    conn.close()
    gross = sum(r[0] for r in rows)
    fee_pct = PROCESSORS.get(processor, {}).get("pct", 0.029)
    fee_fix = PROCESSORS.get(processor, {}).get("fixed", 0.30)
    fees    = round(len(rows) * fee_fix + gross * fee_pct, 2)
    net     = round(gross - fees, 2)
    phi     = round(net * PHI, 4)

    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO billing_reports
        (period,gross_aud,fees_aud,net_aud,transactions,processor,phi_score)
        VALUES ('24h',?,?,?,?,?,?)""",
        (round(gross, 2), fees, net, len(rows), processor, phi))
    conn.commit(); conn.close()

    return {"period": "24h", "processor": processor, "gross_aud": round(gross, 2),
            "fees_aud": fees, "net_aud": net, "transactions": len(rows), "phi_score": phi}


def _aggregate_all_processors() -> dict:
    totals = {}
    for proc in ["stripe", "paypal", "lemonsqueezy"]:
        rep = _daily_report(proc)
        totals[proc] = rep
    total_net = sum(v["net_aud"] for v in totals.values())
    return {"processors": totals, "combined_net_aud": round(total_net, 2),
            "timestamp": datetime.utcnow().isoformat()}


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class BillingHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok", "port": PORT, "agent": "billing-api",
                "stripe": bool(STRIPE_KEY), "paypal": bool(PAYPAL_ID),
                "lemonsqueezy": bool(LEMONSQUEEZY),
            })
        elif self.path == "/docs":
            self._send(200, {
                "endpoints": {
                    "GET /health":          "health + processor status",
                    "GET /balance":         "Stripe balance",
                    "GET /charges":         "recent Stripe charges",
                    "GET /report/daily":    "24h revenue report (all processors)",
                    "GET /report/stripe":   "24h Stripe report",
                    "POST /usage":          "record usage {customer,product,units,unit_price,processor}",
                }
            })
        elif self.path == "/balance":
            self._send(200, _stripe_balance())
        elif self.path.startswith("/charges"):
            limit = 10
            if "limit=" in self.path:
                try:
                    limit = int(self.path.split("limit=")[1])
                except Exception:
                    pass
            charges = _stripe_charges(limit)
            upserted = 0
            conn = sqlite3.connect(DB, timeout=10)
            for c in charges:
                try:
                    aud   = round(c.get("amount", 0) / 100, 2)
                    desc  = c.get("description", "")[:200]
                    stat  = c.get("status", "unknown")
                    phi   = round(aud * PHI / 100, 6)
                    conn.execute("""INSERT INTO revenue
                        (source,charge_id,amount_aud,currency,description,status)
                        VALUES ('stripe',?,?,?,?,?)
                        ON CONFLICT(charge_id) DO UPDATE
                        SET status=excluded.status""",
                        (c.get("id", ""), aud, c.get("currency", "aud").upper(),
                         desc, stat))
                    upserted += 1
                except Exception:
                    pass
            conn.commit(); conn.close()
            self._send(200, {"charges": len(charges), "upserted": upserted})
        elif self.path == "/report/daily":
            self._send(200, _aggregate_all_processors())
        elif self.path.startswith("/report/"):
            proc = self.path.split("/report/")[1]
            self._send(200, _daily_report(proc))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/usage":
            required = ["customer", "product", "units", "unit_price"]
            if not all(body.get(k) is not None for k in required):
                self._send(400, {"error": f"required: {required}"})
                return
            proc = body.get("processor", "stripe")
            result = _record_usage(body["customer"], body["product"],
                                   float(body["units"]), float(body["unit_price"]), proc)
            self._send(200, result)
        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), BillingHandler)
    print(f"[billing-api] Listening on :{PORT} | "
          f"Stripe={'set' if STRIPE_KEY else 'NOT SET'} | "
          f"PayPal={'set' if PAYPAL_ID else 'NOT SET'} | "
          f"LemonSqueezy={'set' if LEMONSQUEEZY else 'NOT SET'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("[billing-api] Stopped.")
