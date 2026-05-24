#!/usr/bin/env python3
"""
FractalMesh PayPal Agent — Port 7797
PayPal REST API: Orders v2 + Payouts + Webhooks

Author : Samuel James Hiotis | ABN 56 628 117 363
System : FractalMesh SaaS Platform
"""

import json
import os
import signal
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PORT = 7797
AGENT_NAME = "fm_paypal"
VAULT_PATH = Path.home() / ".secrets" / "fractal.env"
ROOT = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"
TOKEN_CACHE: dict = {"token": None, "expires_at": 0.0}
TOKEN_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Vault loader
# ---------------------------------------------------------------------------
def load_vault(path: Path) -> None:
    """Load key=value pairs from vault file into os.environ."""
    if not path.exists():
        print(f"[{AGENT_NAME}] WARN: vault not found at {path}", flush=True)
        return
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    print(f"[{AGENT_NAME}] Vault loaded from {path}", flush=True)


# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------
def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS paypal_orders (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT    NOT NULL,
            amount   REAL    NOT NULL,
            currency TEXT    NOT NULL DEFAULT 'AUD',
            status   TEXT    NOT NULL DEFAULT 'CREATED',
            ts       TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS paypal_payouts (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            payout_id TEXT    NOT NULL,
            recipient TEXT    NOT NULL,
            amount    REAL    NOT NULL,
            status    TEXT    NOT NULL DEFAULT 'PENDING',
            ts        TEXT    NOT NULL DEFAULT (datetime('now'))
        );
    """)
    con.commit()
    con.close()
    print(f"[{AGENT_NAME}] DB initialised at {DB_PATH}", flush=True)


def db_con() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# PayPal helpers
# ---------------------------------------------------------------------------
def _base_url() -> str:
    sandbox = os.environ.get("PAYPAL_SANDBOX", "false").lower() == "true"
    return "https://api-m.sandbox.paypal.com" if sandbox else "https://api-m.paypal.com"


def _token() -> str:
    """Return a cached OAuth2 token; refresh if expired."""
    with TOKEN_LOCK:
        now = time.time()
        if TOKEN_CACHE["token"] and now < TOKEN_CACHE["expires_at"]:
            return TOKEN_CACHE["token"]

        client_id = os.environ.get("PAYPAL_CLIENT_ID", "")
        client_secret = os.environ.get("PAYPAL_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise RuntimeError("PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET not set")

        credentials = b64encode(f"{client_id}:{client_secret}".encode()).decode()
        url = f"{_base_url()}/v1/oauth2/token"
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())

        TOKEN_CACHE["token"] = payload["access_token"]
        # Cache for 8 hours minus a small buffer
        TOKEN_CACHE["expires_at"] = now + min(payload.get("expires_in", 28800), 28800) - 60
        return TOKEN_CACHE["token"]


def _pp(method: str, path: str, body: dict | None = None) -> dict:
    """Generic PayPal API call with Bearer token."""
    url = f"{_base_url()}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        raise RuntimeError(f"PayPal {method} {path} → {exc.code}: {body_bytes.decode()}") from exc


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class PayPalHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        print(f"[{AGENT_NAME}] {self.address_string()} {fmt % args}", flush=True)

    # ---- utility -----------------------------------------------------------

    def _send(self, code: int, payload: dict | list | str) -> None:
        if isinstance(payload, str):
            body = payload.encode()
            ct = "text/plain"
        else:
            body = json.dumps(payload, indent=2).encode()
            ct = "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def _qs(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def _path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    # ---- GET ---------------------------------------------------------------

    def do_GET(self):
        path = self._path()
        qs = self._qs()
        try:
            if path == "/health":
                self._send(200, {
                    "status": "ok",
                    "agent": AGENT_NAME,
                    "port": PORT,
                    "sandbox": os.environ.get("PAYPAL_SANDBOX", "false").lower() == "true",
                    "configured": bool(os.environ.get("PAYPAL_CLIENT_ID")),
                })

            elif path == "/balance":
                result = _pp("GET", "/v1/reporting/balances")
                self._send(200, result)

            elif path == "/orders":
                con = db_con()
                rows = con.execute(
                    "SELECT * FROM paypal_orders ORDER BY ts DESC LIMIT 100"
                ).fetchall()
                con.close()
                self._send(200, [dict(r) for r in rows])

            elif path == "/transactions":
                start = qs.get("start", "")
                params = {"start_date": start} if start else {}
                qstr = ("?" + urllib.parse.urlencode(params)) if params else ""
                result = _pp("GET", f"/v1/reporting/transactions{qstr}")
                self._send(200, result)

            else:
                self._send(404, {"error": "not found", "path": path})

        except Exception as exc:
            print(f"[{AGENT_NAME}] ERROR GET {path}: {exc}", flush=True)
            self._send(500, {"error": str(exc)})

    # ---- POST --------------------------------------------------------------

    def do_POST(self):
        path = self._path()
        try:
            body = self._read_json()

            if path == "/order":
                amount   = str(body["amount"])
                currency = body.get("currency", "AUD").upper()
                desc     = body.get("description", "FractalMesh payment")
                ret_url  = body.get("return_url", "https://fractalmesh.net/success")
                can_url  = body.get("cancel_url", "https://fractalmesh.net/cancel")

                payload = {
                    "intent": "CAPTURE",
                    "purchase_units": [{
                        "amount": {"currency_code": currency, "value": amount},
                        "description": desc,
                    }],
                    "application_context": {
                        "return_url": ret_url,
                        "cancel_url": can_url,
                        "brand_name": "FractalMesh",
                    },
                }
                result = _pp("POST", "/v2/checkout/orders", payload)
                order_id = result.get("id", "")

                con = db_con()
                con.execute(
                    "INSERT INTO paypal_orders (order_id, amount, currency, status) VALUES (?,?,?,?)",
                    (order_id, float(amount), currency, result.get("status", "CREATED")),
                )
                con.commit()
                con.close()
                self._send(201, result)

            elif path == "/capture":
                order_id = body["order_id"]
                result = _pp("POST", f"/v2/checkout/orders/{order_id}/capture", {})
                status = result.get("status", "COMPLETED")

                con = db_con()
                con.execute(
                    "UPDATE paypal_orders SET status=? WHERE order_id=?",
                    (status, order_id),
                )
                con.commit()
                con.close()
                self._send(200, result)

            elif path == "/payout":
                recipient = body["recipient_email"]
                amount    = str(body["amount"])
                currency  = body.get("currency", "AUD").upper()
                note      = body.get("note", "FractalMesh payout")

                payload = {
                    "sender_batch_header": {
                        "sender_batch_id": f"FM_{int(time.time())}",
                        "email_subject": "You have a payment from FractalMesh",
                        "email_message": note,
                    },
                    "items": [{
                        "recipient_type": "EMAIL",
                        "amount": {"value": amount, "currency": currency},
                        "receiver": recipient,
                        "note": note,
                    }],
                }
                result = _pp("POST", "/v1/payments/payouts", payload)
                batch_id = result.get("batch_header", {}).get("payout_batch_id", "")

                con = db_con()
                con.execute(
                    "INSERT INTO paypal_payouts (payout_id, recipient, amount, status) VALUES (?,?,?,?)",
                    (batch_id, recipient, float(amount), "PENDING"),
                )
                con.commit()
                con.close()
                self._send(201, result)

            elif path == "/webhook":
                event_type = body.get("event_type", "UNKNOWN")
                resource   = body.get("resource", {})
                print(f"[{AGENT_NAME}] Webhook event: {event_type} | resource_id={resource.get('id', '')}", flush=True)

                # Update order/payout status if relevant
                if "PAYMENT.CAPTURE" in event_type:
                    order_id = resource.get("id", "")
                    status   = resource.get("status", event_type)
                    if order_id:
                        con = db_con()
                        con.execute(
                            "UPDATE paypal_orders SET status=? WHERE order_id=?",
                            (status, order_id),
                        )
                        con.commit()
                        con.close()

                elif "PAYMENT.PAYOUTS" in event_type:
                    payout_id = resource.get("payout_batch_id", "")
                    status    = resource.get("batch_header", {}).get("batch_status", event_type)
                    if payout_id:
                        con = db_con()
                        con.execute(
                            "UPDATE paypal_payouts SET status=? WHERE payout_id=?",
                            (status, payout_id),
                        )
                        con.commit()
                        con.close()

                self._send(200, {"received": True, "event_type": event_type})

            else:
                self._send(404, {"error": "not found", "path": path})

        except KeyError as exc:
            self._send(400, {"error": f"missing field: {exc}"})
        except Exception as exc:
            print(f"[{AGENT_NAME}] ERROR POST {path}: {exc}", flush=True)
            self._send(500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
_server: HTTPServer | None = None


def _shutdown(signum, _frame):
    print(f"[{AGENT_NAME}] Signal {signum} received — shutting down", flush=True)
    if _server:
        threading.Thread(target=_server.shutdown, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global _server

    load_vault(VAULT_PATH)
    init_db()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _server = HTTPServer(("0.0.0.0", PORT), PayPalHandler)
    print(f"[{AGENT_NAME}] Listening on http://0.0.0.0:{PORT}", flush=True)
    _server.serve_forever()
    print(f"[{AGENT_NAME}] Stopped.", flush=True)


if __name__ == "__main__":
    main()
