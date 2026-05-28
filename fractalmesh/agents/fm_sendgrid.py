"""
FractalMesh OMEGA Titan — SendGrid Mail API v3 Agent
Port: 7817
"""

import json
import logging
import os
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    with open(_VAULT) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / "fmsaas"
DB_PATH  = BASE_DIR / "database" / "sovereign.db"
LOG_PATH = BASE_DIR / "logs" / "fm_sendgrid.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

PORT           = int(os.environ.setdefault("SENDGRID_PORT", "7817"))
API_KEY        = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL     = os.getenv("SENDGRID_FROM_EMAIL", "")
BASE_URL       = "https://api.sendgrid.com/v3"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_sendgrid")

# ---------------------------------------------------------------------------
# SQLite — WAL mode
# ---------------------------------------------------------------------------
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sendgrid_sends (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            to_email   TEXT,
            subject    TEXT,
            status     TEXT,
            message_id TEXT,
            ts         TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


_db = _get_db()

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _sg_headers(extra: dict = None) -> dict:
    h = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _sg_get(path: str, params: dict = None) -> tuple[int, dict | list]:
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"{BASE_URL}{path}{qs}"
    req = urllib.request.Request(url, headers=_sg_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode())


def _sg_post(path: str, payload: dict) -> tuple[int, dict | str]:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=_sg_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            message_id = resp.headers.get("X-Message-Id", "")
            body = json.loads(raw.decode()) if raw else {}
            return resp.status, body, message_id
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, raw, ""


def _sg_post_no_raise(path: str, payload: dict):
    """Post and return (status, body, message_id) without raising on 4xx."""
    return _sg_post(path, payload)


# ---------------------------------------------------------------------------
# Mail payload builders
# ---------------------------------------------------------------------------
def _build_mail_payload(
    to_email: str,
    to_name: str,
    subject: str,
    text_body: str,
    html_body: str,
    from_email: str = None,
    from_name: str = None,
) -> dict:
    from_email = from_email or FROM_EMAIL
    return {
        "personalizations": [
            {
                "to": [{"email": to_email, "name": to_name or to_email}],
            }
        ],
        "from": {"email": from_email, "name": from_name or "FractalMesh"},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": text_body or " "},
            {"type": "text/html",  "value": html_body  or text_body or " "},
        ],
    }


def _build_template_payload(
    to_email: str,
    to_name: str,
    from_email: str,
    from_name: str,
    template_id: str,
    dynamic_data: dict,
) -> dict:
    from_email = from_email or FROM_EMAIL
    return {
        "personalizations": [
            {
                "to": [{"email": to_email, "name": to_name or to_email}],
                "dynamic_template_data": dynamic_data or {},
            }
        ],
        "from": {"email": from_email, "name": from_name or "FractalMesh"},
        "template_id": template_id,
    }


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_running = True


def _handle_signal(signum, frame):
    global _running
    log.info("Received signal %s — shutting down.", signum)
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
def _json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode())


def _send_json(handler: BaseHTTPRequestHandler, code: int, payload):
    body = json.dumps(payload, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class SendGridHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/health":
                self._health()
            elif path == "/templates":
                self._templates()
            elif path == "/stats":
                start = qs.get("start_date", [""])[0]
                end   = qs.get("end_date",   [""])[0]
                self._stats(start, end)
            elif path == "/contacts":
                query = qs.get("q", [""])[0]
                self._contacts(query)
            else:
                _send_json(self, 404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode() if hasattr(exc, "read") else str(exc)
            _send_json(self, exc.code, {"error": str(exc), "detail": detail})
        except Exception as exc:
            log.exception("Unhandled error in GET %s", path)
            _send_json(self, 500, {"error": str(exc)})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/send":
                self._send()
            elif path == "/bulk":
                self._bulk()
            elif path == "/template":
                self._template()
            elif path == "/contact":
                self._contact()
            elif path == "/suppression":
                self._suppression()
            else:
                _send_json(self, 404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode() if hasattr(exc, "read") else str(exc)
            _send_json(self, exc.code, {"error": str(exc), "detail": detail})
        except Exception as exc:
            log.exception("Unhandled error in POST %s", path)
            _send_json(self, 500, {"error": str(exc)})

    # ---- handlers ----

    def _health(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            _, stats = _sg_get("/stats", params={"start_date": today, "aggregated_by": "day"})
        except Exception as exc:
            stats = {"error": str(exc)}
        _send_json(self, 200, {
            "status": "ok",
            "agent": "fm_sendgrid",
            "port": PORT,
            "stats_today": stats,
        })

    def _send(self):
        body = _json_body(self)
        to_email   = body.get("to_email", "")
        to_name    = body.get("to_name", "")
        subject    = body.get("subject", "(no subject)")
        text_body  = body.get("text", "")
        html_body  = body.get("html", "")
        from_email = body.get("from_email", FROM_EMAIL)
        from_name  = body.get("from_name", "FractalMesh")

        if not to_email:
            _send_json(self, 400, {"error": "to_email required"})
            return

        payload = _build_mail_payload(
            to_email, to_name, subject, text_body, html_body, from_email, from_name
        )
        status, result, msg_id = _sg_post("/mail/send", payload)
        db_status = "sent" if status in (200, 202) else "failed"
        _db.execute(
            "INSERT INTO sendgrid_sends (to_email, subject, status, message_id) VALUES (?,?,?,?)",
            (to_email, subject, db_status, msg_id),
        )
        _db.commit()
        log.info("Email sent to %s — status %d — message_id %s", to_email, status, msg_id)
        _send_json(self, status if status in (200, 202) else 500, {
            "status": db_status,
            "message_id": msg_id,
            "to": to_email,
            "subject": subject,
            "sg_response": result,
        })

    def _bulk(self):
        body = _json_body(self)
        recipients = body.get("recipients", [])  # [{email, name}]
        subject    = body.get("subject", "(no subject)")
        text_body  = body.get("text", "")
        html_body  = body.get("html", "")
        from_email = body.get("from_email", FROM_EMAIL)
        from_name  = body.get("from_name", "FractalMesh")
        delay      = float(body.get("delay", 1.0))

        if not recipients:
            _send_json(self, 400, {"error": "recipients list required"})
            return

        results = []
        for rec in recipients:
            to_email = rec.get("email", "")
            to_name  = rec.get("name", "")
            if not to_email:
                results.append({"email": to_email, "status": "skipped", "reason": "no email"})
                continue

            payload = _build_mail_payload(
                to_email, to_name, subject, text_body, html_body, from_email, from_name
            )
            sg_status, sg_result, msg_id = _sg_post("/mail/send", payload)
            db_status = "sent" if sg_status in (200, 202) else "failed"
            _db.execute(
                "INSERT INTO sendgrid_sends (to_email, subject, status, message_id) VALUES (?,?,?,?)",
                (to_email, subject, db_status, msg_id),
            )
            _db.commit()
            results.append({
                "email": to_email,
                "status": db_status,
                "message_id": msg_id,
                "sg_status": sg_status,
            })
            log.info("Bulk send to %s — status %d", to_email, sg_status)
            if delay > 0:
                time.sleep(delay)

        _send_json(self, 200, {"sent": len(results), "results": results})

    def _template(self):
        body = _json_body(self)
        to_email      = body.get("to_email", "")
        to_name       = body.get("to_name", "")
        from_email    = body.get("from_email", FROM_EMAIL)
        from_name     = body.get("from_name", "FractalMesh")
        template_id   = body.get("template_id", "")
        dynamic_data  = body.get("dynamic_template_data", {})

        if not to_email or not template_id:
            _send_json(self, 400, {"error": "to_email and template_id required"})
            return

        payload = _build_template_payload(
            to_email, to_name, from_email, from_name, template_id, dynamic_data
        )
        sg_status, sg_result, msg_id = _sg_post("/mail/send", payload)
        db_status = "sent" if sg_status in (200, 202) else "failed"
        _db.execute(
            "INSERT INTO sendgrid_sends (to_email, subject, status, message_id) VALUES (?,?,?,?)",
            (to_email, f"[template:{template_id}]", db_status, msg_id),
        )
        _db.commit()
        log.info("Template email to %s (template %s) — status %d", to_email, template_id, sg_status)
        _send_json(self, sg_status if sg_status in (200, 202) else 500, {
            "status": db_status,
            "message_id": msg_id,
            "template_id": template_id,
            "to": to_email,
        })

    def _templates(self):
        _, data = _sg_get("/templates", params={"generations": "dynamic", "page_size": "50"})
        _send_json(self, 200, data)

    def _stats(self, start_date: str, end_date: str):
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
        params = {
            "start_date": start_date,
            "end_date":   end_date,
            "aggregated_by": "day",
        }
        _, data = _sg_get("/stats", params=params)

        # Flatten totals for convenience
        totals = {"delivered": 0, "opens": 0, "clicks": 0, "bounces": 0, "requests": 0}
        if isinstance(data, list):
            for day in data:
                for stat in day.get("stats", []):
                    m = stat.get("metrics", {})
                    totals["delivered"] += m.get("delivered", 0)
                    totals["opens"]     += m.get("opens", 0)
                    totals["clicks"]    += m.get("clicks", 0)
                    totals["bounces"]   += m.get("bounces", 0)
                    totals["requests"]  += m.get("requests", 0)

        _send_json(self, 200, {
            "start_date": start_date,
            "end_date":   end_date,
            "totals": totals,
            "daily": data,
        })

    def _contact(self):
        body = _json_body(self)
        email      = body.get("email", "")
        first_name = body.get("first_name", "")
        last_name  = body.get("last_name", "")
        list_ids   = body.get("list_ids", [])

        if not email:
            _send_json(self, 400, {"error": "email required"})
            return

        payload = {
            "contacts": [{
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
            }],
        }
        if list_ids:
            payload["list_ids"] = list_ids

        sg_status, sg_result, _ = _sg_post("/marketing/contacts", payload)
        log.info("Contact added: %s — status %d", email, sg_status)
        _send_json(self, sg_status if sg_status < 300 else 500, {
            "status": "added" if sg_status < 300 else "failed",
            "email": email,
            "sg_response": sg_result,
        })

    def _contacts(self, query: str):
        if not query:
            _send_json(self, 400, {"error": "q parameter required"})
            return
        payload = {"query": f'email LIKE \'%{query}%\''}
        url = f"{BASE_URL}/marketing/contacts/search"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers=_sg_headers(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        _send_json(self, 200, result)

    def _suppression(self):
        body = _json_body(self)
        email = body.get("email", "")
        if not email:
            _send_json(self, 400, {"error": "email required"})
            return
        payload = {"recipient_emails": [email]}
        sg_status, sg_result, _ = _sg_post("/asm/suppressions/global", payload)
        log.info("Suppression added: %s — status %d", email, sg_status)
        _send_json(self, sg_status if sg_status < 300 else 500, {
            "status": "suppressed" if sg_status < 300 else "failed",
            "email": email,
            "sg_response": sg_result,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    server = HTTPServer(("0.0.0.0", PORT), SendGridHandler)
    log.info("fm_sendgrid listening on port %d", PORT)
    while _running:
        server.handle_request()
    log.info("fm_sendgrid stopped.")


if __name__ == "__main__":
    main()
