#!/usr/bin/env python3
"""
FractalMesh Circle.so Agent — Port 7798
Circle Community Platform: Members + Spaces + Posts + DMs + Webhooks

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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PORT = 7798
AGENT_NAME = "fm_circle"
VAULT_PATH = Path.home() / ".secrets" / "fractal.env"
ROOT = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"
CIRCLE_BASE = "https://app.circle.so/api/v1"


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
        CREATE TABLE IF NOT EXISTS circle_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type   TEXT NOT NULL,
            member_email TEXT NOT NULL DEFAULT '',
            space_id     TEXT NOT NULL DEFAULT '',
            post_id      TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'received',
            ts           TEXT NOT NULL DEFAULT (datetime('now'))
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


def _log_event(event_type: str, member_email: str = "", space_id: str = "",
               post_id: str = "", status: str = "received") -> None:
    con = db_con()
    con.execute(
        "INSERT INTO circle_events (event_type, member_email, space_id, post_id, status) "
        "VALUES (?,?,?,?,?)",
        (event_type, member_email, str(space_id), str(post_id), status),
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Circle API helper
# ---------------------------------------------------------------------------
def _api_key() -> str:
    key = os.environ.get("CIRCLE_API_KEY", "")
    if not key:
        raise RuntimeError("CIRCLE_API_KEY not set in vault")
    return key


def _community_id() -> str:
    cid = os.environ.get("CIRCLE_COMMUNITY_ID", "")
    if not cid:
        raise RuntimeError("CIRCLE_COMMUNITY_ID not set in vault")
    return cid


def _circle(method: str, path: str, body: dict | None = None,
             params: dict | None = None) -> dict | list:
    """Generic Circle API call with Bearer token."""
    url = f"{CIRCLE_BASE}{path}"
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Authorization": f"Bearer {_api_key()}",
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
        raise RuntimeError(
            f"Circle {method} {path} → {exc.code}: {body_bytes.decode()}"
        ) from exc


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class CircleHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
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
                    "community_id": os.environ.get("CIRCLE_COMMUNITY_ID", ""),
                    "configured": bool(os.environ.get("CIRCLE_API_KEY")),
                })

            elif path == "/members":
                page = int(qs.get("page", 1))
                result = _circle("GET", "/community_members", params={
                    "community_id": _community_id(),
                    "per_page": 50,
                    "page": page,
                })
                self._send(200, result)

            elif path == "/spaces":
                result = _circle("GET", "/spaces", params={
                    "community_id": _community_id(),
                })
                self._send(200, result)

            elif path == "/posts":
                space_id = qs.get("space_id", "")
                if not space_id:
                    self._send(400, {"error": "space_id query param required"})
                    return
                page = int(qs.get("page", 1))
                result = _circle("GET", "/posts", params={
                    "community_id": _community_id(),
                    "space_id": space_id,
                    "per_page": 20,
                    "page": page,
                })
                self._send(200, result)

            elif path == "/events":
                con = db_con()
                rows = con.execute(
                    "SELECT * FROM circle_events ORDER BY ts DESC LIMIT 200"
                ).fetchall()
                con.close()
                self._send(200, [dict(r) for r in rows])

            elif path == "/analytics":
                # Member count
                members_result = _circle("GET", "/community_members", params={
                    "community_id": _community_id(),
                    "per_page": 1,
                    "page": 1,
                })
                member_count = members_result.get("meta", {}).get("total_count", 0) \
                    if isinstance(members_result, dict) else 0

                # Spaces
                spaces_result = _circle("GET", "/spaces", params={
                    "community_id": _community_id(),
                })
                spaces = spaces_result if isinstance(spaces_result, list) else []
                active_spaces = [s for s in spaces if s.get("is_hidden") is False]

                # Recent posts from DB events
                con = db_con()
                recent_posts = con.execute(
                    "SELECT COUNT(*) FROM circle_events WHERE event_type='post_created' "
                    "AND ts >= datetime('now', '-7 days')"
                ).fetchone()[0]
                con.close()

                self._send(200, {
                    "member_count": member_count,
                    "total_spaces": len(spaces),
                    "active_spaces": len(active_spaces),
                    "posts_last_7_days": recent_posts,
                })

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

            if path == "/invite":
                email    = body["email"]
                name     = body.get("name", "")
                space_id = body.get("space_id", "")

                payload: dict = {
                    "community_id": _community_id(),
                    "email": email,
                    "name": name,
                }
                if space_id:
                    payload["space_ids"] = [space_id]

                result = _circle("POST", "/community_members/invite", body=payload)
                _log_event("member_invited", member_email=email,
                           space_id=str(space_id), status="invited")
                self._send(201, result)

            elif path == "/post":
                space_id  = str(body["space_id"])
                title     = body["title"]
                post_body = body["body"]
                published = body.get("published", True)

                payload = {
                    "community_id": _community_id(),
                    "space_id": space_id,
                    "name": title,
                    "body": post_body,
                    "status": "published" if published else "draft",
                }
                result = _circle("POST", "/posts", body=payload)
                post_id = str(result.get("id", "")) if isinstance(result, dict) else ""
                _log_event("post_created", space_id=space_id, post_id=post_id,
                           status="published" if published else "draft")
                self._send(201, result)

            elif path == "/dm":
                member_id   = str(body["member_id"])
                dm_body     = body["body"]

                payload = {
                    "community_id": _community_id(),
                    "receiver_id": member_id,
                    "body": dm_body,
                }
                result = _circle("POST", "/direct_messages", body=payload)
                _log_event("dm_sent", space_id="", post_id=member_id, status="sent")
                self._send(201, result)

            elif path == "/webhook":
                event_type   = body.get("type", "unknown")
                member_email = ""
                space_id     = ""
                post_id      = ""

                # Normalise common Circle webhook shapes
                if event_type == "member_joined":
                    member_email = body.get("data", {}).get("email", "")
                    space_id     = str(body.get("data", {}).get("space_id", ""))

                elif event_type == "post_created":
                    post_id  = str(body.get("data", {}).get("id", ""))
                    space_id = str(body.get("data", {}).get("space_id", ""))

                _log_event(event_type, member_email=member_email,
                           space_id=space_id, post_id=post_id)
                print(
                    f"[{AGENT_NAME}] Webhook: {event_type} | "
                    f"member={member_email} space={space_id} post={post_id}",
                    flush=True,
                )
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

    _server = HTTPServer(("0.0.0.0", PORT), CircleHandler)
    print(f"[{AGENT_NAME}] Listening on http://0.0.0.0:{PORT}", flush=True)
    _server.serve_forever()
    print(f"[{AGENT_NAME}] Stopped.", flush=True)


if __name__ == "__main__":
    main()
