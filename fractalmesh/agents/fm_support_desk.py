#!/usr/bin/env python3
"""
fm_support_desk.py — FractalMesh OMEGA Titan Customer Support Desk (Port 7868)
AI-powered ticketing system with auto-classification, SLA enforcement, and KB.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, sqlite3, time, hashlib, hmac, threading, re, html
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
PORT             = int(os.getenv("SUPPORT_DESK_PORT", "7868"))
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM    = os.getenv("SENDGRID_FROM_EMAIL", "")
ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
MCP_PORT         = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET       = os.getenv("MCP_SECRET", "")
ADMIN_SECRET     = os.getenv("ADMIN_SECRET", "")

ROOT   = Path.home() / "fmsaas"
DB     = ROOT / "database" / "sovereign.db"
LOG    = ROOT / "logs" / "fm_support_desk.log"

ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Minimal logger
# ---------------------------------------------------------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SUPPORT] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("fm_support_desk")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _db()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS tickets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_ref      TEXT UNIQUE,
            customer_email  TEXT,
            customer_name   TEXT,
            subject         TEXT,
            status          TEXT DEFAULT 'open',
            priority        TEXT DEFAULT 'normal',
            category        TEXT,
            queue           TEXT DEFAULT 'general',
            sentiment_score REAL DEFAULT 0,
            ai_draft        TEXT,
            created_at      REAL,
            updated_at      REAL,
            resolved_at     REAL,
            sla_due_at      REAL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id    INTEGER,
            sender_type  TEXT,
            sender_email TEXT,
            content      TEXT,
            is_internal  INTEGER DEFAULT 0,
            created_at   REAL
        );

        CREATE TABLE IF NOT EXISTS agents_desk (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT UNIQUE,
            name         TEXT,
            queue        TEXT,
            active       INTEGER DEFAULT 1,
            tickets_open INTEGER DEFAULT 0,
            created_at   REAL
        );

        CREATE TABLE IF NOT EXISTS kb_articles (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            title         TEXT,
            content       TEXT,
            category      TEXT,
            tags          TEXT,
            helpful_count INTEGER DEFAULT 0,
            created_at    REAL
        );

        CREATE TABLE IF NOT EXISTS sla_rules (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            priority         TEXT UNIQUE,
            response_hours   REAL,
            resolution_hours REAL
        );
    """)
    conn.commit()

    # Seed SLA rules
    cur.execute("SELECT COUNT(*) FROM sla_rules")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT OR IGNORE INTO sla_rules (priority, response_hours, resolution_hours) VALUES (?,?,?)",
            [("high", 2.0, 8.0), ("normal", 8.0, 24.0), ("low", 24.0, 72.0)],
        )
        conn.commit()

    # Seed KB articles
    cur.execute("SELECT COUNT(*) FROM kb_articles")
    if cur.fetchone()[0] == 0:
        now = time.time()
        cur.executemany(
            "INSERT INTO kb_articles (title, content, category, tags, created_at) VALUES (?,?,?,?,?)",
            [
                (
                    "How to access your account",
                    "To access your FractalMesh OMEGA Titan account, navigate to the dashboard at "
                    "https://app.fractalmesh.io and enter your registered email and password. "
                    "If you have forgotten your password, click 'Forgot Password' and follow the "
                    "reset instructions sent to your email. Two-factor authentication can be enabled "
                    "under Account Settings > Security.",
                    "account",
                    "login,access,password,2fa,authentication",
                    now,
                ),
                (
                    "Payment and billing FAQ",
                    "FractalMesh OMEGA Titan accepts major credit cards (Visa, Mastercard, Amex) and "
                    "cryptocurrency (BTC, ETH, USDC) via our Coinbase Commerce integration. "
                    "Invoices are generated on the 1st of each month and emailed automatically. "
                    "Refund requests must be submitted within 14 days of the charge. "
                    "To update your payment method, go to Billing > Payment Methods in your dashboard. "
                    "For disputed charges, please open a support ticket with your invoice number.",
                    "billing",
                    "payment,billing,invoice,refund,credit card,crypto",
                    now,
                ),
                (
                    "API rate limits and quotas",
                    "The FractalMesh OMEGA Titan API enforces the following rate limits by default: "
                    "Free tier — 60 requests/minute, 10,000 requests/day. "
                    "Pro tier — 600 requests/minute, 500,000 requests/day. "
                    "Enterprise tier — custom limits negotiated per contract. "
                    "Rate limit headers are returned with every response: X-RateLimit-Limit, "
                    "X-RateLimit-Remaining, and X-RateLimit-Reset. "
                    "Exceeding limits returns HTTP 429. Quota increases can be requested via "
                    "the dashboard under API Settings > Request Quota Increase.",
                    "technical",
                    "api,rate-limit,quota,429,throttle",
                    now,
                ),
            ],
        )
        conn.commit()

    conn.close()


# ---------------------------------------------------------------------------
# Ticket reference generation
# ---------------------------------------------------------------------------
_ticket_counter_lock = threading.Lock()


def _next_ticket_ref() -> str:
    date_str = time.strftime("%Y%m%d")
    prefix = f"TKT-{date_str}-"
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ticket_ref FROM tickets WHERE ticket_ref LIKE ? ORDER BY ticket_ref DESC LIMIT 1",
            (prefix + "%",),
        )
        row = cur.fetchone()
        if row:
            last_seq = int(row[0].split("-")[-1])
            seq = last_seq + 1
        else:
            seq = 1
    finally:
        conn.close()
    return f"{prefix}{seq:04d}"


# ---------------------------------------------------------------------------
# Auto-classification
# ---------------------------------------------------------------------------
_URGENT_RE   = re.compile(r'\b(urgent|asap|critical|broken|down)\b', re.I)
_BILLING_RE  = re.compile(r'\b(billing|payment|refund|invoice|charge|credit)\b', re.I)
_QUESTION_RE = re.compile(r'\b(question|how|help|what|why|when|where)\b', re.I)
_TECH_RE     = re.compile(r'\b(api|error|bug|crash|install|config|technical|integration|code|token|key)\b', re.I)
_SALES_RE    = re.compile(r'\b(price|pricing|plan|upgrade|demo|trial|discount|quote|enterprise)\b', re.I)

_POSITIVE_WORDS = {'great', 'love', 'thanks', 'thank', 'perfect', 'awesome', 'excellent',
                   'wonderful', 'fantastic', 'helpful', 'amazing', 'good'}
_NEGATIVE_WORDS = {'angry', 'frustrated', 'terrible', 'broken', 'refund', 'awful',
                   'horrible', 'useless', 'worst', 'unacceptable', 'disappointed', 'bad'}

CATEGORY_TO_QUEUE = {
    "billing":   "billing",
    "technical": "technical",
    "sales":     "sales",
    "general":   "general",
}


def _classify_ticket(subject: str, body: str):
    text = f"{subject} {body}"

    # Priority
    if _URGENT_RE.search(text) or _BILLING_RE.search(text):
        priority = "high"
    elif _QUESTION_RE.search(text):
        priority = "normal"
    else:
        priority = "normal"

    # Category
    if _BILLING_RE.search(text):
        category = "billing"
    elif _TECH_RE.search(text):
        category = "technical"
    elif _SALES_RE.search(text):
        category = "sales"
    else:
        category = "general"

    queue = CATEGORY_TO_QUEUE.get(category, "general")

    # Sentiment
    words = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        sentiment = 0.0
    else:
        sentiment = round((pos - neg) / total, 4)

    return priority, category, queue, sentiment


def _get_sla_due(priority: str, created_at: float) -> float:
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT response_hours FROM sla_rules WHERE priority=?", (priority,))
        row = cur.fetchone()
        hours = row[0] if row else 8.0
    finally:
        conn.close()
    return created_at + hours * 3600.0


# ---------------------------------------------------------------------------
# AI draft generation
# ---------------------------------------------------------------------------
def _generate_ai_draft(subject: str, body: str) -> str:
    system_prompt = (
        "You are a helpful customer support agent for FractalMesh OMEGA Titan. "
        "Be concise, empathetic, and professional."
    )
    user_message = f"Customer ticket: {subject}\n\n{body}"

    # Try MCP bus first
    mcp_url = f"http://localhost:{MCP_PORT}/dispatch"
    payload = json.dumps({
        "intent": "ai_chat",
        "secret": MCP_SECRET,
        "payload": {
            "system": system_prompt,
            "message": user_message,
        },
    }).encode()
    try:
        req = Request(mcp_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            draft = data.get("response") or data.get("result") or data.get("text") or ""
            if draft:
                return draft
    except Exception:
        pass

    # Fallback: direct HTTP to ai_assistant on port 7865
    ai_url = "http://localhost:7865/chat"
    payload2 = json.dumps({"system": system_prompt, "message": user_message}).encode()
    try:
        req2 = Request(ai_url, data=payload2, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req2, timeout=15) as resp2:
            data2 = json.loads(resp2.read())
            draft = data2.get("response") or data2.get("reply") or data2.get("text") or ""
            if draft:
                return draft
    except Exception:
        pass

    # Final fallback: direct Anthropic API
    if ANTHROPIC_KEY:
        ant_url = "https://api.anthropic.com/v1/messages"
        ant_payload = json.dumps({
            "model": "claude-haiku-4-5",
            "max_tokens": 512,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }).encode()
        try:
            req3 = Request(
                ant_url,
                data=ant_payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urlopen(req3, timeout=20) as resp3:
                data3 = json.loads(resp3.read())
                content = data3.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
        except Exception as e:
            log.warning("Anthropic fallback error: %s", e)

    return ""


def _generate_ai_draft_async(ticket_id: int, subject: str, body: str):
    def _worker():
        try:
            draft = _generate_ai_draft(subject, body)
            if draft:
                conn = _db()
                try:
                    conn.execute(
                        "UPDATE tickets SET ai_draft=?, updated_at=? WHERE id=?",
                        (draft, time.time(), ticket_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            log.warning("AI draft async error: %s", e)

    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# SendGrid email helper
# ---------------------------------------------------------------------------
def _send_email(to_email: str, subject: str, body: str) -> bool:
    if not SENDGRID_API_KEY or not SENDGRID_FROM:
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }).encode()
    try:
        req = Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=payload,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except Exception as e:
        log.warning("SendGrid error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Background SLA checker
# ---------------------------------------------------------------------------
def _sla_checker_loop():
    while True:
        try:
            now = time.time()
            conn = _db()
            cur = conn.cursor()

            # SLA breach: open tickets past their SLA due time
            cur.execute(
                "SELECT ticket_ref, customer_email, subject FROM tickets "
                "WHERE status='open' AND sla_due_at IS NOT NULL AND sla_due_at < ?",
                (now,),
            )
            breaches = cur.fetchall()
            for b in breaches:
                log.warning("SLA BREACH: %s (%s) — %s", b[0], b[1], b[2])

            # Escalate tickets open > 48h to high priority
            cutoff_48h = now - 48 * 3600
            cur.execute(
                "UPDATE tickets SET priority='high', updated_at=? "
                "WHERE status='open' AND created_at < ? AND priority != 'high'",
                (now, cutoff_48h),
            )
            escalated = cur.rowcount
            if escalated:
                log.info("Escalated %d tickets open >48h to high priority", escalated)

            conn.commit()
            conn.close()
        except Exception as e:
            log.error("SLA checker error: %s", e)
        time.sleep(300)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _json(data) -> bytes:
    return json.dumps(data, default=str).encode()


def _parse_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _auth_admin(handler) -> bool:
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(handler.headers.get("X-Admin-Secret", ""), ADMIN_SECRET)


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


def _rows_to_list(rows) -> list:
    return [dict(r) for r in rows]


def _query_param(path: str, key: str, default=None):
    if "?" not in path:
        return default
    qs = path.split("?", 1)[1]
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k == key:
                return v
    return default


def _all_query_params(path: str) -> dict:
    params = {}
    if "?" not in path:
        return params
    qs = path.split("?", 1)[1]
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return params


def _is_agent(email: str) -> bool:
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM agents_desk WHERE email=? AND active=1", (email,))
        return cur.fetchone() is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class SupportDeskHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh/SupportDesk"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, data: dict):
        body = _json(data)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, data: dict):
        self._send(200, data)

    def _created(self, data: dict):
        self._send(201, data)

    def _bad(self, msg: str):
        self._send(400, {"error": msg})

    def _unauth(self):
        self._send(403, {"error": "forbidden"})

    def _not_found(self, msg: str = "not found"):
        self._send(404, {"error": msg})

    def _server_error(self, msg: str):
        self._send(500, {"error": msg})

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if path == "/health":
                self._handle_health()
            elif path == "/tickets":
                self._handle_list_tickets()
            elif path == "/tickets/search":
                self._handle_search_tickets()
            elif re.match(r'^/tickets/TKT-\d{8}-\d{4}$', path):
                ref = path.split("/")[-1]
                self._handle_get_ticket(ref)
            elif path == "/agents":
                self._handle_list_agents()
            elif path == "/kb":
                self._handle_list_kb()
            elif re.match(r'^/kb/\d+$', path):
                kb_id = int(path.split("/")[-1])
                self._handle_get_kb(kb_id)
            elif path == "/sla":
                self._handle_sla()
            elif path == "/stats":
                self._handle_stats()
            else:
                self._not_found()
        except Exception as e:
            log.error("GET %s error: %s", self.path, e)
            self._server_error(str(e))

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if path == "/tickets":
                self._handle_create_ticket()
            elif re.match(r'^/tickets/TKT-\d{8}-\d{4}/reply$', path):
                ref = path.split("/")[-2]
                self._handle_reply(ref)
            elif re.match(r'^/tickets/TKT-\d{8}-\d{4}/resolve$', path):
                ref = path.split("/")[-2]
                self._handle_resolve(ref)
            elif re.match(r'^/tickets/TKT-\d{8}-\d{4}/assign$', path):
                ref = path.split("/")[-2]
                self._handle_assign(ref)
            elif path == "/agents":
                self._handle_create_agent()
            elif path == "/kb":
                self._handle_create_kb()
            else:
                self._not_found()
        except Exception as e:
            log.error("POST %s error: %s", self.path, e)
            self._server_error(str(e))

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if re.match(r'^/kb/\d+/helpful$', path):
                kb_id = int(path.split("/")[-2])
                self._handle_kb_helpful(kb_id)
            else:
                self._not_found()
        except Exception as e:
            log.error("PUT %s error: %s", self.path, e)
            self._server_error(str(e))

    # -----------------------------------------------------------------------
    # GET handlers
    # -----------------------------------------------------------------------
    def _handle_health(self):
        now = time.time()
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM tickets WHERE status='open'")
            open_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tickets WHERE status='resolved'")
            resolved_count = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM tickets WHERE status='open' AND sla_due_at IS NOT NULL AND sla_due_at < ?",
                (now,),
            )
            breach_count = cur.fetchone()[0]
        finally:
            conn.close()
        self._ok({
            "status": "ok",
            "uptime_seconds": round(now - _START_TIME, 1),
            "tickets_open": open_count,
            "tickets_resolved": resolved_count,
            "sla_breaches": breach_count,
            "port": PORT,
        })

    def _handle_list_tickets(self):
        params = _all_query_params(self.path)
        conn = _db()
        try:
            cur = conn.cursor()
            conditions = []
            values = []
            for field in ("status", "priority", "category", "queue", "customer_email"):
                if field in params:
                    conditions.append(f"{field}=?")
                    values.append(params[field])
            limit = int(params.get("limit", 50))
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            cur.execute(
                f"SELECT * FROM tickets {where} ORDER BY created_at DESC LIMIT ?",
                values + [limit],
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        self._ok({"tickets": _rows_to_list(rows), "count": len(rows)})

    def _handle_search_tickets(self):
        q = _query_param(self.path, "q", "")
        if not q:
            self._bad("q parameter required")
            return
        q = q.replace("+", " ")
        pattern = f"%{q}%"
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT DISTINCT t.* FROM tickets t
                   LEFT JOIN messages m ON m.ticket_id = t.id
                   WHERE t.subject LIKE ? OR m.content LIKE ?
                   ORDER BY t.created_at DESC LIMIT 50""",
                (pattern, pattern),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        self._ok({"tickets": _rows_to_list(rows), "count": len(rows), "query": q})

    def _handle_get_ticket(self, ref: str):
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM tickets WHERE ticket_ref=?", (ref,))
            ticket = cur.fetchone()
            if not ticket:
                self._not_found(f"ticket {ref} not found")
                return
            ticket_dict = _row_to_dict(ticket)
            cur.execute(
                "SELECT * FROM messages WHERE ticket_id=? ORDER BY created_at ASC",
                (ticket_dict["id"],),
            )
            messages = _rows_to_list(cur.fetchall())
        finally:
            conn.close()
        self._ok({"ticket": ticket_dict, "messages": messages})

    def _handle_list_agents(self):
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM agents_desk ORDER BY name ASC")
            rows = cur.fetchall()
        finally:
            conn.close()
        self._ok({"agents": _rows_to_list(rows), "count": len(rows)})

    def _handle_list_kb(self):
        params = _all_query_params(self.path)
        conn = _db()
        try:
            cur = conn.cursor()
            conditions = []
            values = []
            if "category" in params:
                conditions.append("category=?")
                values.append(params["category"])
            if "tags" in params:
                conditions.append("tags LIKE ?")
                values.append(f"%{params['tags']}%")
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            cur.execute(
                f"SELECT * FROM kb_articles {where} ORDER BY helpful_count DESC, created_at DESC",
                values,
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        self._ok({"articles": _rows_to_list(rows), "count": len(rows)})

    def _handle_get_kb(self, kb_id: int):
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM kb_articles WHERE id=?", (kb_id,))
            row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            self._not_found(f"KB article {kb_id} not found")
            return
        self._ok({"article": _row_to_dict(row)})

    def _handle_sla(self):
        now = time.time()
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM sla_rules ORDER BY response_hours ASC")
            rules = _rows_to_list(cur.fetchall())
            cur.execute(
                "SELECT COUNT(*) FROM tickets WHERE status='open' AND sla_due_at IS NOT NULL AND sla_due_at < ?",
                (now,),
            )
            breach_count = cur.fetchone()[0]
        finally:
            conn.close()
        self._ok({"sla_rules": rules, "current_breaches": breach_count})

    def _handle_stats(self):
        conn = _db()
        try:
            cur = conn.cursor()
            # Ticket volume by day (last 30 days)
            cur.execute(
                """SELECT date(datetime(created_at,'unixepoch')) AS day, COUNT(*) AS count
                   FROM tickets
                   WHERE created_at > ?
                   GROUP BY day ORDER BY day DESC""",
                (time.time() - 30 * 86400,),
            )
            by_day = [{"day": r[0], "count": r[1]} for r in cur.fetchall()]

            # Avg resolution time (hours)
            cur.execute(
                "SELECT AVG(resolved_at - created_at) / 3600.0 FROM tickets WHERE resolved_at IS NOT NULL"
            )
            avg_res = cur.fetchone()[0]

            # Category breakdown
            cur.execute("SELECT category, COUNT(*) FROM tickets GROUP BY category")
            by_category = {r[0] or "unknown": r[1] for r in cur.fetchall()}

            # Priority breakdown
            cur.execute("SELECT priority, COUNT(*) FROM tickets GROUP BY priority")
            by_priority = {r[0] or "normal": r[1] for r in cur.fetchall()}

            # Total counts
            cur.execute("SELECT COUNT(*) FROM tickets")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tickets WHERE status='open'")
            total_open = cur.fetchone()[0]
        finally:
            conn.close()
        self._ok({
            "total_tickets": total,
            "total_open": total_open,
            "avg_resolution_hours": round(avg_res, 2) if avg_res else None,
            "volume_by_day": by_day,
            "by_category": by_category,
            "by_priority": by_priority,
        })

    # -----------------------------------------------------------------------
    # POST handlers
    # -----------------------------------------------------------------------
    def _handle_create_ticket(self):
        body = _parse_body(self)
        email = body.get("customer_email", "").strip()
        name  = body.get("customer_name", "").strip()
        subj  = body.get("subject", "").strip()
        msg   = body.get("body", "").strip()

        if not email or not subj or not msg:
            self._bad("customer_email, subject, and body are required")
            return

        now = time.time()
        with _ticket_counter_lock:
            ref = _next_ticket_ref()

        priority, category, queue, sentiment = _classify_ticket(subj, msg)
        sla_due = _get_sla_due(priority, now)

        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO tickets
                   (ticket_ref, customer_email, customer_name, subject, status,
                    priority, category, queue, sentiment_score, created_at, updated_at, sla_due_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ref, email, name, subj, "open",
                 priority, category, queue, sentiment, now, now, sla_due),
            )
            ticket_id = cur.lastrowid

            cur.execute(
                "INSERT INTO messages (ticket_id, sender_type, sender_email, content, is_internal, created_at) VALUES (?,?,?,?,?,?)",
                (ticket_id, "customer", email, msg, 0, now),
            )
            conn.commit()

            cur.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
            ticket = _row_to_dict(cur.fetchone())
        finally:
            conn.close()

        # Kick off AI draft generation asynchronously
        _generate_ai_draft_async(ticket_id, subj, msg)

        log.info("Ticket created: %s by %s (priority=%s category=%s)", ref, email, priority, category)
        self._created({"ticket": ticket, "message": "Ticket created. AI draft generating in background."})

    def _handle_reply(self, ref: str):
        body = _parse_body(self)
        content      = body.get("content", "").strip()
        sender_email = body.get("sender_email", "").strip()
        is_internal  = int(bool(body.get("is_internal", False)))

        if not content or not sender_email:
            self._bad("content and sender_email are required")
            return

        now = time.time()
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM tickets WHERE ticket_ref=?", (ref,))
            ticket = cur.fetchone()
            if not ticket:
                self._not_found(f"ticket {ref} not found")
                conn.close()
                return
            ticket_dict = _row_to_dict(ticket)

            is_agent_sender = _is_agent(sender_email)
            sender_type = "agent" if is_agent_sender else "customer"

            cur.execute(
                "INSERT INTO messages (ticket_id, sender_type, sender_email, content, is_internal, created_at) VALUES (?,?,?,?,?,?)",
                (ticket_dict["id"], sender_type, sender_email, content, is_internal, now),
            )
            cur.execute(
                "UPDATE tickets SET updated_at=? WHERE ticket_ref=?",
                (now, ref),
            )
            conn.commit()

            cur.execute(
                "SELECT * FROM messages WHERE ticket_id=? ORDER BY created_at DESC LIMIT 1",
                (ticket_dict["id"],),
            )
            msg_row = _row_to_dict(cur.fetchone())
        finally:
            conn.close()

        # If agent replied and not internal, optionally email the customer
        emailed = False
        if is_agent_sender and not is_internal:
            customer_email = ticket_dict.get("customer_email", "")
            if customer_email:
                emailed = _send_email(
                    customer_email,
                    f"Re: [{ref}] {ticket_dict.get('subject', '')}",
                    f"Dear {ticket_dict.get('customer_name', 'Customer')},\n\n{content}\n\n"
                    f"-- FractalMesh OMEGA Titan Support\nTicket: {ref}",
                )

        self._created({"message": msg_row, "emailed_customer": emailed})

    def _handle_resolve(self, ref: str):
        if not _auth_admin(self):
            self._unauth()
            return
        now = time.time()
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM tickets WHERE ticket_ref=?", (ref,))
            row = cur.fetchone()
            if not row:
                self._not_found(f"ticket {ref} not found")
                conn.close()
                return
            cur.execute(
                "UPDATE tickets SET status='resolved', resolved_at=?, updated_at=? WHERE ticket_ref=?",
                (now, now, ref),
            )
            conn.commit()
            cur.execute("SELECT * FROM tickets WHERE ticket_ref=?", (ref,))
            ticket = _row_to_dict(cur.fetchone())
        finally:
            conn.close()
        log.info("Ticket resolved: %s", ref)
        self._ok({"ticket": ticket})

    def _handle_assign(self, ref: str):
        if not _auth_admin(self):
            self._unauth()
            return
        body = _parse_body(self)
        agent_email = body.get("agent_email", "").strip()
        if not agent_email:
            self._bad("agent_email is required")
            return

        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM agents_desk WHERE email=? AND active=1", (agent_email,))
            if not cur.fetchone():
                self._not_found(f"active agent {agent_email} not found")
                conn.close()
                return

            cur.execute("SELECT id, queue FROM tickets WHERE ticket_ref=?", (ref,))
            trow = cur.fetchone()
            if not trow:
                self._not_found(f"ticket {ref} not found")
                conn.close()
                return
            ticket_id = trow[0]

            # Update ticket queue to agent's queue
            cur.execute("SELECT queue FROM agents_desk WHERE email=?", (agent_email,))
            aq = cur.fetchone()
            agent_queue = aq[0] if aq else "general"

            now = time.time()
            cur.execute(
                "UPDATE tickets SET queue=?, updated_at=? WHERE id=?",
                (agent_queue, now, ticket_id),
            )
            # Track open ticket count on agent
            cur.execute(
                "UPDATE agents_desk SET tickets_open = tickets_open + 1 WHERE email=?",
                (agent_email,),
            )
            conn.commit()
            cur.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
            ticket = _row_to_dict(cur.fetchone())
        finally:
            conn.close()
        log.info("Ticket %s assigned to %s", ref, agent_email)
        self._ok({"ticket": ticket, "assigned_to": agent_email})

    def _handle_create_agent(self):
        if not _auth_admin(self):
            self._unauth()
            return
        body = _parse_body(self)
        email = body.get("email", "").strip()
        name  = body.get("name", "").strip()
        queue = body.get("queue", "general").strip()
        if not email or not name:
            self._bad("email and name are required")
            return
        now = time.time()
        conn = _db()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO agents_desk (email, name, queue, active, tickets_open, created_at) VALUES (?,?,?,1,0,?)",
                    (email, name, queue, now),
                )
                conn.commit()
                agent_id = cur.lastrowid
            except sqlite3.IntegrityError:
                self._bad(f"agent {email} already exists")
                conn.close()
                return
            cur.execute("SELECT * FROM agents_desk WHERE id=?", (agent_id,))
            agent = _row_to_dict(cur.fetchone())
        finally:
            conn.close()
        log.info("Agent created: %s (%s)", email, name)
        self._created({"agent": agent})

    def _handle_create_kb(self):
        if not _auth_admin(self):
            self._unauth()
            return
        body = _parse_body(self)
        title    = body.get("title", "").strip()
        content  = body.get("content", "").strip()
        category = body.get("category", "general").strip()
        tags     = body.get("tags", "").strip()
        if not title or not content:
            self._bad("title and content are required")
            return
        now = time.time()
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO kb_articles (title, content, category, tags, helpful_count, created_at) VALUES (?,?,?,?,0,?)",
                (title, content, category, tags, now),
            )
            conn.commit()
            article_id = cur.lastrowid
            cur.execute("SELECT * FROM kb_articles WHERE id=?", (article_id,))
            article = _row_to_dict(cur.fetchone())
        finally:
            conn.close()
        log.info("KB article created: %s", title)
        self._created({"article": article})

    # -----------------------------------------------------------------------
    # PUT handlers
    # -----------------------------------------------------------------------
    def _handle_kb_helpful(self, kb_id: int):
        conn = _db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM kb_articles WHERE id=?", (kb_id,))
            if not cur.fetchone():
                self._not_found(f"KB article {kb_id} not found")
                conn.close()
                return
            cur.execute(
                "UPDATE kb_articles SET helpful_count = helpful_count + 1 WHERE id=?",
                (kb_id,),
            )
            conn.commit()
            cur.execute("SELECT helpful_count FROM kb_articles WHERE id=?", (kb_id,))
            count = cur.fetchone()[0]
        finally:
            conn.close()
        self._ok({"id": kb_id, "helpful_count": count})


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
def _run():
    _init_db()
    log.info("Support Desk DB initialised at %s", DB)

    sla_thread = threading.Thread(target=_sla_checker_loop, daemon=True, name="sla-checker")
    sla_thread.start()
    log.info("SLA checker background thread started")

    server = HTTPServer(("0.0.0.0", PORT), SupportDeskHandler)
    log.info("FractalMesh Customer Support Desk listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down Support Desk")
        server.server_close()


if __name__ == "__main__":
    _run()
