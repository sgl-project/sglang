#!/usr/bin/env python3
"""
fm_contract_manager.py — Contract & Legal Document Manager (Port 7896)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import os
import json
import sqlite3
import time
import hashlib
import hmac
import secrets
import base64
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT            = int(os.getenv("CONTRACT_MANAGER_PORT", "7896"))
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
SG_KEY          = os.getenv("SENDGRID_API_KEY", "")
SG_FROM         = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.ai")
ADMIN_SECRET    = os.getenv("ADMIN_SECRET", "")
BUSINESS_NAME   = os.getenv("BUSINESS_NAME", "IronVision Nexus")
BUSINESS_ABN    = os.getenv("BUSINESS_ABN", "56 628 117 363")

ROOT   = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB     = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

# ── database ──────────────────────────────────────────────────────────────────
def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS contract_templates (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            template_id  TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            category     TEXT NOT NULL DEFAULT 'general',
            content      TEXT NOT NULL,
            variables    TEXT NOT NULL DEFAULT '[]',
            version      TEXT NOT NULL DEFAULT '1.0',
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS contracts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id      TEXT UNIQUE NOT NULL,
            title            TEXT NOT NULL,
            template_id      TEXT,
            content          TEXT NOT NULL,
            status           TEXT NOT NULL DEFAULT 'draft',
            party_a_name     TEXT NOT NULL DEFAULT '',
            party_a_email    TEXT NOT NULL DEFAULT '',
            party_b_name     TEXT NOT NULL DEFAULT '',
            party_b_email    TEXT NOT NULL DEFAULT '',
            value            REAL NOT NULL DEFAULT 0,
            currency         TEXT NOT NULL DEFAULT 'AUD',
            effective_date   REAL,
            expiry_date      REAL,
            signed_by_a      INTEGER NOT NULL DEFAULT 0,
            signed_at_a      REAL,
            signed_by_b      INTEGER NOT NULL DEFAULT 0,
            signed_at_b      REAL,
            signature_hash   TEXT,
            metadata         TEXT NOT NULL DEFAULT '{}',
            created_at       REAL NOT NULL,
            updated_at       REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS contract_signatures (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sig_id          TEXT UNIQUE NOT NULL,
            contract_id     TEXT NOT NULL,
            signer_email    TEXT NOT NULL,
            signer_name     TEXT NOT NULL DEFAULT '',
            signer_role     TEXT NOT NULL DEFAULT 'party',
            ip_hash         TEXT,
            signature_token TEXT UNIQUE NOT NULL,
            signed_at       REAL,
            content_hash    TEXT
        );
        CREATE TABLE IF NOT EXISTS contract_amendments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            amendment_id TEXT UNIQUE NOT NULL,
            contract_id  TEXT NOT NULL,
            description  TEXT NOT NULL,
            old_content  TEXT NOT NULL,
            new_content  TEXT NOT NULL,
            amended_by   TEXT NOT NULL DEFAULT '',
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_contracts_status   ON contracts(status);
        CREATE INDEX IF NOT EXISTS idx_contracts_expiry   ON contracts(expiry_date);
        CREATE INDEX IF NOT EXISTS idx_sigs_token         ON contract_signatures(signature_token);
        CREATE INDEX IF NOT EXISTS idx_sigs_contract      ON contract_signatures(contract_id);
    """)
    con.commit()
    _seed_templates(con)
    con.close()

_SERVICE_TEMPLATE = """SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of {{start_date}} between:

**Service Provider**: {{party_a_name}} (Party A)
**Client**: {{party_b_name}} (Party B)

1. SERVICES
Party A agrees to provide the following services to Party B:
{{service_description}}

2. PAYMENT
Party B agrees to pay Party A {{payment_amount}} AUD according to the following schedule:
{{payment_schedule}}

3. TERM
This Agreement commences on {{start_date}} and continues for {{duration}}.

4. INTELLECTUAL PROPERTY
All work product created under this Agreement shall be owned by Party B upon full payment.

5. CONFIDENTIALITY
Both parties agree to maintain the confidentiality of each other's proprietary information.

6. TERMINATION
Either party may terminate this Agreement with 30 days written notice.

7. GOVERNING LAW
This Agreement is governed by the laws of New South Wales, Australia.

Prepared by: {business_name} | ABN: {business_abn}
""".format(business_name=BUSINESS_NAME, business_abn=BUSINESS_ABN)

_NDA_TEMPLATE = """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("NDA") is entered into between:

**Disclosing Party**: {{disclosing_party}}
**Receiving Party**: {{receiving_party}}

1. PURPOSE
The parties wish to explore a potential business relationship related to: {{purpose}}

2. CONFIDENTIAL INFORMATION
"Confidential Information" means any non-public information disclosed by the Disclosing Party.

3. OBLIGATIONS
The Receiving Party agrees to: (a) hold Confidential Information in strict confidence;
(b) not disclose to third parties without prior written consent; (c) use only for the Purpose.

4. TERM
This NDA is effective from the date of signing and remains in effect for {{duration}}.

5. EXCEPTIONS
Obligations do not apply to information that is publicly known, independently developed,
or required to be disclosed by law.

6. GOVERNING LAW
This Agreement is governed by the laws of New South Wales, Australia.

Prepared by: {business_name} | ABN: {business_abn}
""".format(business_name=BUSINESS_NAME, business_abn=BUSINESS_ABN)

_LICENSE_TEMPLATE = """SOFTWARE LICENSE AGREEMENT

This Software License Agreement is between:

**Licensor**: {{party_a_name}}
**Licensee**: {{party_b_name}}

1. GRANT OF LICENSE
Licensor grants Licensee a {{license_type}} license to use {{software_name}} in {{territory}}.

2. LICENSE FEE
Licensee agrees to pay a license fee of {{fee}} AUD.

3. RESTRICTIONS
Licensee may not: (a) sublicense or resell; (b) reverse engineer; (c) remove copyright notices.

4. INTELLECTUAL PROPERTY
All intellectual property rights in {{software_name}} remain with the Licensor.

5. WARRANTY DISCLAIMER
The software is provided "AS IS" without warranty of any kind.

6. LIMITATION OF LIABILITY
Licensor's liability is limited to the amount paid under this Agreement.

7. GOVERNING LAW
This Agreement is governed by the laws of New South Wales, Australia.

Prepared by: {business_name} | ABN: {business_abn}
""".format(business_name=BUSINESS_NAME, business_abn=BUSINESS_ABN)

def _seed_templates(con):
    if con.execute("SELECT COUNT(*) FROM contract_templates").fetchone()[0] > 0:
        return
    now = time.time()
    templates = [
        ("tmpl_service", "Service Agreement", "service", _SERVICE_TEMPLATE,
         json.dumps(["service_description","payment_amount","payment_schedule","start_date","duration"])),
        ("tmpl_nda", "Non-Disclosure Agreement", "legal", _NDA_TEMPLATE,
         json.dumps(["disclosing_party","receiving_party","purpose","duration"])),
        ("tmpl_license", "Software License", "license", _LICENSE_TEMPLATE,
         json.dumps(["software_name","license_type","fee","territory"])),
    ]
    for tid, name, cat, content, variables in templates:
        con.execute(
            "INSERT INTO contract_templates(template_id,name,category,content,variables,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (tid, name, cat, content, variables, now, now)
        )
    con.commit()

# ── helpers ───────────────────────────────────────────────────────────────────
def _j(data, status=200):
    body = json.dumps(data, default=str).encode()
    return status, body

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _verify_admin(headers):
    h = headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(h, ADMIN_SECRET)

def _render_template(content, variables):
    for k, v in variables.items():
        content = content.replace("{{" + k + "}}", str(v))
    return content

def _hash_ip(ip):
    return hashlib.sha256((ip or "").encode()).hexdigest()[:16]

def _content_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()

def _send_email(to_email, subject, html_body):
    if not SG_KEY:
        return
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SG_FROM, "name": BUSINESS_NAME},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }).encode()
    req = urllib.request.Request("https://api.sendgrid.com/v3/mail/send", data=payload)
    req.add_header("Authorization", f"Bearer {SG_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

def _call_claude(prompt):
    if not ANTHROPIC_KEY:
        return {"error": "ANTHROPIC_API_KEY not configured"}
    payload = json.dumps({
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload)
    req.add_header("x-api-key", ANTHROPIC_KEY)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return {"analysis": data["content"][0]["text"]}
    except Exception as e:
        return {"error": str(e)}

# ── background daemon ─────────────────────────────────────────────────────────
def _expiry_daemon():
    while True:
        time.sleep(3600)
        try:
            con = _db()
            now = time.time()
            # expire contracts past their expiry_date
            rows = con.execute(
                "SELECT contract_id, party_a_email, party_b_email, title FROM contracts "
                "WHERE expiry_date IS NOT NULL AND expiry_date < ? AND status IN ('signed','active')",
                (now,)
            ).fetchall()
            for row in rows:
                con.execute(
                    "UPDATE contracts SET status='expired', updated_at=? WHERE contract_id=?",
                    (now, row["contract_id"])
                )
                for email in [row["party_a_email"], row["party_b_email"]]:
                    if email:
                        threading.Thread(
                            target=_send_email,
                            args=(email, f"Contract Expired: {row['title']}",
                                  f"<p>Your contract <strong>{row['title']}</strong> has expired.</p>"),
                            daemon=True
                        ).start()
            # warn about contracts expiring in 7 days
            warn_cutoff = now + 7 * 86400
            warn_rows = con.execute(
                "SELECT contract_id, party_a_email, party_b_email, title, expiry_date FROM contracts "
                "WHERE expiry_date IS NOT NULL AND expiry_date BETWEEN ? AND ? AND status IN ('signed','active')",
                (now, warn_cutoff)
            ).fetchall()
            for row in warn_rows:
                days_left = int((row["expiry_date"] - now) / 86400)
                for email in [row["party_a_email"], row["party_b_email"]]:
                    if email:
                        threading.Thread(
                            target=_send_email,
                            args=(email, f"Contract Expiring Soon: {row['title']}",
                                  f"<p>Your contract <strong>{row['title']}</strong> expires in {days_left} days.</p>"),
                            daemon=True
                        ).start()
            con.commit()
            con.close()
        except Exception:
            pass

threading.Thread(target=_expiry_daemon, daemon=True).start()

# ── HTTP handler ──────────────────────────────────────────────────────────────
class ContractHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
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
        path = parsed.path.rstrip("/")
        parts = path.strip("/").split("/")
        qs = parse_qs(parsed.query)

        try:
            code, body = self._route_get(parts, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.strip("/").split("/")

        try:
            data = self._read_body()
            code, body = self._route_post(parts, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    # ── GET routing ───────────────────────────────────────────────────────────
    def _route_get(self, parts, qs):
        con = _db()
        try:
            if parts == ["health"]:
                return _j({"status": "ok", "port": PORT, "agent": "fm_contract_manager"})

            if parts == ["templates"]:
                rows = con.execute(
                    "SELECT template_id,name,category,variables,version,created_at FROM contract_templates WHERE active=1"
                ).fetchall()
                return _j([dict(r) for r in rows])

            if len(parts) == 2 and parts[0] == "templates":
                row = con.execute(
                    "SELECT * FROM contract_templates WHERE template_id=? AND active=1", (parts[1],)
                ).fetchone()
                if not row:
                    return _err("Template not found", 404)
                return _j(dict(row))

            if parts == ["contracts"]:
                if not _verify_admin(self.headers):
                    return _err("Unauthorized", 403)
                status_filter = qs.get("status", [None])[0]
                if status_filter:
                    rows = con.execute(
                        "SELECT * FROM contracts WHERE status=? ORDER BY created_at DESC", (status_filter,)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM contracts ORDER BY created_at DESC"
                    ).fetchall()
                return _j([dict(r) for r in rows])

            if len(parts) == 2 and parts[0] == "contracts":
                row = con.execute(
                    "SELECT * FROM contracts WHERE contract_id=?", (parts[1],)
                ).fetchone()
                if not row:
                    return _err("Contract not found", 404)
                contract = dict(row)
                sigs = con.execute(
                    "SELECT sig_id,signer_email,signer_name,signer_role,signed_at FROM contract_signatures WHERE contract_id=?",
                    (parts[1],)
                ).fetchall()
                contract["signatures"] = [dict(s) for s in sigs]
                return _j(contract)

            if len(parts) == 2 and parts[0] == "sign":
                row = con.execute(
                    "SELECT cs.*, c.title, c.content, c.party_a_name, c.party_b_name "
                    "FROM contract_signatures cs JOIN contracts c ON cs.contract_id=c.contract_id "
                    "WHERE cs.signature_token=?", (parts[1],)
                ).fetchone()
                if not row:
                    return _err("Invalid token", 404)
                if row["signed_at"]:
                    return _err("Already signed", 400)
                return _j({
                    "title": row["title"],
                    "signer_name": row["signer_name"],
                    "signer_email": row["signer_email"],
                    "signer_role": row["signer_role"],
                    "party_a": row["party_a_name"],
                    "party_b": row["party_b_name"],
                    "content_preview": row["content"][:500] + "...",
                })

            if parts == ["expiring"]:
                if not _verify_admin(self.headers):
                    return _err("Unauthorized", 403)
                cutoff = time.time() + 30 * 86400
                rows = con.execute(
                    "SELECT contract_id,title,party_a_name,party_b_name,expiry_date,status,value FROM contracts "
                    "WHERE expiry_date IS NOT NULL AND expiry_date <= ? AND status IN ('signed','active') ORDER BY expiry_date",
                    (cutoff,)
                ).fetchall()
                return _j([dict(r) for r in rows])

            if parts == ["dashboard"]:
                if not _verify_admin(self.headers):
                    return _err("Unauthorized", 403)
                now = time.time()
                stats = con.execute(
                    "SELECT status, COUNT(*) as cnt, COALESCE(SUM(value),0) as total_value "
                    "FROM contracts GROUP BY status"
                ).fetchall()
                counts = {r["status"]: r["cnt"] for r in stats}
                values = {r["status"]: r["total_value"] for r in stats}
                expiring_soon = con.execute(
                    "SELECT COUNT(*) FROM contracts WHERE expiry_date IS NOT NULL AND expiry_date <= ? AND status IN ('signed','active')",
                    (now + 30 * 86400,)
                ).fetchone()[0]
                total_value = con.execute("SELECT COALESCE(SUM(value),0) FROM contracts WHERE status='signed'").fetchone()[0]
                return _j({
                    "counts": counts,
                    "total_signed_value_aud": round(total_value, 2),
                    "expiring_in_30_days": expiring_soon,
                })

            return _err("Not found", 404)
        finally:
            con.close()

    # ── POST routing ──────────────────────────────────────────────────────────
    def _route_post(self, parts, data):
        con = _db()
        now = time.time()
        try:
            if parts == ["templates"] or (len(parts) == 1 and parts[0] == "templates"):
                if not _verify_admin(self.headers):
                    return _err("Unauthorized", 403)
                tid = "tmpl_" + secrets.token_hex(8)
                name = data.get("name", "")
                if not name:
                    return _err("name required")
                con.execute(
                    "INSERT INTO contract_templates(template_id,name,category,content,variables,version,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (tid, name, data.get("category","general"), data.get("content",""),
                     json.dumps(data.get("variables",[])), data.get("version","1.0"), now, now)
                )
                con.commit()
                return _j({"template_id": tid, "name": name}, 201)

            if parts == ["contracts"]:
                title = data.get("title","")
                party_a_email = data.get("party_a_email","")
                party_b_email = data.get("party_b_email","")
                if not all([title, party_a_email, party_b_email]):
                    return _err("title, party_a_email, party_b_email required")
                template_id = data.get("template_id")
                if template_id:
                    tmpl = con.execute(
                        "SELECT content FROM contract_templates WHERE template_id=?", (template_id,)
                    ).fetchone()
                    content = _render_template(tmpl["content"] if tmpl else "", data.get("variables", {}))
                else:
                    content = data.get("content", "")
                # Also render party names into content
                content = _render_template(content, {
                    "party_a_name": data.get("party_a_name", ""),
                    "party_b_name": data.get("party_b_name", ""),
                })
                cid = "ctr_" + secrets.token_hex(10)
                expiry = None
                if data.get("expiry_date_days"):
                    expiry = now + int(data["expiry_date_days"]) * 86400
                elif data.get("expiry_date"):
                    expiry = float(data["expiry_date"])
                con.execute(
                    "INSERT INTO contracts(contract_id,title,template_id,content,party_a_name,party_a_email,"
                    "party_b_name,party_b_email,value,currency,effective_date,expiry_date,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, title, template_id, content,
                     data.get("party_a_name",""), party_a_email,
                     data.get("party_b_name",""), party_b_email,
                     data.get("value", 0), data.get("currency","AUD"),
                     data.get("effective_date"), expiry, now, now)
                )
                con.commit()
                return _j({"contract_id": cid, "title": title, "status": "draft"}, 201)

            if len(parts) == 3 and parts[0] == "contracts" and parts[2] == "send":
                cid = parts[1]
                row = con.execute("SELECT * FROM contracts WHERE contract_id=?", (cid,)).fetchone()
                if not row:
                    return _err("Contract not found", 404)
                contract = dict(row)
                parties = [
                    (contract["party_a_email"], contract["party_a_name"], "party_a"),
                    (contract["party_b_email"], contract["party_b_name"], "party_b"),
                ]
                tokens = []
                for email, name, role in parties:
                    existing = con.execute(
                        "SELECT signature_token FROM contract_signatures WHERE contract_id=? AND signer_email=?",
                        (cid, email)
                    ).fetchone()
                    if existing:
                        tok = existing["signature_token"]
                    else:
                        tok = secrets.token_urlsafe(32)
                        sig_id = "sig_" + secrets.token_hex(8)
                        con.execute(
                            "INSERT INTO contract_signatures(sig_id,contract_id,signer_email,signer_name,signer_role,signature_token) "
                            "VALUES(?,?,?,?,?,?)",
                            (sig_id, cid, email, name, role, tok)
                        )
                    tokens.append({"email": email, "role": role, "token": tok})
                    threading.Thread(
                        target=_send_email,
                        args=(email,
                              f"Action Required: Sign '{contract['title']}'",
                              f"<p>Hi {name},</p><p>Please review and sign the contract: "
                              f"<strong>{contract['title']}</strong></p>"
                              f"<p>Your signature token: <code>{tok}</code></p>"
                              f"<p>Use POST /sign/{tok} with {{\"agreed\": true}} to sign.</p>"),
                        daemon=True
                    ).start()
                con.execute("UPDATE contracts SET status='sent', updated_at=? WHERE contract_id=?", (now, cid))
                con.commit()
                return _j({"contract_id": cid, "sent_to": tokens})

            if len(parts) == 2 and parts[0] == "sign":
                tok = parts[1]
                if not data.get("agreed"):
                    return _err("agreed: true required")
                row = con.execute(
                    "SELECT cs.*, c.content, c.contract_id, c.party_a_email, c.party_b_email, c.title "
                    "FROM contract_signatures cs JOIN contracts c ON cs.contract_id=c.contract_id "
                    "WHERE cs.signature_token=?", (tok,)
                ).fetchone()
                if not row:
                    return _err("Invalid token", 404)
                if row["signed_at"]:
                    return _err("Already signed", 400)
                ip = self.client_address[0]
                ch = _content_hash(row["content"])
                con.execute(
                    "UPDATE contract_signatures SET signed_at=?, signer_name=?, content_hash=?, ip_hash=? "
                    "WHERE signature_token=?",
                    (now, data.get("signer_name", row["signer_name"]), ch, _hash_ip(ip), tok)
                )
                # update contract signed_by fields
                role = row["signer_role"]
                if role == "party_a":
                    con.execute("UPDATE contracts SET signed_by_a=1, signed_at_a=?, updated_at=? WHERE contract_id=?",
                                (now, now, row["contract_id"]))
                else:
                    con.execute("UPDATE contracts SET signed_by_b=1, signed_at_b=?, updated_at=? WHERE contract_id=?",
                                (now, now, row["contract_id"]))
                con.commit()
                # check if fully signed
                c = con.execute("SELECT * FROM contracts WHERE contract_id=?", (row["contract_id"],)).fetchone()
                if c["signed_by_a"] and c["signed_by_b"]:
                    sigs = con.execute(
                        "SELECT content_hash FROM contract_signatures WHERE contract_id=? ORDER BY signed_at",
                        (row["contract_id"],)
                    ).fetchall()
                    combined = "".join(s["content_hash"] for s in sigs if s["content_hash"])
                    sig_hash = _content_hash(combined)
                    con.execute(
                        "UPDATE contracts SET status='signed', signature_hash=?, updated_at=? WHERE contract_id=?",
                        (sig_hash, now, row["contract_id"])
                    )
                    con.commit()
                    for email in [c["party_a_email"], c["party_b_email"]]:
                        threading.Thread(
                            target=_send_email,
                            args=(email, f"Contract Fully Signed: {c['title']}",
                                  f"<p>Contract <strong>{c['title']}</strong> has been signed by all parties.</p>"
                                  f"<p>Signature hash: <code>{sig_hash[:16]}...</code></p>"),
                            daemon=True
                        ).start()
                    return _j({"status": "signed", "signature_hash": sig_hash})
                con.commit()
                threading.Thread(
                    target=_send_email,
                    args=(row["signer_email"], f"Signature Confirmed: {row['title']}",
                          f"<p>Your signature has been recorded for <strong>{row['title']}</strong>.</p>"
                          f"<p>Content hash: <code>{ch[:16]}...</code></p>"),
                    daemon=True
                ).start()
                return _j({"status": "signed_partial", "message": "Awaiting other party signature"})

            if len(parts) == 3 and parts[0] == "contracts" and parts[2] == "amend":
                cid = parts[1]
                row = con.execute("SELECT * FROM contracts WHERE contract_id=?", (cid,)).fetchone()
                if not row:
                    return _err("Contract not found", 404)
                description = data.get("description","")
                new_content = data.get("new_content","")
                if not all([description, new_content]):
                    return _err("description and new_content required")
                aid = "amd_" + secrets.token_hex(8)
                old_content = row["content"]
                con.execute(
                    "INSERT INTO contract_amendments(amendment_id,contract_id,description,old_content,new_content,amended_by,created_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (aid, cid, description, old_content, new_content, data.get("amended_by",""), now)
                )
                # reset signatures if was signed
                if row["status"] == "signed":
                    con.execute(
                        "UPDATE contracts SET content=?,status='amended',signed_by_a=0,signed_by_b=0,"
                        "signed_at_a=NULL,signed_at_b=NULL,signature_hash=NULL,updated_at=? WHERE contract_id=?",
                        (new_content, now, cid)
                    )
                else:
                    con.execute("UPDATE contracts SET content=?, updated_at=? WHERE contract_id=?",
                                (new_content, now, cid))
                con.commit()
                return _j({"amendment_id": aid, "contract_id": cid})

            if len(parts) == 3 and parts[0] == "contracts" and parts[2] == "terminate":
                if not _verify_admin(self.headers):
                    return _err("Unauthorized", 403)
                cid = parts[1]
                con.execute(
                    "UPDATE contracts SET status='terminated', updated_at=? WHERE contract_id=?",
                    (now, cid)
                )
                con.commit()
                return _j({"contract_id": cid, "status": "terminated"})

            if parts == ["ai", "analyse"]:
                content = data.get("content", "")
                cid = data.get("contract_id", "")
                if cid and not content:
                    row = con.execute("SELECT content FROM contracts WHERE contract_id=?", (cid,)).fetchone()
                    if row:
                        content = row["content"]
                if not content:
                    return _err("content or contract_id required")
                prompt = (
                    "Analyse this contract and return a JSON object with: "
                    "key_terms (list), obligations (list), risks (list), "
                    "payment_terms (string), expiry_date (string if found), "
                    "governing_law (string), recommendations (list).\n\n"
                    f"CONTRACT:\n{content[:4000]}"
                )
                result = _call_claude(prompt)
                return _j(result)

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), ContractHandler)
    print(f"[fm_contract_manager] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
