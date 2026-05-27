#!/usr/bin/env python3
"""
fm_localization.py — Multi-Language & Localisation Service (Port 7904)
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

PORT         = int(os.getenv("LOCALIZATION_PORT", "7904"))
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

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
        CREATE TABLE IF NOT EXISTS locales (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            locale_code  TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            native_name  TEXT NOT NULL,
            direction    TEXT NOT NULL DEFAULT 'ltr',
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS translation_keys (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            key_id       TEXT UNIQUE NOT NULL,
            namespace    TEXT NOT NULL DEFAULT 'common',
            key_path     TEXT NOT NULL,
            description  TEXT,
            default_text TEXT NOT NULL,
            UNIQUE(namespace, key_path)
        );
        CREATE TABLE IF NOT EXISTS translations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            key_id       TEXT NOT NULL,
            locale_code  TEXT NOT NULL,
            value        TEXT NOT NULL,
            translated_by TEXT DEFAULT 'manual',
            approved     INTEGER NOT NULL DEFAULT 0,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL,
            UNIQUE(key_id, locale_code)
        );
        CREATE TABLE IF NOT EXISTS translation_jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id       TEXT UNIQUE NOT NULL,
            namespace    TEXT NOT NULL,
            source_locale TEXT NOT NULL DEFAULT 'en',
            target_locale TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            total_keys   INTEGER NOT NULL DEFAULT 0,
            done_keys    INTEGER NOT NULL DEFAULT 0,
            created_at   REAL NOT NULL,
            completed_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_trans_key_locale ON translations(key_id, locale_code);
        CREATE INDEX IF NOT EXISTS idx_trans_keys_ns    ON translation_keys(namespace);
    """)
    con.commit()
    _seed_locales(con)
    _seed_translations(con)
    con.close()

def _seed_locales(con):
    if con.execute("SELECT COUNT(*) FROM locales").fetchone()[0] > 0:
        return
    now = time.time()
    locales = [
        ("en",    "English",            "English",       "ltr"),
        ("es",    "Spanish",            "Español",       "ltr"),
        ("fr",    "French",             "Français",      "ltr"),
        ("de",    "German",             "Deutsch",       "ltr"),
        ("zh",    "Chinese Simplified", "简体中文",       "ltr"),
        ("ja",    "Japanese",           "日本語",         "ltr"),
        ("ar",    "Arabic",             "العربية",        "rtl"),
        ("pt",    "Portuguese",         "Português",     "ltr"),
        ("it",    "Italian",            "Italiano",      "ltr"),
        ("ko",    "Korean",             "한국어",         "ltr"),
        ("hi",    "Hindi",              "हिन्दी",        "ltr"),
        ("id",    "Indonesian",         "Bahasa Indonesia", "ltr"),
    ]
    for code, name, native, direction in locales:
        con.execute(
            "INSERT INTO locales(locale_code,name,native_name,direction,created_at) VALUES(?,?,?,?,?)",
            (code, name, native, direction, now)
        )
    con.commit()

def _seed_translations(con):
    if con.execute("SELECT COUNT(*) FROM translation_keys").fetchone()[0] > 0:
        return
    now = time.time()
    keys = [
        ("common", "nav.home",         "Home",         "Navigation home link"),
        ("common", "nav.about",        "About",        "Navigation about link"),
        ("common", "nav.pricing",      "Pricing",      "Navigation pricing link"),
        ("common", "nav.contact",      "Contact",      "Navigation contact link"),
        ("common", "btn.submit",       "Submit",       "Submit button"),
        ("common", "btn.cancel",       "Cancel",       "Cancel button"),
        ("common", "btn.save",         "Save",         "Save button"),
        ("common", "btn.delete",       "Delete",       "Delete button"),
        ("common", "msg.success",      "Success!",     "Success message"),
        ("common", "msg.error",        "An error occurred. Please try again.", "Generic error"),
        ("auth",   "login.title",      "Sign In",      "Login page title"),
        ("auth",   "login.email",      "Email Address","Login email field"),
        ("auth",   "login.password",   "Password",     "Login password field"),
        ("auth",   "login.submit",     "Sign In",      "Login submit button"),
        ("auth",   "register.title",   "Create Account","Register page title"),
    ]
    for ns, path, default, desc in keys:
        kid = "key_" + secrets.token_hex(8)
        try:
            con.execute(
                "INSERT INTO translation_keys(key_id,namespace,key_path,description,default_text) VALUES(?,?,?,?,?)",
                (kid, ns, path, desc, default)
            )
        except sqlite3.IntegrityError:
            pass
    con.commit()

def _translate_with_claude(text, target_locale, context=""):
    if not ANTHROPIC_KEY:
        return None
    locale_names = {
        "es": "Spanish", "fr": "French", "de": "German", "zh": "Chinese (Simplified)",
        "ja": "Japanese", "ar": "Arabic", "pt": "Portuguese", "it": "Italian",
        "ko": "Korean", "hi": "Hindi", "id": "Indonesian Bahasa",
    }
    lang = locale_names.get(target_locale, target_locale)
    prompt = (
        f"Translate the following UI text to {lang}. "
        f"Return ONLY the translation, no explanations.\n"
        + (f"Context: {context}\n" if context else "")
        + f"Text: {text}"
    )
    payload = json.dumps({
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload)
    req.add_header("x-api-key", ANTHROPIC_KEY)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        return data["content"][0]["text"].strip()
    except Exception:
        return None

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

class L10nHandler(BaseHTTPRequestHandler):
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
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret,Accept-Language")
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

    def do_PUT(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._put(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p == ["health"]:
                return _j({"status": "ok", "port": PORT, "agent": "fm_localization"})

            if p == ["locales"]:
                rows = con.execute("SELECT * FROM locales WHERE active=1 ORDER BY name").fetchall()
                return _j([dict(r) for r in rows])

            # GET /t/{locale}/{namespace} — get all translations for locale+namespace as flat dict
            if len(p) == 3 and p[0] == "t":
                locale = p[1]
                ns = p[2]
                keys = con.execute(
                    "SELECT k.key_path, k.default_text, t.value, t.approved "
                    "FROM translation_keys k "
                    "LEFT JOIN translations t ON k.key_id=t.key_id AND t.locale_code=? "
                    "WHERE k.namespace=?",
                    (locale, ns)
                ).fetchall()
                result = {}
                for row in keys:
                    result[row["key_path"]] = row["value"] if row["value"] else row["default_text"]
                return _j(result)

            # GET /t/{locale} — get all translations for locale as {namespace: {key: value}}
            if len(p) == 2 and p[0] == "t":
                locale = p[1]
                keys = con.execute(
                    "SELECT k.namespace, k.key_path, k.default_text, t.value "
                    "FROM translation_keys k "
                    "LEFT JOIN translations t ON k.key_id=t.key_id AND t.locale_code=?",
                    (locale,)
                ).fetchall()
                result = {}
                for row in keys:
                    ns = row["namespace"]
                    if ns not in result:
                        result[ns] = {}
                    result[ns][row["key_path"]] = row["value"] if row["value"] else row["default_text"]
                return _j(result)

            if p == ["keys"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                ns = qs.get("namespace", [None])[0]
                if ns:
                    rows = con.execute("SELECT * FROM translation_keys WHERE namespace=?", (ns,)).fetchall()
                else:
                    rows = con.execute("SELECT * FROM translation_keys ORDER BY namespace, key_path").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["namespaces"]:
                rows = con.execute("SELECT DISTINCT namespace FROM translation_keys ORDER BY namespace").fetchall()
                return _j([r["namespace"] for r in rows])

            if p == ["coverage"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                total_keys = con.execute("SELECT COUNT(*) FROM translation_keys").fetchone()[0]
                locales = con.execute("SELECT locale_code FROM locales WHERE active=1 AND locale_code != 'en'").fetchall()
                coverage = []
                for loc in locales:
                    translated = con.execute(
                        "SELECT COUNT(*) FROM translations WHERE locale_code=? AND approved=1",
                        (loc["locale_code"],)
                    ).fetchone()[0]
                    pct = round((translated / total_keys * 100) if total_keys > 0 else 0, 1)
                    coverage.append({"locale": loc["locale_code"], "translated": translated,
                                     "total": total_keys, "pct": pct})
                return _j(coverage)

            if p == ["jobs"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute("SELECT * FROM translation_jobs ORDER BY created_at DESC LIMIT 20").fetchall()
                return _j([dict(r) for r in rows])

            return _err("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["keys"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                ns = data.get("namespace", "common")
                path = data.get("key_path", "")
                if not path:
                    return _err("key_path required")
                kid = "key_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO translation_keys(key_id,namespace,key_path,description,default_text) VALUES(?,?,?,?,?)",
                    (kid, ns, path, data.get("description"), data.get("default_text",""))
                )
                con.commit()
                return _j({"key_id": kid}, 201)

            if p == ["translate"]:
                # auto-translate a key to a target locale
                key_path = data.get("key_path","")
                ns = data.get("namespace","common")
                target_locale = data.get("target_locale","")
                if not all([key_path, target_locale]):
                    return _err("key_path and target_locale required")
                row = con.execute(
                    "SELECT * FROM translation_keys WHERE namespace=? AND key_path=?", (ns, key_path)
                ).fetchone()
                if not row:
                    return _err("Key not found", 404)
                translated = _translate_with_claude(row["default_text"], target_locale, data.get("context",""))
                if not translated:
                    return _err("Translation failed (check ANTHROPIC_API_KEY)")
                con.execute(
                    "INSERT INTO translations(key_id,locale_code,value,translated_by,approved,created_at,updated_at) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(key_id,locale_code) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (row["key_id"], target_locale, translated, "claude-haiku", 0, now, now)
                )
                con.commit()
                return _j({"key_path": key_path, "locale": target_locale, "value": translated})

            if p == ["jobs"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                ns = data.get("namespace","common")
                target = data.get("target_locale","")
                if not target:
                    return _err("target_locale required")
                total_keys = con.execute(
                    "SELECT COUNT(*) FROM translation_keys WHERE namespace=?", (ns,)
                ).fetchone()[0]
                jid = "job_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO translation_jobs(job_id,namespace,target_locale,status,total_keys,created_at) VALUES(?,?,?,?,?,?)",
                    (jid, ns, target, "running", total_keys, now)
                )
                con.commit()
                threading.Thread(target=self._run_job, args=(jid, ns, target, now), daemon=True).start()
                return _j({"job_id": jid, "total_keys": total_keys}, 201)

            if p == ["bulk"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                locale = data.get("locale_code","")
                translations = data.get("translations", {})
                upserted = 0
                for path, value in translations.items():
                    parts = path.split(":", 1)
                    ns = parts[0] if len(parts) == 2 else "common"
                    key_path = parts[1] if len(parts) == 2 else path
                    row = con.execute(
                        "SELECT key_id FROM translation_keys WHERE namespace=? AND key_path=?", (ns, key_path)
                    ).fetchone()
                    if row:
                        con.execute(
                            "INSERT INTO translations(key_id,locale_code,value,translated_by,approved,created_at,updated_at) VALUES(?,?,?,?,1,?,?) "
                            "ON CONFLICT(key_id,locale_code) DO UPDATE SET value=excluded.value, approved=1, updated_at=excluded.updated_at",
                            (row["key_id"], locale, value, "import", now, now)
                        )
                        upserted += 1
                con.commit()
                return _j({"upserted": upserted})

            return _err("Not found", 404)
        finally:
            con.close()

    def _put(self, p, data):
        con = _db()
        now = time.time()
        try:
            # PUT /translations/{key_id}/{locale} — update a specific translation
            if len(p) == 3 and p[0] == "translations":
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                con.execute(
                    "INSERT INTO translations(key_id,locale_code,value,translated_by,approved,created_at,updated_at) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(key_id,locale_code) DO UPDATE SET value=excluded.value, approved=excluded.approved, updated_at=excluded.updated_at",
                    (p[1], p[2], data.get("value",""), data.get("translated_by","manual"),
                     int(data.get("approved", True)), now, now)
                )
                con.commit()
                return _j({"key_id": p[1], "locale": p[2], "updated": True})
            return _err("Not found", 404)
        finally:
            con.close()

    def _run_job(self, job_id, namespace, target_locale, started_at):
        con = _db()
        try:
            keys = con.execute(
                "SELECT * FROM translation_keys WHERE namespace=?", (namespace,)
            ).fetchall()
            done = 0
            for key in keys:
                # skip if already translated
                existing = con.execute(
                    "SELECT value FROM translations WHERE key_id=? AND locale_code=?",
                    (key["key_id"], target_locale)
                ).fetchone()
                if existing:
                    done += 1
                    continue
                translated = _translate_with_claude(key["default_text"], target_locale)
                if translated:
                    now = time.time()
                    con.execute(
                        "INSERT INTO translations(key_id,locale_code,value,translated_by,approved,created_at,updated_at) VALUES(?,?,?,?,0,?,?) "
                        "ON CONFLICT(key_id,locale_code) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (key["key_id"], target_locale, translated, "claude-haiku", now, now)
                    )
                    con.commit()
                done += 1
                con.execute("UPDATE translation_jobs SET done_keys=? WHERE job_id=?", (done, job_id))
                con.commit()
            con.execute(
                "UPDATE translation_jobs SET status='completed', done_keys=?, completed_at=? WHERE job_id=?",
                (done, time.time(), job_id)
            )
            con.commit()
        except Exception as e:
            con.execute(
                "UPDATE translation_jobs SET status='failed' WHERE job_id=?", (job_id,)
            )
            con.commit()
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), L10nHandler)
    print(f"[fm_localization] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
