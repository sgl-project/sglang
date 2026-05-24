#!/usr/bin/env python3
"""
fm_sovereign_memory.py — Sovereign Memory Consolidator for FractalMesh OMEGA Titan
Central memory/knowledge store: facts, decisions, agent outputs, learned patterns, summaries.
Provides RAG-like retrieval via keyword search. Backs up to GitHub Gist periodically.
Port: 7853
"""
import os
import json
import gzip
import time
import sqlite3
import hashlib
import threading
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Vault / env bootstrap
# ---------------------------------------------------------------------------
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT          = int(os.getenv("SOVEREIGN_MEMORY_PORT", "7853"))
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
GITHUB_ORG    = os.getenv("GITHUB_ORG", "")
ADMIN_SECRET  = os.getenv("ADMIN_SECRET", "")

ROOT = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB_PATH = os.path.join(ROOT, "database", "sovereign_memory.db")

VALID_CATEGORIES = {
    "fact", "decision", "lead", "revenue", "agent_output",
    "pattern", "preference", "config", "error_log", "opportunity",
}
CONSOLIDATION_INTERVAL = 3600  # seconds
CONSOLIDATION_THRESHOLD = 20   # entries per category before summary

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY,
            category     TEXT    NOT NULL,
            key          TEXT    NOT NULL,
            value        TEXT    NOT NULL,
            source       TEXT,
            confidence   REAL    DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL,
            tags         TEXT,
            expires_at   REAL,
            created_at   REAL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_cat_key
            ON memories(category, key);
        CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category);

        CREATE TABLE IF NOT EXISTS memory_links (
            id           INTEGER PRIMARY KEY,
            memory_id_a  INTEGER NOT NULL,
            memory_id_b  INTEGER NOT NULL,
            relationship TEXT,
            strength     REAL DEFAULT 0.5,
            created_at   REAL
        );

        CREATE TABLE IF NOT EXISTS consolidations (
            id           INTEGER PRIMARY KEY,
            category     TEXT,
            summary      TEXT,
            memory_count INTEGER,
            created_at   REAL
        );

        CREATE TABLE IF NOT EXISTS backups (
            id               INTEGER PRIMARY KEY,
            destination      TEXT,
            records_backed_up INTEGER,
            gist_id          TEXT,
            status           TEXT,
            created_at       REAL
        );
    """)
    conn.commit()
    conn.close()


def _upsert_memory(conn, category, key, value_str, source, confidence, tags_str, expires_at):
    """Insert or update a memory row by (category, key). Returns memory_id."""
    now = time.time()
    cur = conn.execute(
        "SELECT id FROM memories WHERE category=? AND key=?", (category, key)
    )
    row = cur.fetchone()
    if row:
        conn.execute(
            """UPDATE memories
               SET value=?, source=?, confidence=?, tags=?, expires_at=?,
                   last_accessed=?
               WHERE id=?""",
            (value_str, source, confidence, tags_str, expires_at, now, row["id"]),
        )
        conn.commit()
        return row["id"]
    else:
        cur = conn.execute(
            """INSERT INTO memories
               (category, key, value, source, confidence, tags, expires_at, created_at, last_accessed)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (category, key, value_str, source, confidence, tags_str, expires_at, now, now),
        )
        conn.commit()
        return cur.lastrowid


def _consolidate_category(conn, category) -> str:
    """Summarise the top-20 most recently accessed memories for a category."""
    rows = conn.execute(
        """SELECT key, value FROM memories
           WHERE category=?
           ORDER BY last_accessed DESC NULLS LAST
           LIMIT 20""",
        (category,),
    ).fetchall()
    parts = []
    for r in rows:
        parts.append(f"{r['key']}: {r['value']}")
    return "; ".join(parts)


def _backup_to_gist(conn):
    """Back up all memories to a GitHub Gist. Returns (gist_id, record_count)."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set")

    rows = conn.execute(
        "SELECT category, key, value, source, confidence, tags, created_at FROM memories"
    ).fetchall()

    by_category = {}
    for r in rows:
        cat = r["category"]
        by_category.setdefault(cat, []).append({
            "key": r["key"],
            "value": r["value"],
            "source": r["source"],
            "confidence": r["confidence"],
            "tags": r["tags"],
            "created_at": r["created_at"],
        })

    content = json.dumps(by_category, indent=2)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "description": f"FractalMesh Sovereign Memory Backup {ts}",
        "public": False,
        "files": {
            "sovereign_memory.json": {"content": content}
        },
    }

    # Check if we have a recent gist_id to PATCH instead of POST
    last_backup = conn.execute(
        "SELECT gist_id FROM backups WHERE status='ok' AND gist_id IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()

    body_bytes = json.dumps(payload).encode()
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "FractalMesh-SovereignMemory/1.0",
    }

    if last_backup and last_backup["gist_id"]:
        gist_id = last_backup["gist_id"]
        url = f"https://api.github.com/gists/{gist_id}"
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="PATCH")
    else:
        url = "https://api.github.com/gists"
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())

    gist_id = result.get("id", "")
    return gist_id, len(rows)


def _run_consolidation():
    """Background consolidation: summarise, prune expired, backup."""
    conn = _get_conn()
    try:
        now = time.time()

        # Prune expired memories
        conn.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
        conn.commit()

        # Consolidate categories with > CONSOLIDATION_THRESHOLD entries
        categories = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        ).fetchall()

        for cat_row in categories:
            cat = cat_row["category"]
            cnt = cat_row["cnt"]
            if cnt > CONSOLIDATION_THRESHOLD:
                summary = _consolidate_category(conn, cat)
                conn.execute(
                    "INSERT INTO consolidations (category, summary, memory_count, created_at) VALUES (?,?,?,?)",
                    (cat, summary, cnt, now),
                )
                conn.commit()

        # Backup to GitHub Gist if token available
        if GITHUB_TOKEN:
            try:
                gist_id, record_count = _backup_to_gist(conn)
                conn.execute(
                    "INSERT INTO backups (destination, records_backed_up, gist_id, status, created_at) VALUES (?,?,?,?,?)",
                    ("github_gist", record_count, gist_id, "ok", now),
                )
                conn.commit()
            except Exception as e:
                conn.execute(
                    "INSERT INTO backups (destination, records_backed_up, gist_id, status, created_at) VALUES (?,?,?,?,?)",
                    ("github_gist", 0, None, f"error: {e}", now),
                )
                conn.commit()
    finally:
        conn.close()


def _consolidation_loop():
    """Daemon thread: run consolidation every CONSOLIDATION_INTERVAL seconds."""
    while True:
        time.sleep(CONSOLIDATION_INTERVAL)
        try:
            _run_consolidation()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _json_response(handler, code, data):
    body = json.dumps(data).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode())


def _require_admin(handler) -> bool:
    if not ADMIN_SECRET:
        return True
    secret = handler.headers.get("X-Admin-Secret", "")
    if secret != ADMIN_SECRET:
        _json_response(handler, 403, {"error": "forbidden"})
        return False
    return True


class SovereignMemoryHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        query = self.path.split("?")[1] if "?" in self.path else ""

        if path == "/health":
            _json_response(self, 200, {
                "status": "ok",
                "service": "fm-sovereign-memory",
                "port": PORT,
            })

        elif path == "/memories":
            self._handle_list_memories(query)

        elif path == "/consolidations":
            self._handle_list_consolidations()

        elif path == "/backups":
            self._handle_list_backups()

        elif path == "/stats":
            self._handle_stats()

        elif path == "/analytics":
            self._handle_analytics()

        else:
            # /memories/{id} or /memories/{id}/links
            parts = [p for p in path.split("/") if p]
            if len(parts) == 2 and parts[0] == "memories":
                try:
                    mem_id = int(parts[1])
                except ValueError:
                    _json_response(self, 400, {"error": "invalid id"})
                    return
                self._handle_get_memory(mem_id)

            elif len(parts) == 3 and parts[0] == "memories" and parts[2] == "links":
                try:
                    mem_id = int(parts[1])
                except ValueError:
                    _json_response(self, 400, {"error": "invalid id"})
                    return
                self._handle_get_links(mem_id)

            else:
                _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/remember":
            self._handle_remember()

        elif path == "/recall":
            self._handle_recall()

        elif path == "/link":
            self._handle_link()

        elif path == "/consolidate":
            self._handle_consolidate()

        elif path == "/backup":
            self._handle_backup()

        elif path == "/import/bulk":
            self._handle_bulk_import()

        else:
            _json_response(self, 404, {"error": "not found"})

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) == 2 and parts[0] == "memories":
            try:
                mem_id = int(parts[1])
            except ValueError:
                _json_response(self, 400, {"error": "invalid id"})
                return
            self._handle_update_memory(mem_id)
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) == 2 and parts[0] == "memories":
            try:
                mem_id = int(parts[1])
            except ValueError:
                _json_response(self, 400, {"error": "invalid id"})
                return
            self._handle_delete_memory(mem_id)
        else:
            _json_response(self, 404, {"error": "not found"})

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_remember(self):
        try:
            body = _read_body(self)
        except Exception:
            _json_response(self, 400, {"error": "invalid json"})
            return

        category = body.get("category", "fact")
        if category not in VALID_CATEGORIES:
            _json_response(self, 400, {"error": f"invalid category; valid: {sorted(VALID_CATEGORIES)}"})
            return

        key = body.get("key")
        value = body.get("value")
        if not key or value is None:
            _json_response(self, 400, {"error": "key and value required"})
            return

        value_str = json.dumps(value) if not isinstance(value, str) else value
        source = body.get("source", "")
        confidence = float(body.get("confidence", 1.0))
        tags = body.get("tags", [])
        tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)
        expires_in_days = body.get("expires_in_days")
        expires_at = None
        if expires_in_days is not None:
            expires_at = time.time() + float(expires_in_days) * 86400

        conn = _get_conn()
        try:
            mem_id = _upsert_memory(conn, category, key, value_str, source, confidence, tags_str, expires_at)
            _json_response(self, 200, {"memory_id": mem_id, "category": category, "key": key})
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_recall(self):
        try:
            body = _read_body(self)
        except Exception:
            _json_response(self, 400, {"error": "invalid json"})
            return

        query = body.get("query", "")
        categories = body.get("categories", [])
        limit = int(body.get("limit", 10))
        min_confidence = float(body.get("min_confidence", 0.0))

        like = f"%{query}%"
        params = [like, like, like]

        sql = """SELECT * FROM memories
                 WHERE (key LIKE ? OR value LIKE ? OR tags LIKE ?)
                 AND confidence >= ?"""
        params = [like, like, like, min_confidence]

        if categories:
            placeholders = ",".join("?" * len(categories))
            sql += f" AND category IN ({placeholders})"
            params.extend(categories)

        sql += " ORDER BY access_count DESC, confidence DESC LIMIT ?"
        params.append(limit)

        conn = _get_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            now = time.time()
            results = []
            for r in rows:
                conn.execute(
                    "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                    (now, r["id"]),
                )
                try:
                    tags_val = json.loads(r["tags"]) if r["tags"] else []
                except Exception:
                    tags_val = r["tags"]
                try:
                    value_val = json.loads(r["value"])
                except Exception:
                    value_val = r["value"]
                results.append({
                    "id": r["id"],
                    "category": r["category"],
                    "key": r["key"],
                    "value": value_val,
                    "confidence": r["confidence"],
                    "tags": tags_val,
                    "access_count": r["access_count"] + 1,
                    "source": r["source"],
                })
            conn.commit()
            _json_response(self, 200, results)
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_list_memories(self, query_string):
        params_map = {}
        if query_string:
            for part in query_string.split("&"):
                if "=" in part:
                    k, _, v = part.partition("=")
                    params_map[k] = v

        category = params_map.get("category")
        tag = params_map.get("tag")
        limit = int(params_map.get("limit", 50))

        sql = "SELECT * FROM memories WHERE 1=1"
        params = []
        if category:
            sql += " AND category=?"
            params.append(category)
        if tag:
            sql += " AND tags LIKE ?"
            params.append(f"%{tag}%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = _get_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for r in rows:
                try:
                    value_val = json.loads(r["value"])
                except Exception:
                    value_val = r["value"]
                try:
                    tags_val = json.loads(r["tags"]) if r["tags"] else []
                except Exception:
                    tags_val = r["tags"]
                results.append({
                    "id": r["id"],
                    "category": r["category"],
                    "key": r["key"],
                    "value": value_val,
                    "source": r["source"],
                    "confidence": r["confidence"],
                    "access_count": r["access_count"],
                    "tags": tags_val,
                    "expires_at": r["expires_at"],
                    "created_at": r["created_at"],
                })
            _json_response(self, 200, results)
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_get_memory(self, mem_id):
        conn = _get_conn()
        try:
            now = time.time()
            row = conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
            if not row:
                _json_response(self, 404, {"error": "not found"})
                return
            conn.execute(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                (now, mem_id),
            )
            conn.commit()
            try:
                value_val = json.loads(row["value"])
            except Exception:
                value_val = row["value"]
            try:
                tags_val = json.loads(row["tags"]) if row["tags"] else []
            except Exception:
                tags_val = row["tags"]
            _json_response(self, 200, {
                "id": row["id"],
                "category": row["category"],
                "key": row["key"],
                "value": value_val,
                "source": row["source"],
                "confidence": row["confidence"],
                "access_count": row["access_count"] + 1,
                "last_accessed": now,
                "tags": tags_val,
                "expires_at": row["expires_at"],
                "created_at": row["created_at"],
            })
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_update_memory(self, mem_id):
        try:
            body = _read_body(self)
        except Exception:
            _json_response(self, 400, {"error": "invalid json"})
            return

        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
            if not row:
                _json_response(self, 404, {"error": "not found"})
                return

            updates = {}
            if "value" in body:
                val = body["value"]
                updates["value"] = json.dumps(val) if not isinstance(val, str) else val
            if "confidence" in body:
                updates["confidence"] = float(body["confidence"])
            if "tags" in body:
                t = body["tags"]
                updates["tags"] = json.dumps(t) if isinstance(t, list) else str(t)

            if not updates:
                _json_response(self, 400, {"error": "nothing to update"})
                return

            set_clause = ", ".join(f"{k}=?" for k in updates)
            values = list(updates.values()) + [mem_id]
            conn.execute(f"UPDATE memories SET {set_clause} WHERE id=?", values)
            conn.commit()
            _json_response(self, 200, {"updated": mem_id})
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_delete_memory(self, mem_id):
        if not _require_admin(self):
            return
        conn = _get_conn()
        try:
            result = conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            conn.commit()
            if result.rowcount == 0:
                _json_response(self, 404, {"error": "not found"})
            else:
                _json_response(self, 200, {"deleted": mem_id})
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_link(self):
        try:
            body = _read_body(self)
        except Exception:
            _json_response(self, 400, {"error": "invalid json"})
            return

        id_a = body.get("memory_id_a")
        id_b = body.get("memory_id_b")
        relationship = body.get("relationship", "related_to")
        strength = float(body.get("strength", 0.5))

        if id_a is None or id_b is None:
            _json_response(self, 400, {"error": "memory_id_a and memory_id_b required"})
            return

        now = time.time()
        conn = _get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO memory_links (memory_id_a, memory_id_b, relationship, strength, created_at) VALUES (?,?,?,?,?)",
                (id_a, id_b, relationship, strength, now),
            )
            # Bidirectional: also insert the reverse
            conn.execute(
                "INSERT INTO memory_links (memory_id_a, memory_id_b, relationship, strength, created_at) VALUES (?,?,?,?,?)",
                (id_b, id_a, relationship, strength, now),
            )
            conn.commit()
            _json_response(self, 200, {"link_id": cur.lastrowid})
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_get_links(self, mem_id):
        conn = _get_conn()
        try:
            links = conn.execute(
                """SELECT ml.id, ml.memory_id_b as linked_id, ml.relationship, ml.strength,
                          m.category, m.key, m.value, m.confidence
                   FROM memory_links ml
                   JOIN memories m ON m.id = ml.memory_id_b
                   WHERE ml.memory_id_a=?""",
                (mem_id,),
            ).fetchall()
            results = []
            for lnk in links:
                try:
                    value_val = json.loads(lnk["value"])
                except Exception:
                    value_val = lnk["value"]
                results.append({
                    "link_id": lnk["id"],
                    "linked_memory_id": lnk["linked_id"],
                    "relationship": lnk["relationship"],
                    "strength": lnk["strength"],
                    "category": lnk["category"],
                    "key": lnk["key"],
                    "value": value_val,
                    "confidence": lnk["confidence"],
                })
            _json_response(self, 200, results)
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_consolidate(self):
        try:
            body = _read_body(self)
        except Exception:
            body = {}

        category_filter = body.get("category")
        conn = _get_conn()
        try:
            now = time.time()
            if category_filter:
                cats = [(category_filter, conn.execute(
                    "SELECT COUNT(*) as cnt FROM memories WHERE category=?",
                    (category_filter,)
                ).fetchone()["cnt"])]
            else:
                cats = [
                    (r["category"], r["cnt"])
                    for r in conn.execute(
                        "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
                    ).fetchall()
                ]

            consolidated = 0
            categories_processed = 0
            for cat, cnt in cats:
                categories_processed += 1
                summary = _consolidate_category(conn, cat)
                conn.execute(
                    "INSERT INTO consolidations (category, summary, memory_count, created_at) VALUES (?,?,?,?)",
                    (cat, summary, cnt, now),
                )
                consolidated += cnt
            conn.commit()
            _json_response(self, 200, {
                "consolidated": consolidated,
                "categories_processed": categories_processed,
            })
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_list_consolidations(self):
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM consolidations ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            results = [dict(r) for r in rows]
            _json_response(self, 200, results)
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_backup(self):
        conn = _get_conn()
        try:
            now = time.time()
            gist_id, record_count = _backup_to_gist(conn)
            conn.execute(
                "INSERT INTO backups (destination, records_backed_up, gist_id, status, created_at) VALUES (?,?,?,?,?)",
                ("github_gist", record_count, gist_id, "ok", now),
            )
            conn.commit()
            _json_response(self, 200, {
                "backed_up": record_count,
                "gist_id": gist_id,
                "status": "ok",
            })
        except Exception as e:
            conn.execute(
                "INSERT INTO backups (destination, records_backed_up, gist_id, status, created_at) VALUES (?,?,?,?,?)",
                ("github_gist", 0, None, f"error: {e}", time.time()),
            )
            conn.commit()
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_list_backups(self):
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM backups ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            _json_response(self, 200, [dict(r) for r in rows])
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_stats(self):
        conn = _get_conn()
        try:
            by_cat = {
                r["category"]: r["cnt"]
                for r in conn.execute(
                    "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
                ).fetchall()
            }
            total_links = conn.execute("SELECT COUNT(*) as cnt FROM memory_links").fetchone()["cnt"]
            last_cons = conn.execute(
                "SELECT created_at FROM consolidations ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            last_bk = conn.execute(
                "SELECT created_at, status FROM backups ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            _json_response(self, 200, {
                "memories_by_category": by_cat,
                "total_memories": sum(by_cat.values()),
                "total_links": total_links,
                "last_consolidation": last_cons["created_at"] if last_cons else None,
                "last_backup": dict(last_bk) if last_bk else None,
            })
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_analytics(self):
        conn = _get_conn()
        try:
            # Most accessed memories
            top_accessed = conn.execute(
                "SELECT id, category, key, access_count FROM memories ORDER BY access_count DESC LIMIT 10"
            ).fetchall()

            # Freshest categories (most recently created entry)
            freshest = conn.execute(
                "SELECT category, MAX(created_at) as latest FROM memories GROUP BY category ORDER BY latest DESC"
            ).fetchall()

            # Memory growth: count by day (last 30 days)
            growth = conn.execute(
                """SELECT CAST(created_at/86400 AS INTEGER) as day_bucket, COUNT(*) as cnt
                   FROM memories
                   WHERE created_at > ?
                   GROUP BY day_bucket
                   ORDER BY day_bucket""",
                (time.time() - 30 * 86400,),
            ).fetchall()

            # Total stored KB (approximate)
            total_size = conn.execute(
                "SELECT SUM(LENGTH(value) + LENGTH(key)) as sz FROM memories"
            ).fetchone()["sz"] or 0

            _json_response(self, 200, {
                "most_accessed": [
                    {"id": r["id"], "category": r["category"], "key": r["key"], "access_count": r["access_count"]}
                    for r in top_accessed
                ],
                "freshest_categories": [
                    {"category": r["category"], "latest_created_at": r["latest"]}
                    for r in freshest
                ],
                "memory_growth_30d": [
                    {"day_bucket": r["day_bucket"], "count": r["cnt"]}
                    for r in growth
                ],
                "total_stored_kb": round(total_size / 1024, 2),
            })
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()

    def _handle_bulk_import(self):
        try:
            body = _read_body(self)
        except Exception:
            _json_response(self, 400, {"error": "invalid json"})
            return

        memories = body.get("memories", [])
        if not isinstance(memories, list):
            _json_response(self, 400, {"error": "memories must be a list"})
            return

        imported = 0
        skipped = 0
        conn = _get_conn()
        try:
            for item in memories:
                try:
                    category = item.get("category", "fact")
                    if category not in VALID_CATEGORIES:
                        skipped += 1
                        continue
                    key = item.get("key")
                    value = item.get("value")
                    if not key or value is None:
                        skipped += 1
                        continue
                    value_str = json.dumps(value) if not isinstance(value, str) else value
                    source = item.get("source", "bulk_import")
                    confidence = float(item.get("confidence", 1.0))
                    tags = item.get("tags", [])
                    tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)
                    expires_in_days = item.get("expires_in_days")
                    expires_at = None
                    if expires_in_days is not None:
                        expires_at = time.time() + float(expires_in_days) * 86400
                    _upsert_memory(conn, category, key, value_str, source, confidence, tags_str, expires_at)
                    imported += 1
                except Exception:
                    skipped += 1
            conn.commit()
            _json_response(self, 200, {"imported": imported, "skipped": skipped})
        except Exception as e:
            _json_response(self, 500, {"error": str(e)})
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _init_db()

    # Start background consolidation daemon
    t = threading.Thread(target=_consolidation_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), SovereignMemoryHandler)
    print(f"[sovereign-memory] Listening on port {PORT}, DB: {DB_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
