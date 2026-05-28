#!/usr/bin/env python3
"""
fm_recommendation_engine.py — AI Recommendation Engine (Port 7902)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import hashlib
import hmac
import json
import math
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

PORT         = int(os.getenv("RECOMMENDATION_ENGINE_PORT", "7902"))
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
        CREATE TABLE IF NOT EXISTS rec_items (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id      TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            category     TEXT NOT NULL,
            tags         TEXT NOT NULL DEFAULT '[]',
            price        REAL DEFAULT 0,
            metadata     TEXT NOT NULL DEFAULT '{}',
            active       INTEGER NOT NULL DEFAULT 1,
            view_count   INTEGER NOT NULL DEFAULT 0,
            purchase_count INTEGER NOT NULL DEFAULT 0,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rec_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id     TEXT UNIQUE NOT NULL,
            user_id      TEXT NOT NULL,
            item_id      TEXT NOT NULL,
            event_type   TEXT NOT NULL,
            value        REAL NOT NULL DEFAULT 1.0,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_profiles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      TEXT UNIQUE NOT NULL,
            category_scores TEXT NOT NULL DEFAULT '{}',
            tag_scores   TEXT NOT NULL DEFAULT '{}',
            last_seen    REAL,
            event_count  INTEGER NOT NULL DEFAULT 0,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS item_similarity (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            item_a       TEXT NOT NULL,
            item_b       TEXT NOT NULL,
            score        REAL NOT NULL,
            computed_at  REAL NOT NULL,
            UNIQUE(item_a, item_b)
        );
        CREATE INDEX IF NOT EXISTS idx_rec_events_user  ON rec_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_rec_events_item  ON rec_events(item_id);
        CREATE INDEX IF NOT EXISTS idx_item_sim_a       ON item_similarity(item_a);
    """)
    con.commit()
    con.close()

# ── recommendation algorithms ─────────────────────────────────────────────────
def _cosine_similarity(vec_a, vec_b):
    keys = set(vec_a) | set(vec_b)
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(v*v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v*v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def _build_item_vector(item):
    vec = {}
    tags = json.loads(item["tags"]) if isinstance(item["tags"], str) else item["tags"]
    for tag in tags:
        vec[f"tag:{tag}"] = 1.0
    vec[f"cat:{item['category']}"] = 2.0
    if item["price"]:
        price_bucket = f"price:{int(item['price'] // 50) * 50}"
        vec[price_bucket] = 0.5
    return vec

def _get_user_vector(con, user_id):
    row = con.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return {}
    vec = {}
    cat_scores = json.loads(row["category_scores"])
    tag_scores = json.loads(row["tag_scores"])
    for k, v in cat_scores.items():
        vec[f"cat:{k}"] = v
    for k, v in tag_scores.items():
        vec[f"tag:{k}"] = v
    return vec

def _update_user_profile(con, user_id, item_id, event_type, value):
    weights = {"view": 0.5, "click": 1.0, "purchase": 5.0, "like": 2.0, "add_to_cart": 3.0}
    weight = weights.get(event_type, 1.0) * value
    item = con.execute("SELECT * FROM rec_items WHERE item_id=?", (item_id,)).fetchone()
    if not item:
        return
    tags = json.loads(item["tags"]) if isinstance(item["tags"], str) else item["tags"]
    category = item["category"]
    now = time.time()
    row = con.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
    if row:
        cat_scores = json.loads(row["category_scores"])
        tag_scores = json.loads(row["tag_scores"])
        cat_scores[category] = cat_scores.get(category, 0) + weight
        for t in tags:
            tag_scores[t] = tag_scores.get(t, 0) + weight * 0.5
        con.execute(
            "UPDATE user_profiles SET category_scores=?, tag_scores=?, last_seen=?, event_count=event_count+1, updated_at=? "
            "WHERE user_id=?",
            (json.dumps(cat_scores), json.dumps(tag_scores), now, now, user_id)
        )
    else:
        cat_scores = {category: weight}
        tag_scores = {t: weight * 0.5 for t in tags}
        con.execute(
            "INSERT INTO user_profiles(user_id,category_scores,tag_scores,last_seen,event_count,updated_at) VALUES(?,?,?,?,1,?)",
            (user_id, json.dumps(cat_scores), json.dumps(tag_scores), now, now)
        )

def _content_based_recommend(con, user_id, limit=10, exclude_seen=True):
    user_vec = _get_user_vector(con, user_id)
    if not user_vec:
        # cold start: return popular items
        rows = con.execute(
            "SELECT * FROM rec_items WHERE active=1 ORDER BY purchase_count DESC, view_count DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    if exclude_seen:
        seen_ids = {r["item_id"] for r in con.execute(
            "SELECT DISTINCT item_id FROM rec_events WHERE user_id=?", (user_id,)
        ).fetchall()}
    else:
        seen_ids = set()

    items = con.execute("SELECT * FROM rec_items WHERE active=1").fetchall()
    scored = []
    for item in items:
        if item["item_id"] in seen_ids:
            continue
        item_vec = _build_item_vector(item)
        score = _cosine_similarity(user_vec, item_vec)
        # boost by popularity
        score += math.log1p(item["purchase_count"]) * 0.1
        scored.append((score, dict(item)))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [{"item": i, "score": round(s, 4)} for s, i in scored[:limit]]

def _collab_recommend(con, user_id, limit=10):
    user_events = con.execute(
        "SELECT item_id, SUM(value) as score FROM rec_events WHERE user_id=? GROUP BY item_id",
        (user_id,)
    ).fetchall()
    user_items = {r["item_id"]: r["score"] for r in user_events}
    if not user_items:
        return []
    # find similar items via precomputed similarity
    candidate_scores = {}
    for item_id, user_score in user_items.items():
        sims = con.execute(
            "SELECT item_b, score FROM item_similarity WHERE item_a=? ORDER BY score DESC LIMIT 10",
            (item_id,)
        ).fetchall()
        for sim in sims:
            if sim["item_b"] not in user_items:
                candidate_scores[sim["item_b"]] = candidate_scores.get(sim["item_b"], 0) + sim["score"] * user_score
    if not candidate_scores:
        return []
    top_ids = sorted(candidate_scores, key=lambda k: candidate_scores[k], reverse=True)[:limit]
    results = []
    for iid in top_ids:
        row = con.execute("SELECT * FROM rec_items WHERE item_id=? AND active=1", (iid,)).fetchone()
        if row:
            results.append({"item": dict(row), "score": round(candidate_scores[iid], 4)})
    return results

def _recompute_similarities(con):
    items = con.execute("SELECT * FROM rec_items WHERE active=1").fetchall()
    item_vecs = [(item["item_id"], _build_item_vector(item)) for item in items]
    now = time.time()
    batch = []
    for i, (ia, va) in enumerate(item_vecs):
        for j, (ib, vb) in enumerate(item_vecs):
            if i >= j:
                continue
            score = _cosine_similarity(va, vb)
            if score > 0.01:
                batch.append((ia, ib, score, now))
                batch.append((ib, ia, score, now))
    if batch:
        con.executemany(
            "INSERT INTO item_similarity(item_a,item_b,score,computed_at) VALUES(?,?,?,?) "
            "ON CONFLICT(item_a,item_b) DO UPDATE SET score=excluded.score, computed_at=excluded.computed_at",
            batch
        )
        con.commit()

def _similarity_daemon():
    while True:
        time.sleep(3600)
        try:
            con = _db()
            _recompute_similarities(con)
            con.close()
        except Exception:
            pass

threading.Thread(target=_similarity_daemon, daemon=True).start()

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

class RecHandler(BaseHTTPRequestHandler):
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
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
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
                return _j({"status": "ok", "port": PORT, "agent": "fm_recommendation_engine"})

            if p == ["items"]:
                cat = qs.get("category", [None])[0]
                limit = int(qs.get("limit", ["50"])[0])
                if cat:
                    rows = con.execute("SELECT * FROM rec_items WHERE active=1 AND category=? LIMIT ?", (cat, limit)).fetchall()
                else:
                    rows = con.execute("SELECT * FROM rec_items WHERE active=1 ORDER BY purchase_count DESC LIMIT ?", (limit,)).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "recommend" and p[1]:
                user_id = p[1]
                limit = int(qs.get("limit", ["10"])[0])
                algorithm = qs.get("algo", ["content"])[0]
                if algorithm == "collab":
                    recs = _collab_recommend(con, user_id, limit)
                else:
                    recs = _content_based_recommend(con, user_id, limit)
                return _j({"user_id": user_id, "algorithm": algorithm, "recommendations": recs})

            if len(p) == 3 and p[0] == "items" and p[2] == "similar":
                limit = int(qs.get("limit", ["10"])[0])
                sims = con.execute(
                    "SELECT i.*, s.score FROM item_similarity s "
                    "JOIN rec_items i ON s.item_b=i.item_id "
                    "WHERE s.item_a=? AND i.active=1 ORDER BY s.score DESC LIMIT ?",
                    (p[1], limit)
                ).fetchall()
                return _j([dict(r) for r in sims])

            if p == ["popular"]:
                limit = int(qs.get("limit", ["20"])[0])
                cat = qs.get("category", [None])[0]
                q = "SELECT * FROM rec_items WHERE active=1"
                vals = []
                if cat:
                    q += " AND category=?"; vals.append(cat)
                q += " ORDER BY purchase_count DESC, view_count DESC LIMIT ?"
                vals.append(limit)
                rows = con.execute(q, vals).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 3 and p[0] == "users" and p[2] == "profile":
                row = con.execute("SELECT * FROM user_profiles WHERE user_id=?", (p[1],)).fetchone()
                if not row:
                    return _err("User profile not found", 404)
                return _j(dict(row))

            if p == ["stats"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                total_items = con.execute("SELECT COUNT(*) FROM rec_items WHERE active=1").fetchone()[0]
                total_events = con.execute("SELECT COUNT(*) FROM rec_events").fetchone()[0]
                total_users = con.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
                sim_pairs = con.execute("SELECT COUNT(*) FROM item_similarity").fetchone()[0]
                return _j({
                    "total_items": total_items,
                    "total_events": total_events,
                    "total_user_profiles": total_users,
                    "similarity_pairs": sim_pairs,
                })

            return _err("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["items"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                iid = "item_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO rec_items(item_id,name,category,tags,price,metadata,created_at) VALUES(?,?,?,?,?,?,?)",
                    (iid, data.get("name",""), data.get("category",""),
                     json.dumps(data.get("tags",[])), data.get("price",0),
                     json.dumps(data.get("metadata",{})), now)
                )
                con.commit()
                threading.Thread(target=self._async_similarities, daemon=True).start()
                return _j({"item_id": iid}, 201)

            if p == ["events"]:
                eid = "ev_" + secrets.token_hex(8)
                user_id = data.get("user_id","")
                item_id = data.get("item_id","")
                event_type = data.get("event_type","view")
                value = float(data.get("value", 1.0))
                if not all([user_id, item_id]):
                    return _err("user_id and item_id required")
                con.execute(
                    "INSERT INTO rec_events(event_id,user_id,item_id,event_type,value,created_at) VALUES(?,?,?,?,?,?)",
                    (eid, user_id, item_id, event_type, value, now)
                )
                # update item counters
                if event_type in ("view",):
                    con.execute("UPDATE rec_items SET view_count=view_count+1 WHERE item_id=?", (item_id,))
                elif event_type == "purchase":
                    con.execute("UPDATE rec_items SET purchase_count=purchase_count+1 WHERE item_id=?", (item_id,))
                _update_user_profile(con, user_id, item_id, event_type, value)
                con.commit()
                return _j({"event_id": eid}, 201)

            if p == ["similarities", "recompute"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                threading.Thread(target=self._async_similarities, daemon=True).start()
                return _j({"status": "recomputing"})

            return _err("Not found", 404)
        finally:
            con.close()

    def _async_similarities(self):
        try:
            con = _db()
            _recompute_similarities(con)
            con.close()
        except Exception:
            pass


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), RecHandler)
    print(f"[fm_recommendation_engine] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
