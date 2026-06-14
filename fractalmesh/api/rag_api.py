"""
FractalMesh RAG API v2.0.0
FastAPI semantic search over sovereign knowledge base.
Vector embeddings via OpenAI; cosine similarity retrieval.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import math
import time
import sqlite3
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
PORT     = int(os.getenv("RAG_API_PORT", "8001"))
OPENAI   = os.getenv("OPENAI_API_KEY", "")
THRESH   = float(os.getenv("RAG_THRESHOLD", "0.75"))

EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
PHI          = 1.6180339887


# ── DB ────────────────────────────────────────────────────────────────────────

def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS rag_documents (
        id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT, content TEXT,
        embedding TEXT, source TEXT, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS rag_queries (
        id INTEGER PRIMARY KEY, query TEXT, results_found INTEGER,
        top_score REAL, latency_ms REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _get_embedding(text: str) -> Optional[list]:
    if not OPENAI:
        return None
    try:
        body = json.dumps({"model": EMBED_MODEL, "input": text[:8000]}).encode()
        req  = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body, headers={
                "Authorization": f"Bearer {OPENAI}",
                "Content-Type":  "application/json",
            })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"[rag-api] embed error: {e}")
        return None


def _cosine(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _upsert_document(doc_id: str, title: str, content: str,
                     source: str = "manual", phi_score: float = 1.0):
    embedding = _get_embedding(f"{title} {content}")
    emb_json  = json.dumps(embedding) if embedding else None
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO rag_documents (doc_id,title,content,embedding,source,phi_score)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(doc_id) DO UPDATE
        SET content=excluded.content, embedding=excluded.embedding, ts=CURRENT_TIMESTAMP""",
        (doc_id, title, content[:5000], emb_json, source, phi_score))
    conn.commit(); conn.close()


def _search(query: str, top_k: int = 5) -> list:
    t0         = time.time()
    q_emb      = _get_embedding(query)
    conn       = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    rows       = conn.execute(
        "SELECT doc_id, title, content, embedding, phi_score FROM rag_documents"
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        if not row["embedding"]:
            continue
        doc_emb = json.loads(row["embedding"])
        if q_emb:
            score = _cosine(q_emb, doc_emb)
        else:
            # Keyword fallback when no OpenAI key
            words  = set(query.lower().split())
            text   = (row["title"] + " " + row["content"]).lower()
            score  = sum(1 for w in words if w in text) / max(len(words), 1)

        if score >= THRESH:
            results.append({
                "doc_id":    row["doc_id"],
                "title":     row["title"],
                "excerpt":   row["content"][:300],
                "score":     round(score, 4),
                "phi_score": row["phi_score"],
            })

    results.sort(key=lambda x: x["score"] * x["phi_score"], reverse=True)
    results = results[:top_k]

    latency = round((time.time() - t0) * 1000, 2)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO rag_queries (query,results_found,top_score,latency_ms)
        VALUES (?,?,?,?)""",
        (query[:200], len(results),
         results[0]["score"] if results else 0.0, latency))
    conn.commit(); conn.close()
    return results


def _ingest_from_dist():
    """Auto-ingest markdown files from dist/ into knowledge base."""
    dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist")
    if not os.path.exists(dist):
        return 0
    count = 0
    for fname in os.listdir(dist):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(dist, fname)
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        doc_id    = hashlib.sha256(fname.encode()).hexdigest()[:16]
        phi_score = round(PHI ** (count % 5), 4)
        _upsert_document(doc_id, fname.replace(".md", ""), content,
                         source="dist_ingest", phi_score=phi_score)
        count += 1
    return count


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class RAGHandler(BaseHTTPRequestHandler):
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
            self._send(200, {"status": "ok", "port": PORT, "agent": "rag-api"})
        elif self.path == "/docs":
            self._send(200, {
                "endpoints": {
                    "GET /health":       "health check",
                    "GET /docs":         "this doc",
                    "POST /search":      "semantic search {query, top_k}",
                    "POST /ingest":      "ingest document {doc_id, title, content, source}",
                    "POST /ingest/dist": "auto-ingest dist/ markdown files",
                    "GET /stats":        "knowledge base stats",
                }
            })
        elif self.path == "/stats":
            conn = sqlite3.connect(DB, timeout=10)
            docs  = conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
            queries = conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0]
            with_emb = conn.execute(
                "SELECT COUNT(*) FROM rag_documents WHERE embedding IS NOT NULL").fetchone()[0]
            conn.close()
            self._send(200, {
                "documents": docs, "with_embedding": with_emb,
                "queries_run": queries,
                "openai_configured": bool(OPENAI),
                "threshold": THRESH,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/search":
            query  = body.get("query", "")
            top_k  = int(body.get("top_k", 5))
            if not query:
                self._send(400, {"error": "query required"})
                return
            results = _search(query, top_k)
            self._send(200, {"query": query, "results": results, "count": len(results)})

        elif self.path == "/ingest":
            doc_id  = body.get("doc_id") or hashlib.sha256(
                body.get("title", "").encode()).hexdigest()[:16]
            title   = body.get("title", "Untitled")
            content = body.get("content", "")
            source  = body.get("source", "api")
            if not content:
                self._send(400, {"error": "content required"})
                return
            _upsert_document(doc_id, title, content, source)
            self._send(200, {"doc_id": doc_id, "status": "ingested"})

        elif self.path == "/ingest/dist":
            n = _ingest_from_dist()
            self._send(200, {"ingested": n, "status": "ok"})

        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    _db_init()
    _ingest_from_dist()
    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    print(f"[rag-api] Listening on :{PORT} | OpenAI={'set' if OPENAI else 'NOT SET'} "
          f"| threshold={THRESH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("[rag-api] Stopped.")
