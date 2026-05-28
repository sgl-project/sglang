#!/usr/bin/env python3
"""
fm_rag_pipeline.py — RAG Pipeline Agent (Port 7806)
Ingest, chunk, index, retrieve, and generate. Pure Python TF-IDF similarity.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import math
import os
import re
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT       = int(os.getenv("RAG_PORT", "7806"))
ROOT       = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB         = ROOT / "database" / "sovereign.db"
LOG        = ROOT / "logs" / "rag_pipeline.log"
OR_URL     = os.getenv("OPENROUTER_URL", "http://127.0.0.1:7791")
RSS_URL    = os.getenv("RSS_URL",        "http://127.0.0.1:7805")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
OVERLAP    = int(os.getenv("RAG_OVERLAP",    "50"))

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [RAG] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("rag")

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS rag_documents (
        id INTEGER PRIMARY KEY, source TEXT, title TEXT, content TEXT,
        chunk_id INT, word_count INT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS rag_queries (
        id INTEGER PRIMARY KEY, query TEXT, retrieved_chunks INT,
        answer_len INT, latency_ms REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

# ── text processing ───────────────────────────────────────────────────────────

_STOPWORDS = set("the a an and or but in on at to for of with is are was were be been have has had do does did will would could should may might shall this that these those".split())

def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text) if w.lower() not in _STOPWORDS and len(w) > 2]

def _chunk(text: str, source: str, title: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> list[dict]:
    words = text.split(); chunks = []; cid = 0; i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text  = " ".join(chunk_words)
        chunks.append({"source":source,"title":title,"content":chunk_text,
                       "chunk_id":cid,"word_count":len(chunk_words)})
        cid += 1; i += chunk_size - overlap
    return chunks

def _tfidf(docs: list[str]) -> list[dict]:
    """Compute TF-IDF vectors as {term: score} dicts."""
    tokenised = [_tokenise(d) for d in docs]
    df: Counter = Counter()
    for tokens in tokenised:
        df.update(set(tokens))
    N = len(docs)
    vectors = []
    for tokens in tokenised:
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            tfidf_score = (count / max(len(tokens),1)) * math.log((N + 1) / (df[term] + 1))
            vec[term]   = tfidf_score
        vectors.append(vec)
    return vectors

def _cosine(v1: dict, v2: dict) -> float:
    common = set(v1) & set(v2)
    if not common: return 0.0
    dot   = sum(v1[t] * v2[t] for t in common)
    mag1  = math.sqrt(sum(s*s for s in v1.values()))
    mag2  = math.sqrt(sum(s*s for s in v2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    c = sqlite3.connect(DB, timeout=10)
    rows = c.execute("SELECT id,source,title,content FROM rag_documents ORDER BY ts DESC LIMIT 500").fetchall()
    c.close()
    if not rows: return []
    docs     = [r[3] for r in rows]
    all_docs = [query] + docs
    vectors  = _tfidf(all_docs)
    qvec     = vectors[0]; dvecs = vectors[1:]
    scored   = sorted([(i, _cosine(qvec, dvecs[i])) for i in range(len(dvecs))],
                      key=lambda x:-x[1])[:top_k]
    return [{"id":rows[i[0]][0],"source":rows[i[0]][1],"title":rows[i[0]][2],
             "content":rows[i[0]][3][:800],"score":round(i[1],4)} for i in scored if i[1] > 0]

# ── ingestion ─────────────────────────────────────────────────────────────────

def _ingest(content: str, source: str, title: str) -> dict:
    chunks = _chunk(content, source, title)
    c = sqlite3.connect(DB, timeout=10)
    saved  = 0
    for ch in chunks:
        c.execute("INSERT INTO rag_documents (source,title,content,chunk_id,word_count) VALUES (?,?,?,?,?)",
            (ch["source"],ch["title"],ch["content"],ch["chunk_id"],ch["word_count"]))
        saved += 1
    c.commit(); c.close()
    log.info("ingest source=%s chunks=%d", source, saved)
    return {"source":source,"chunks":saved,"words":sum(ch["word_count"] for ch in chunks)}

def _ingest_url(url: str, title: str) -> dict:
    try:
        req = urllib.request.Request(url,
            headers={"User-Agent":"FractalMesh-RAG/2.1","Accept":"text/html,text/plain"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8","replace")
        # Strip HTML tags
        text = re.sub(r'<[^>]+>','',raw)
        text = re.sub(r'\s+',' ',text).strip()
        return _ingest(text[:50000], url, title or url)
    except Exception as e: return {"error":str(e),"url":url}

def _ingest_rss() -> dict:
    try:
        with urllib.request.urlopen(f"{RSS_URL}/digest",timeout=5) as r:
            digest = json.loads(r.read()).get("digest",[])
    except Exception as e: return {"error":str(e)}
    total = 0
    for item in digest:
        if item.get("link"): r = _ingest_url(item["link"],item.get("title","")); total += r.get("chunks",0)
    return {"ingested":len(digest),"chunks":total}

def _generate(query: str, chunks: list) -> str:
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i,c in enumerate(chunks))
    prompt  = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer precisely using only the context."
    payload = json.dumps({"task":"extract","tier":"balanced","prompt":prompt,"max_tokens":512}).encode()
    req = urllib.request.Request(f"{OR_URL}/route",data=payload,
        headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read()).get("content","")
    except Exception as e: return f"[generation unavailable: {e}]"

class RAGHandler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _r(self,code,body):
        p=json.dumps(body).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        import urllib.parse
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]
        if ep == "/health":
            c = sqlite3.connect(DB,timeout=5)
            nd = c.execute("SELECT COUNT(DISTINCT source) FROM rag_documents").fetchone()[0]
            nc = c.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]; c.close()
            self._r(200,{"status":"ok","documents":nd,"chunks":nc})
        elif ep == "/docs":
            c = sqlite3.connect(DB,timeout=5)
            rows = c.execute("SELECT DISTINCT source,title,COUNT(*) as chunks,MAX(ts) as ts FROM rag_documents GROUP BY source ORDER BY ts DESC LIMIT 50").fetchall()
            c.close(); self._r(200,{"docs":[{"source":r[0],"title":r[1],"chunks":r[2],"ts":r[3]} for r in rows]})
        elif ep == "/chunks":
            src = qs.get("source",[""])[0]
            c = sqlite3.connect(DB,timeout=5)
            rows = c.execute("SELECT id,chunk_id,word_count,content FROM rag_documents WHERE source=? ORDER BY chunk_id",(src,)).fetchall()
            c.close(); self._r(200,{"source":src,"chunks":[{"id":r[0],"chunk_id":r[1],"words":r[2],"preview":r[3][:200]} for r in rows]})
        else:
            self._r(404,{"error":"not_found"})

    def do_POST(self):
        try:
            n=int(self.headers.get("Content-Length",0)); d=json.loads(self.rfile.read(n))
            t0=time.time(); ep=self.path.split("?")[0]
            if ep=="/ingest":
                self._r(200,_ingest(d.get("content",""),d.get("source","manual"),d.get("title","")))
            elif ep=="/ingest_url":
                self._r(200,_ingest_url(d.get("url",""),d.get("title","")))
            elif ep=="/ingest_rss":
                self._r(200,_ingest_rss())
            elif ep=="/query":
                query  = d.get("query",""); top_k = int(d.get("top_k",5))
                chunks = _retrieve(query,top_k)
                answer = _generate(query,chunks) if chunks else "No relevant documents found."
                lat    = (time.time()-t0)*1000
                c = sqlite3.connect(DB,timeout=5)
                c.execute("INSERT INTO rag_queries (query,retrieved_chunks,answer_len,latency_ms) VALUES (?,?,?,?)",
                    (query,len(chunks),len(answer),lat)); c.commit(); c.close()
                self._r(200,{"query":query,"chunks_retrieved":len(chunks),"chunks":chunks,"answer":answer,"latency_ms":round(lat,1)})
            elif ep=="/delete":
                doc_id=d.get("id","")
                c=sqlite3.connect(DB,timeout=5)
                c.execute("DELETE FROM rag_documents WHERE source=?",(doc_id,)); c.commit(); c.close()
                self._r(200,{"deleted":doc_id})
            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: self._r(500,{"error":str(e)})

_running=True
def _shutdown(*_): global _running; _running=False
signal.signal(signal.SIGTERM,_shutdown); signal.signal(signal.SIGINT,_shutdown)

def main():
    _db_init(); server=HTTPServer(("0.0.0.0",PORT),RAGHandler)
    log.info("RAG Pipeline on port %d | chunk=%d overlap=%d",PORT,CHUNK_SIZE,OVERLAP)
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__=="__main__": main()
