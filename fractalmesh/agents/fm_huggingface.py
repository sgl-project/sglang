#!/usr/bin/env python3
"""
fm_huggingface.py — Hugging Face Hub Agent (Port 7790)
Inference API routing, dataset publishing, Spaces management, model card updates.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("HF_PORT", "7790"))
ROOT         = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
LOG          = ROOT / "logs" / "huggingface.log"
HF_TOKEN     = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", ""))
HF_USERNAME  = os.getenv("HF_USERNAME", "fractalmesh")
HF_API       = "https://api-inference.huggingface.co"
HF_HUB_API   = "https://huggingface.co/api"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HUGGINGFACE] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("huggingface")

# ── model registry ────────────────────────────────────────────────────────────
# Primary inference models — vault key HF_MODEL_<NAME> to override
MODELS = {
    "text_generation": os.getenv("HF_MODEL_TEXT",    "mistralai/Mistral-7B-Instruct-v0.3"),
    "text_embedding":  os.getenv("HF_MODEL_EMBED",   "BAAI/bge-small-en-v1.5"),
    "classification":  os.getenv("HF_MODEL_CLASS",   "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    "summarization":   os.getenv("HF_MODEL_SUMMARY", "facebook/bart-large-cnn"),
    "image_gen":       os.getenv("HF_MODEL_IMAGE",   "stabilityai/stable-diffusion-xl-base-1.0"),
    "zero_shot":       os.getenv("HF_MODEL_ZERO",    "facebook/bart-large-mnli"),
    "code":            os.getenv("HF_MODEL_CODE",    "bigcode/starcoder2-7b"),
    "reranker":        os.getenv("HF_MODEL_RERANK",  "cross-encoder/ms-marco-MiniLM-L-6-v2"),
}

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hf_inference_log (
            id          INTEGER PRIMARY KEY,
            model_type  TEXT,
            model_id    TEXT,
            input_len   INTEGER,
            output_len  INTEGER,
            latency_ms  REAL,
            status      TEXT,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hf_usage (
            model_id    TEXT PRIMARY KEY,
            call_count  INTEGER DEFAULT 0,
            total_ms    REAL DEFAULT 0,
            last_used   DATETIME
        )
    """)
    conn.commit()
    conn.close()

def _db_log(model_type: str, model_id: str, input_len: int,
            output_len: int, latency_ms: float, status: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO hf_inference_log (model_type,model_id,input_len,output_len,latency_ms,status) "
            "VALUES (?,?,?,?,?,?)",
            (model_type, model_id, input_len, output_len, latency_ms, status),
        )
        conn.execute("""
            INSERT INTO hf_usage (model_id,call_count,total_ms,last_used)
            VALUES (?,1,?,CURRENT_TIMESTAMP)
            ON CONFLICT(model_id) DO UPDATE SET
                call_count=call_count+1,
                total_ms=total_ms+excluded.total_ms,
                last_used=CURRENT_TIMESTAMP
        """, (model_id, latency_ms))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

# ── inference helpers ─────────────────────────────────────────────────────────

def _hf_infer(model_id: str, payload: dict, wait: bool = True) -> dict:
    """Call HF Inference API. Returns raw response dict."""
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not configured", "model": model_id}
    headers = {
        "Authorization":  f"Bearer {HF_TOKEN}",
        "Content-Type":   "application/json",
    }
    if wait:
        headers["X-Wait-For-Model"] = "true"

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{HF_API}/models/{model_id}",
        data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"http_{e.code}", "detail": body[:300]}
    except Exception as e:
        return {"error": str(e)}

def _infer(task: str, inputs: str | list | dict, **kwargs) -> dict:
    model_id = MODELS.get(task, MODELS["text_generation"])
    t0       = time.time()
    payload  = {"inputs": inputs, "parameters": kwargs} if kwargs else {"inputs": inputs}
    result   = _hf_infer(model_id, payload)
    latency  = (time.time() - t0) * 1000
    in_len   = len(str(inputs))
    out_len  = len(str(result))
    status   = "error" if "error" in result else "ok"
    _db_log(task, model_id, in_len, out_len, latency, status)
    log.info("infer task=%s model=%s latency=%.0fms status=%s", task, model_id, latency, status)
    return {"task": task, "model": model_id, "result": result, "latency_ms": round(latency, 1)}

# ── Hub API helpers ───────────────────────────────────────────────────────────

def _hub_get(path: str) -> dict:
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not configured"}
    req = urllib.request.Request(
        f"{HF_HUB_API}/{path.lstrip('/')}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def _list_models() -> dict:
    return _hub_get(f"models?author={HF_USERNAME}&limit=20")

def _list_datasets() -> dict:
    return _hub_get(f"datasets?author={HF_USERNAME}&limit=20")

def _list_spaces() -> dict:
    return _hub_get(f"spaces?author={HF_USERNAME}&limit=20")

def _get_model_info(model_id: str) -> dict:
    return _hub_get(f"models/{model_id}")

def _usage_stats() -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute("""
            SELECT model_id, call_count, total_ms, last_used
            FROM hf_usage ORDER BY call_count DESC LIMIT 20
        """).fetchall()
        conn.close()
        return [{"model_id": r[0], "call_count": r[1],
                 "avg_ms": round(r[2]/max(r[1],1), 1), "last_used": r[3]}
                for r in rows]
    except Exception:
        return []

# ── HTTP handler ───────────────────────────────────────────────────────────────

class HuggingFaceHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "hf_token": bool(HF_TOKEN),
                                "models": MODELS, "username": HF_USERNAME})
        elif self.path == "/models":
            self._respond(200, {"registered": MODELS, "hub": _list_models()})
        elif self.path == "/datasets":
            self._respond(200, _list_datasets())
        elif self.path == "/spaces":
            self._respond(200, _list_spaces())
        elif self.path == "/usage":
            self._respond(200, {"usage": _usage_stats()})
        elif self.path.startswith("/model/"):
            mid = self.path[7:]
            self._respond(200, _get_model_info(mid))
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))

            if self.path == "/infer":
                task   = data.get("task", "text_generation")
                inputs = data.get("inputs", data.get("prompt", ""))
                params = data.get("parameters", {})
                if not inputs:
                    self._respond(400, {"error": "inputs required"})
                    return
                result = _infer(task, inputs, **params)
                self._respond(200, result)

            elif self.path == "/generate":
                prompt    = data.get("prompt", "")
                max_tokens = data.get("max_new_tokens", 256)
                temp      = data.get("temperature", 0.7)
                result    = _infer("text_generation", prompt,
                                   max_new_tokens=max_tokens, temperature=temp,
                                   return_full_text=False)
                self._respond(200, result)

            elif self.path == "/embed":
                text   = data.get("text", data.get("inputs", ""))
                result = _infer("text_embedding", text)
                self._respond(200, result)

            elif self.path == "/classify":
                text       = data.get("text", "")
                candidates = data.get("candidate_labels", ["positive", "negative", "neutral"])
                result     = _infer("zero_shot", {"text": text, "candidate_labels": candidates})
                self._respond(200, result)

            elif self.path == "/summarize":
                text     = data.get("text", "")
                max_len  = data.get("max_length", 130)
                min_len  = data.get("min_length", 30)
                result   = _infer("summarization", text, max_length=max_len, min_length=min_len)
                self._respond(200, result)

            else:
                self._respond(404, {"error": "unknown_path"})

        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
        except Exception as e:
            log.error("handler_error: %s", e)
            self._respond(500, {"error": str(e)})

# ── main ───────────────────────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), HuggingFaceHandler)
    log.info("Hugging Face agent listening on port %d", PORT)
    log.info("Username: %s | Token: %s | Models: %d",
             HF_USERNAME, "configured" if HF_TOKEN else "fallback", len(MODELS))
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Hugging Face agent stopped")

if __name__ == "__main__":
    main()
