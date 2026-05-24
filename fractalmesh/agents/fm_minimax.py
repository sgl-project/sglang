#!/usr/bin/env python3
"""
fm_minimax.py — MiniMax AI Agent (Port 7808)
Text generation via Anthropic-compatible API + TTS via speech-2.8-hd.
Models: MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.5, MiniMax-M2.1, MiniMax-M2
TTS: speech-2.8-hd with voice selection, speed, pitch, sound effects.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, binascii, urllib.request, urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT         = int(os.getenv("MINIMAX_PORT", "7808"))
ROOT         = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
LOG          = ROOT / "logs" / "minimax.log"
AUDIO_DIR    = ROOT / "audio"

# MiniMax API — uses ANTHROPIC env vars for the Anthropic-compat endpoint
MM_API_KEY   = os.getenv("MINIMAX_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
MM_BASE      = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io")
MM_ANTH_BASE = f"{MM_BASE}/anthropic"           # Anthropic-compat
MM_TTS_URL   = f"{MM_BASE}/v1/t2a_v2"          # TTS endpoint

DEFAULT_MODEL= os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
DEFAULT_VOICE= os.getenv("MINIMAX_VOICE", "English_expressive_narrator")

SUPPORTED_MODELS = [
    "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5", "MiniMax-M2.5-highspeed",
    "MiniMax-M2.1", "MiniMax-M2.1-highspeed",
    "MiniMax-M2",
]

VOICES = {
    "English_expressive_narrator": "Expressive English narrator (default)",
    "English_female_calm":         "Calm female English voice",
    "English_male_deep":           "Deep male English voice",
    "English_newsreader":          "Professional newsreader",
    "English_podcast_host":        "Casual podcast host",
}

SOUND_EFFECTS = ["none", "spacious_echo", "studio", "telephone", "concert_hall"]

for p in (ROOT, LOG.parent, DB.parent, AUDIO_DIR): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MINIMAX] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("minimax")

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS minimax_requests (
        id INTEGER PRIMARY KEY, request_type TEXT, model TEXT, prompt_len INT,
        output_len INT, latency_ms REAL, cost_tokens INT, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS minimax_tts (
        id INTEGER PRIMARY KEY, text_len INT, voice TEXT, audio_path TEXT,
        audio_format TEXT, latency_ms REAL, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _db_log_llm(model, prompt_len, output_len, latency_ms, tokens, status):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO minimax_requests (request_type,model,prompt_len,output_len,latency_ms,cost_tokens,status) VALUES (?,?,?,?,?,?,?)",
                  ("llm",model,prompt_len,output_len,latency_ms,tokens,status))
        c.commit(); c.close()
    except Exception as e: log.warning("db: %s",e)

def _db_log_tts(text_len, voice, path, fmt, latency_ms, status):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO minimax_tts (text_len,voice,audio_path,audio_format,latency_ms,status) VALUES (?,?,?,?,?,?)",
                  (text_len,voice,path,fmt,latency_ms,status))
        c.commit(); c.close()
    except Exception as e: log.warning("db: %s",e)

# ── LLM via Anthropic-compatible endpoint ─────────────────────────────────────

def _mm_chat(messages: list, model: str = DEFAULT_MODEL, max_tokens: int = 1024,
             system: str = "", temperature: float = 0.7,
             thinking: bool = False) -> dict:
    """Call MiniMax via Anthropic-compatible API."""
    if not MM_API_KEY:
        return {"error": "MINIMAX_API_KEY not configured"}
    if model not in SUPPORTED_MODELS:
        model = DEFAULT_MODEL

    body: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
        "temperature": max(0.01, min(1.0, temperature)),  # range (0,1]
    }
    if system:
        body["system"] = system
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2048}

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{MM_ANTH_BASE}/v1/messages",
        data=payload,
        method="POST",
        headers={
            "x-api-key":       MM_API_KEY,
            "anthropic-version":"2023-06-01",
            "Content-Type":    "application/json",
            "Accept":          "application/json",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result  = json.loads(r.read())
            latency = (time.time()-t0)*1000
            usage   = result.get("usage",{})
            # Extract content blocks
            content_blocks = result.get("content",[])
            text_out   = "".join(b.get("text","") for b in content_blocks if b.get("type")=="text")
            thinking_out = "".join(b.get("thinking","") for b in content_blocks if b.get("type")=="thinking")
            tokens = usage.get("input_tokens",0) + usage.get("output_tokens",0)
            _db_log_llm(model, len(str(messages)), len(text_out), latency, tokens, "ok")
            log.info("llm model=%s tokens=%d latency=%.0fms", model, tokens, latency)
            return {
                "model": model, "content": text_out,
                "thinking": thinking_out if thinking_out else None,
                "usage": usage, "latency_ms": round(latency,1),
                "stop_reason": result.get("stop_reason"),
            }
    except urllib.error.HTTPError as e:
        body_err = e.read().decode()[:300]
        _db_log_llm(model, 0, 0, (time.time()-t0)*1000, 0, f"err_{e.code}")
        return {"error": f"http_{e.code}", "detail": body_err}
    except Exception as e:
        _db_log_llm(model, 0, 0, (time.time()-t0)*1000, 0, "error")
        return {"error": str(e)}

def _mm_generate(prompt: str, system: str = "", model: str = DEFAULT_MODEL,
                 max_tokens: int = 1024, temperature: float = 0.7,
                 thinking: bool = False) -> dict:
    messages = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    return _mm_chat(messages, model, max_tokens, system, temperature, thinking)

# ── TTS via speech-2.8-hd ─────────────────────────────────────────────────────

def _mm_tts(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0,
            pitch: int = 0, sound_effect: str = "none",
            output_format: str = "mp3", sample_rate: int = 32000,
            bitrate: int = 128000) -> dict:
    if not MM_API_KEY:
        return {"error": "MINIMAX_API_KEY not configured"}

    body = {
        "model": "speech-2.8-hd",
        "text":  text[:10000],
        "stream": False,
        "voice_setting": {
            "voice_id": voice,
            "speed":    max(0.5, min(2.0, speed)),
            "vol":      1,
            "pitch":    max(-12, min(12, pitch)),
        },
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate":     bitrate,
            "format":      output_format,
            "channel":     1,
        },
        "output_format": "hex",
    }
    if sound_effect and sound_effect != "none":
        body["voice_modify"] = {
            "pitch": 0, "intensity": 0, "timbre": 0,
            "sound_effects": sound_effect,
        }
    # Pronunciation expansions for Australian context
    body["pronunciation_dict"] = {"tone": [
        "ABN/A B N",
        "FractalMesh/Fractal Mesh",
        "MCP/M C P",
        "DePIN/De Pin",
    ]}
    body["language_boost"] = "auto"

    payload = json.dumps(body).encode()
    req = urllib.request.Request(MM_TTS_URL, data=payload, method="POST",
        headers={"Authorization": f"Bearer {MM_API_KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result  = json.loads(r.read())
            latency = (time.time()-t0)*1000
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}", "detail": e.read().decode()[:300]}
    except Exception as e:
        return {"error": str(e)}

    # result is an array; first item is the complete audio
    chunks = result if isinstance(result, list) else [result]
    final_chunk = next((c for c in chunks if c.get("data",{}).get("status") == 2), chunks[0] if chunks else {})
    audio_hex   = final_chunk.get("data",{}).get("audio","")
    extra       = final_chunk.get("extra_info",{})

    if not audio_hex:
        _db_log_tts(len(text), voice, "", output_format, latency, "no_audio")
        return {"error": "no_audio_returned", "raw": chunks}

    # Save audio to disk
    audio_bytes = binascii.unhexlify(audio_hex)
    ts_str      = time.strftime("%Y%m%d_%H%M%S")
    filename    = f"tts_{ts_str}.{output_format}"
    audio_path  = AUDIO_DIR / filename
    audio_path.write_bytes(audio_bytes)

    _db_log_tts(len(text), voice, str(audio_path), output_format, latency, "ok")
    log.info("tts voice=%s len=%d audio=%s latency=%.0fms", voice, len(text), filename, latency)
    return {
        "status":      "ok",
        "audio_file":  str(audio_path),
        "audio_size":  len(audio_bytes),
        "duration_ms": extra.get("audio_length"),
        "word_count":  extra.get("word_count"),
        "voice":       voice,
        "format":      output_format,
        "latency_ms":  round(latency,1),
    }

# ── HTTP handler ──────────────────────────────────────────────────────────────

class MMHandler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _r(self,code,body):
        p=json.dumps(body).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def _serve_audio(self, path: str):
        try:
            audio = Path(path).read_bytes()
            self.send_response(200); self.send_header("Content-Type","audio/mpeg")
            self.send_header("Content-Length",str(len(audio))); self.end_headers()
            self.wfile.write(audio)
        except Exception as e:
            self._r(404,{"error":str(e)})

    def do_GET(self):
        import urllib.parse
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]
        if ep == "/health":
            self._r(200,{"status":"ok","api_key":bool(MM_API_KEY),"default_model":DEFAULT_MODEL,
                         "supported_models":SUPPORTED_MODELS,"voices":list(VOICES.keys())})
        elif ep == "/models":
            self._r(200,{"models":[{"id":m,"context_window":204800} for m in SUPPORTED_MODELS]})
        elif ep == "/voices":
            self._r(200,{"voices":VOICES,"sound_effects":SOUND_EFFECTS})
        elif ep == "/usage":
            try:
                c=sqlite3.connect(DB,timeout=5)
                llm = c.execute("SELECT model,COUNT(*),SUM(cost_tokens),AVG(latency_ms) FROM minimax_requests GROUP BY model").fetchall()
                tts = c.execute("SELECT COUNT(*),SUM(text_len),AVG(latency_ms) FROM minimax_tts WHERE status='ok'").fetchone()
                c.close()
                self._r(200,{"llm_usage":[{"model":r[0],"calls":r[1],"tokens":r[2],"avg_ms":round(r[3] or 0,1)} for r in llm],
                             "tts_usage":{"calls":tts[0],"chars":tts[1],"avg_ms":round(tts[2] or 0,1)} if tts else {}})
            except Exception as e: self._r(500,{"error":str(e)})
        elif ep == "/audio":
            fname = qs.get("file",[""])[0]
            if fname: self._serve_audio(str(AUDIO_DIR/fname))
            else: self._r(400,{"error":"file required"})
        else:
            self._r(404,{"error":"not_found"})

    def do_POST(self):
        try:
            n=int(self.headers.get("Content-Length",0)); d=json.loads(self.rfile.read(n))
            ep=self.path.split("?")[0]

            if ep in ("/generate","/chat","/messages"):
                # Support both simple prompt and full messages format
                if "messages" in d:
                    result = _mm_chat(
                        d["messages"], d.get("model",DEFAULT_MODEL),
                        d.get("max_tokens",1024), d.get("system",""),
                        d.get("temperature",0.7), d.get("thinking",False))
                else:
                    result = _mm_generate(
                        d.get("prompt",d.get("text","")),
                        d.get("system",""), d.get("model",DEFAULT_MODEL),
                        d.get("max_tokens",1024), d.get("temperature",0.7),
                        d.get("thinking",False))
                self._r(200 if "error" not in result else 502, result)

            elif ep == "/tts":
                text   = d.get("text","")
                if not text: self._r(400,{"error":"text required"}); return
                result = _mm_tts(
                    text, d.get("voice",DEFAULT_VOICE),
                    d.get("speed",1.0), d.get("pitch",0),
                    d.get("sound_effect","none"),
                    d.get("format","mp3"))
                self._r(200 if "error" not in result else 502, result)

            elif ep == "/speak":
                # Convenience: generate text then TTS it
                prompt = d.get("prompt","")
                if not prompt: self._r(400,{"error":"prompt required"}); return
                llm_r  = _mm_generate(prompt, d.get("system",""), d.get("model",DEFAULT_MODEL),
                                      d.get("max_tokens",512))
                if "error" in llm_r: self._r(502,llm_r); return
                tts_r  = _mm_tts(llm_r["content"], d.get("voice",DEFAULT_VOICE),
                                  d.get("speed",1.0), d.get("pitch",0),
                                  d.get("sound_effect","none"))
                self._r(200,{"llm":llm_r,"tts":tts_r})

            elif ep == "/reason":
                # Extended thinking mode
                result = _mm_generate(
                    d.get("prompt",""), d.get("system",""),
                    d.get("model","MiniMax-M2.7"),
                    d.get("max_tokens",4096), d.get("temperature",1.0),
                    thinking=True)
                self._r(200 if "error" not in result else 502, result)

            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: log.error("handler: %s",e); self._r(500,{"error":str(e)})

_running=True
def _shutdown(*_): global _running; _running=False
signal.signal(signal.SIGTERM,_shutdown); signal.signal(signal.SIGINT,_shutdown)

def main():
    _db_init(); server=HTTPServer(("0.0.0.0",PORT),MMHandler)
    log.info("MiniMax agent on port %d | model=%s | voice=%s | api_key=%s",
             PORT,DEFAULT_MODEL,DEFAULT_VOICE,bool(MM_API_KEY))
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__=="__main__": main()
