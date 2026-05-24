"""
FractalMesh OMEGA Titan — ElevenLabs TTS Agent
Port: 7815
"""

import os
import json
import sqlite3
import logging
import signal
import time
import urllib.request
import urllib.error
import urllib.parse
import base64
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    with open(_VAULT) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
BASE_DIR    = Path.home() / "fmsaas"
DB_PATH     = BASE_DIR / "database" / "sovereign.db"
LOG_PATH    = BASE_DIR / "logs" / "fm_elevenlabs.log"
AUDIO_DIR   = BASE_DIR / "audio"

for _d in (DB_PATH.parent, LOG_PATH.parent, AUDIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PORT        = int(os.environ.setdefault("ELEVENLABS_PORT", "7815"))
API_KEY     = os.getenv("ELEVENLABS_API_KEY", "")
BASE_URL    = "https://api.elevenlabs.io/v1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_elevenlabs")

# ---------------------------------------------------------------------------
# SQLite — WAL mode
# ---------------------------------------------------------------------------
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elevenlabs_generations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            voice_id  TEXT,
            model_id  TEXT,
            text_len  INTEGER,
            file_path TEXT,
            ts        TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


_db = _get_db()

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _el_get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"xi-api-key": API_KEY})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _el_post_json(path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _el_post_tts_binary(path: str, payload: dict) -> bytes:
    """Return raw audio bytes from ElevenLabs TTS endpoint."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "xi-api-key": API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _multipart_encode(fields: dict, files: list) -> tuple[bytes, str]:
    """
    Build a multipart/form-data body.
    fields: {name: value}
    files:  [{name, filename, content_type, data (bytes)}]
    Returns (body_bytes, content_type_header_value).
    """
    boundary = uuid.uuid4().hex
    parts = []
    crlf = b"\r\n"
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}'.encode()
        )
    for f in files:
        header = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="{f["name"]}"; filename="{f["filename"]}"\r\n'
            f'Content-Type: {f["content_type"]}\r\n\r\n'
        ).encode()
        parts.append(header + f["data"])
    body = crlf.join(parts) + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _save_audio(audio_bytes: bytes) -> Path:
    ts = int(time.time() * 1000)
    path = AUDIO_DIR / f"{ts}.mp3"
    with open(path, "wb") as fh:
        fh.write(audio_bytes)
    return path


def _db_log_generation(voice_id: str, model_id: str, text_len: int, file_path: str):
    _db.execute(
        "INSERT INTO elevenlabs_generations (voice_id, model_id, text_len, file_path) VALUES (?,?,?,?)",
        (voice_id, model_id, text_len, file_path),
    )
    _db.commit()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_running = True


def _handle_signal(signum, frame):
    global _running
    log.info("Received signal %s — shutting down.", signum)
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
def _json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode())


def _send_json(handler: BaseHTTPRequestHandler, code: int, payload: dict):
    body = json.dumps(payload, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class ElevenLabsHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default stderr noise
        log.debug("HTTP %s", fmt % args)

    # ---- routing ----

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/health":
                self._health()
            elif path == "/voices":
                self._voices()
            elif path == "/models":
                self._models()
            elif path == "/history":
                self._history()
            elif path == "/quota":
                self._quota()
            else:
                _send_json(self, 404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            _send_json(self, exc.code, {"error": str(exc)})
        except Exception as exc:
            log.exception("Unhandled error in GET %s", path)
            _send_json(self, 500, {"error": str(exc)})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/tts":
                self._tts()
            elif path == "/tts/stream":
                self._tts_stream()
            elif path == "/clone":
                self._clone()
            else:
                _send_json(self, 404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            _send_json(self, exc.code, {"error": str(exc)})
        except Exception as exc:
            log.exception("Unhandled error in POST %s", path)
            _send_json(self, 500, {"error": str(exc)})

    # ---- handlers ----

    def _health(self):
        sub = _el_get("/user/subscription")
        _send_json(self, 200, {
            "status": "ok",
            "agent": "fm_elevenlabs",
            "port": PORT,
            "subscription": sub,
        })

    def _voices(self):
        data = _el_get("/voices")
        _send_json(self, 200, data)

    def _models(self):
        data = _el_get("/models")
        _send_json(self, 200, data)

    def _history(self):
        data = _el_get("/history")
        _send_json(self, 200, data)

    def _quota(self):
        sub = _el_get("/user/subscription")
        _send_json(self, 200, {
            "character_count": sub.get("character_count"),
            "character_limit": sub.get("character_limit"),
            "next_reset": sub.get("next_character_count_reset_unix"),
            "subscription": sub,
        })

    def _tts(self):
        body = _json_body(self)
        text = body.get("text", "")
        voice_id = body.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # default: Rachel
        model_id = body.get("model_id", "eleven_multilingual_v2")
        voice_settings = body.get("voice_settings", {"stability": 0.5, "similarity_boost": 0.75})

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        audio_bytes = _el_post_tts_binary(f"/text-to-speech/{voice_id}", payload)
        saved = _save_audio(audio_bytes)
        _db_log_generation(voice_id, model_id, len(text), str(saved))
        log.info("TTS generated: %s (%d bytes)", saved.name, len(audio_bytes))
        _send_json(self, 200, {
            "file": str(saved),
            "size_bytes": len(audio_bytes),
            "voice_id": voice_id,
            "text_len": len(text),
        })

    def _tts_stream(self):
        """Stream TTS — fetches full response, streams to client, also saves to disk."""
        body = _json_body(self)
        text = body.get("text", "")
        voice_id = body.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        model_id = body.get("model_id", "eleven_multilingual_v2")
        voice_settings = body.get("voice_settings", {"stability": 0.5, "similarity_boost": 0.75})

        url = f"{BASE_URL}/text-to-speech/{voice_id}/stream"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "xi-api-key": API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        ts = int(time.time() * 1000)
        saved = AUDIO_DIR / f"{ts}.mp3"
        chunks = []

        with urllib.request.urlopen(req, timeout=120) as resp:
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                hex_len = format(len(chunk), "x").encode()
                self.wfile.write(hex_len + b"\r\n" + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        audio_bytes = b"".join(chunks)
        with open(saved, "wb") as fh:
            fh.write(audio_bytes)
        _db_log_generation(voice_id, model_id, len(text), str(saved))
        log.info("TTS stream saved: %s (%d bytes)", saved.name, len(audio_bytes))

    def _clone(self):
        body = _json_body(self)
        name = body.get("name", "cloned_voice")
        # Expect base64-encoded audio sample
        b64_sample = body.get("audio_base64", "")
        if not b64_sample:
            _send_json(self, 400, {"error": "audio_base64 required"})
            return
        audio_data = base64.b64decode(b64_sample)

        fields = {"name": name}
        files = [{
            "name": "files",
            "filename": "sample.mp3",
            "content_type": "audio/mpeg",
            "data": audio_data,
        }]
        body_bytes, ct = _multipart_encode(fields, files)

        url = f"{BASE_URL}/voices/add"
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"xi-api-key": API_KEY, "Content-Type": ct},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())

        log.info("Voice cloned: %s", result)
        _send_json(self, 200, result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    server = HTTPServer(("0.0.0.0", PORT), ElevenLabsHandler)
    log.info("fm_elevenlabs listening on port %d", PORT)
    while _running:
        server.handle_request()
    log.info("fm_elevenlabs stopped.")


if __name__ == "__main__":
    main()
