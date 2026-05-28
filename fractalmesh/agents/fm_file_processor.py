"""
FractalMesh OMEGA Titan — File Processing & Conversion Engine
Port: 7876  (FILE_PROCESSOR_PORT env var)
stdlib only: no third-party dependencies
"""

import base64
import csv
import gzip
import hashlib
import hmac
import html
import io
import json
import math
import mimetypes
import os
import re
import sqlite3
import struct
import tarfile
import textwrap
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Vault / env loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT = int(os.environ.get("FILE_PROCESSOR_PORT", 7876))
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")

DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS file_jobs (
            id              INTEGER PRIMARY KEY,
            job_id          TEXT UNIQUE,
            operation       TEXT,
            input_filename  TEXT,
            input_size      INTEGER,
            input_mime      TEXT,
            output_filename TEXT,
            output_size     INTEGER,
            status          TEXT DEFAULT 'pending',
            progress        INTEGER DEFAULT 0,
            result_data     TEXT,
            error           TEXT,
            created_at      REAL,
            started_at      REAL,
            finished_at     REAL
        );
        CREATE TABLE IF NOT EXISTS processed_files (
            id            INTEGER PRIMARY KEY,
            file_hash     TEXT UNIQUE,
            original_name TEXT,
            stored_name   TEXT,
            file_path     TEXT,
            mime_type     TEXT,
            size_bytes    INTEGER,
            metadata      TEXT,
            created_at    REAL
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Background purge thread
# ---------------------------------------------------------------------------

def _purge_old_jobs():
    while True:
        time.sleep(300)
        cutoff = time.time() - (7 * 86400)
        try:
            conn = get_conn()
            conn.execute(
                "DELETE FROM file_jobs WHERE status IN ('done','failed') AND finished_at < ?",
                (cutoff,)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Operation helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "as","by","from","up","about","into","through","during","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","that","this","these",
    "those","it","its","he","she","they","we","you","i","me","my","your","his",
    "her","our","their","what","which","who","whom","not","no","so","if","then",
    "than","too","very","just","more","also","there","here","when","where","how",
    "all","each","every","both","few","more","most","other","some","such","own",
    "same","only","after","before","between","while","because","although","since"
}


def _detect_delimiter(sample: str) -> str:
    counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
    return max(counts, key=counts.get)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


# --- Operation 1: csv_to_json ---

def op_csv_to_json(input_bytes: bytes, params: dict) -> dict:
    text = input_bytes.decode("utf-8", errors="replace")
    delimiter = _detect_delimiter(text[:4096])
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = [dict(row) for row in reader]
    return {
        "rows": rows,
        "row_count": len(rows),
        "columns": reader.fieldnames or [],
        "delimiter_detected": delimiter,
    }


# --- Operation 2: json_to_csv ---

def op_json_to_csv(input_bytes: bytes, params: dict) -> dict:
    data = json.loads(input_bytes.decode("utf-8", errors="replace"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array or object")
    flat_rows = [_flatten_dict(row) if isinstance(row, dict) else {"value": row} for row in data]
    all_keys = []
    seen = set()
    for row in flat_rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat_rows)
    csv_text = buf.getvalue()
    return {
        "csv": csv_text,
        "csv_base64": base64.b64encode(csv_text.encode()).decode(),
        "row_count": len(flat_rows),
        "columns": all_keys,
    }


# --- Operation 3: text_analyse ---

def op_text_analyse(input_bytes: bytes, params: dict) -> dict:
    text = input_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    words_raw = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w.strip("'") for w in words_raw if w.strip("'")]
    sentences = re.split(r"[.!?]+", text)
    sentence_count = sum(1 for s in sentences if s.strip())
    unique_words = set(words)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 1]
    freq: dict = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    top20 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
    reading_time_sec = (len(words) / 200) * 60 if words else 0
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
        "sentence_count": sentence_count,
        "unique_word_count": len(unique_words),
        "avg_word_length": round(avg_word_len, 2),
        "reading_time_seconds": round(reading_time_sec, 1),
        "reading_time_minutes": round(reading_time_sec / 60, 2),
        "top_20_words": [{"word": w, "count": c} for w, c in top20],
    }


# --- Operation 4: csv_analyse ---

def op_csv_analyse(input_bytes: bytes, params: dict) -> dict:
    text = input_bytes.decode("utf-8", errors="replace")
    delimiter = _detect_delimiter(text[:4096])
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = [dict(row) for row in reader]
    columns = reader.fieldnames or []
    if not rows:
        return {"columns": columns, "row_count": 0, "column_stats": {}}

    col_stats = {}
    for col in columns:
        values = [row.get(col, "") for row in rows]
        null_count = sum(1 for v in values if v is None or v == "")
        numeric_vals = []
        for v in values:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass

        if numeric_vals and len(numeric_vals) >= len(values) * 0.5:
            mean = sum(numeric_vals) / len(numeric_vals)
            variance = sum((x - mean) ** 2 for x in numeric_vals) / len(numeric_vals) if len(numeric_vals) > 1 else 0
            col_stats[col] = {
                "dtype": "numeric",
                "null_count": null_count,
                "min": min(numeric_vals),
                "max": max(numeric_vals),
                "mean": round(mean, 4),
                "std": round(math.sqrt(variance), 4),
                "count": len(numeric_vals),
            }
        else:
            freq: dict = {}
            for v in values:
                if v:
                    freq[v] = freq.get(v, 0) + 1
            top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            col_stats[col] = {
                "dtype": "text",
                "null_count": null_count,
                "unique_count": len(freq),
                "top_5_values": [{"value": v, "count": c} for v, c in top5],
            }

    return {
        "columns": columns,
        "row_count": len(rows),
        "column_stats": col_stats,
        "delimiter_detected": delimiter,
    }


# --- Operation 5: json_validate ---

def op_json_validate(input_bytes: bytes, params: dict) -> dict:
    text = input_bytes.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        valid = True
        error_info = None
    except json.JSONDecodeError as e:
        valid = False
        error_info = {"message": str(e), "line": e.lineno, "col": e.colno, "pos": e.pos}
        return {"valid": False, "error": error_info}

    def _count_structure(obj, depth=0):
        stats = {"arrays": 0, "objects": 0, "primitives": 0, "keys": 0, "max_depth": depth}
        if isinstance(obj, dict):
            stats["objects"] += 1
            stats["keys"] += len(obj)
            for v in obj.values():
                sub = _count_structure(v, depth + 1)
                stats["arrays"] += sub["arrays"]
                stats["objects"] += sub["objects"]
                stats["primitives"] += sub["primitives"]
                stats["keys"] += sub["keys"]
                stats["max_depth"] = max(stats["max_depth"], sub["max_depth"])
        elif isinstance(obj, list):
            stats["arrays"] += 1
            for item in obj:
                sub = _count_structure(item, depth + 1)
                stats["arrays"] += sub["arrays"]
                stats["objects"] += sub["objects"]
                stats["primitives"] += sub["primitives"]
                stats["keys"] += sub["keys"]
                stats["max_depth"] = max(stats["max_depth"], sub["max_depth"])
        else:
            stats["primitives"] += 1
        return stats

    s = _count_structure(data)
    return {
        "valid": True,
        "root_type": type(data).__name__,
        "total_keys": s["keys"],
        "max_nesting_depth": s["max_depth"],
        "array_count": s["arrays"],
        "object_count": s["objects"],
        "primitive_count": s["primitives"],
    }


# --- Operation 6: markdown_to_html ---

def op_markdown_to_html(input_bytes: bytes, params: dict) -> dict:
    md = input_bytes.decode("utf-8", errors="replace")
    lines = md.split("\n")
    html_parts = []
    in_code_block = False
    code_lang = ""
    code_lines = []
    in_list_ul = False
    in_list_ol = False
    in_blockquote = False

    def close_lists():
        nonlocal in_list_ul, in_list_ol
        if in_list_ul:
            html_parts.append("</ul>")
            in_list_ul = False
        if in_list_ol:
            html_parts.append("</ol>")
            in_list_ol = False

    def close_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            html_parts.append("</blockquote>")
            in_blockquote = False

    def inline(text: str) -> str:
        # Images before links
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        text = re.sub(r"_([^_]+)_", r"<em>\1</em>", text)
        return text

    for line in lines:
        # Code block fence
        if line.startswith("```"):
            if not in_code_block:
                close_lists()
                close_blockquote()
                in_code_block = True
                code_lang = line[3:].strip()
                code_lines = []
            else:
                in_code_block = False
                lang_attr = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                code_content = html.escape("\n".join(code_lines))
                html_parts.append(f"<pre><code{lang_attr}>{code_content}</code></pre>")
                code_lang = ""
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            close_lists()
            close_blockquote()
            level = len(heading_match.group(1))
            content = inline(html.escape(heading_match.group(2)))
            html_parts.append(f"<h{level}>{content}</h{level}>")
            continue

        # Blockquote
        if line.startswith("> "):
            close_lists()
            if not in_blockquote:
                html_parts.append("<blockquote>")
                in_blockquote = True
            content = inline(html.escape(line[2:]))
            html_parts.append(f"<p>{content}</p>")
            continue
        else:
            close_blockquote()

        # Unordered list
        ul_match = re.match(r"^[-*+]\s+(.*)", line)
        if ul_match:
            if in_list_ol:
                html_parts.append("</ol>")
                in_list_ol = False
            if not in_list_ul:
                html_parts.append("<ul>")
                in_list_ul = True
            content = inline(html.escape(ul_match.group(1)))
            html_parts.append(f"<li>{content}</li>")
            continue

        # Ordered list
        ol_match = re.match(r"^\d+\.\s+(.*)", line)
        if ol_match:
            if in_list_ul:
                html_parts.append("</ul>")
                in_list_ul = False
            if not in_list_ol:
                html_parts.append("<ol>")
                in_list_ol = True
            content = inline(html.escape(ol_match.group(1)))
            html_parts.append(f"<li>{content}</li>")
            continue

        close_lists()

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", line):
            html_parts.append("<hr>")
            continue

        # Empty line = paragraph break
        if line.strip() == "":
            html_parts.append("")
            continue

        # Regular paragraph
        content = inline(html.escape(line))
        html_parts.append(f"<p>{content}</p>")

    close_lists()
    close_blockquote()
    result_html = "\n".join(html_parts)
    return {
        "html": result_html,
        "html_base64": base64.b64encode(result_html.encode()).decode(),
    }


# --- Operation 7: html_to_text ---

def op_html_to_text(input_bytes: bytes, params: dict) -> dict:
    text = input_bytes.decode("utf-8", errors="replace")
    # Replace block-level tags with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(p|div|h[1-6]|li|tr|blockquote)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return {
        "text": text,
        "text_base64": base64.b64encode(text.encode()).decode(),
        "char_count": len(text),
        "line_count": len(text.splitlines()),
    }


# --- Operation 8: zip_list ---

def op_zip_list(input_bytes: bytes, params: dict) -> dict:
    buf = io.BytesIO(input_bytes)
    if not zipfile.is_zipfile(buf):
        raise ValueError("Input is not a valid ZIP file")
    buf.seek(0)
    entries = []
    with zipfile.ZipFile(buf, "r") as zf:
        for info in zf.infolist():
            entries.append({
                "filename": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "is_dir": info.is_dir(),
                "compress_type": info.compress_type,
                "date_time": list(info.date_time),
            })
    return {"entries": entries, "entry_count": len(entries)}


# --- Operation 9: zip_extract_file ---

def op_zip_extract_file(input_bytes: bytes, params: dict) -> dict:
    target = params.get("filename") or params.get("target_filename")
    if not target:
        raise ValueError("params.filename is required for zip_extract_file")
    buf = io.BytesIO(input_bytes)
    with zipfile.ZipFile(buf, "r") as zf:
        names = zf.namelist()
        if target not in names:
            raise ValueError(f"File '{target}' not found in ZIP. Available: {names[:20]}")
        data = zf.read(target)
    mime, _ = mimetypes.guess_type(target)
    return {
        "filename": target,
        "content_base64": base64.b64encode(data).decode(),
        "size": len(data),
        "mime_type": mime or "application/octet-stream",
    }


# --- Operation 10: gzip_compress ---

def op_gzip_compress(input_bytes: bytes, params: dict) -> dict:
    level = int(params.get("level", 6))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as gz:
        gz.write(input_bytes)
    compressed = buf.getvalue()
    ratio = len(input_bytes) / len(compressed) if compressed else 0
    return {
        "compressed_base64": base64.b64encode(compressed).decode(),
        "original_size": len(input_bytes),
        "compressed_size": len(compressed),
        "compression_ratio": round(ratio, 3),
        "space_saved_pct": round((1 - len(compressed) / max(len(input_bytes), 1)) * 100, 2),
    }


# --- Operation 11: gzip_decompress ---

def op_gzip_decompress(input_bytes: bytes, params: dict) -> dict:
    buf = io.BytesIO(input_bytes)
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        data = gz.read()
    return {
        "content_base64": base64.b64encode(data).decode(),
        "decompressed_size": len(data),
        "compressed_size": len(input_bytes),
    }


# --- Operation 12: base64_encode ---

def op_base64_encode(input_bytes: bytes, params: dict) -> dict:
    encoded = base64.b64encode(input_bytes).decode()
    return {
        "encoded": encoded,
        "original_size": len(input_bytes),
        "encoded_size": len(encoded),
    }


# --- Operation 13: base64_decode ---

def op_base64_decode(input_bytes: bytes, params: dict) -> dict:
    # input_bytes here is the raw base64 string bytes (not double-encoded)
    decoded = base64.b64decode(input_bytes)
    return {
        "content_base64": base64.b64encode(decoded).decode(),
        "decoded_size": len(decoded),
    }


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

OPERATIONS = {
    "csv_to_json": {
        "fn": op_csv_to_json,
        "description": "Parse CSV (base64-encoded) to JSON array with automatic delimiter detection",
        "required_params": [],
        "optional_params": [],
    },
    "json_to_csv": {
        "fn": op_json_to_csv,
        "description": "Flatten JSON array to CSV; nested objects become dot-path columns",
        "required_params": [],
        "optional_params": [],
    },
    "text_analyse": {
        "fn": op_text_analyse,
        "description": "Word/char/line counts, top-20 words (excluding stopwords), reading time, sentence count",
        "required_params": [],
        "optional_params": [],
    },
    "csv_analyse": {
        "fn": op_csv_analyse,
        "description": "Column stats: null counts, numeric min/max/mean/std, top-5 text values, dtype inference",
        "required_params": [],
        "optional_params": [],
    },
    "json_validate": {
        "fn": op_json_validate,
        "description": "Validate JSON, report parse errors with line/col, count keys/nesting/types",
        "required_params": [],
        "optional_params": [],
    },
    "markdown_to_html": {
        "fn": op_markdown_to_html,
        "description": "Convert Markdown to HTML: headings, bold, italic, code, blockquotes, lists, links, images",
        "required_params": [],
        "optional_params": [],
    },
    "html_to_text": {
        "fn": op_html_to_text,
        "description": "Strip HTML tags, decode entities, preserve paragraph breaks",
        "required_params": [],
        "optional_params": [],
    },
    "zip_list": {
        "fn": op_zip_list,
        "description": "List ZIP archive contents: filename, size, compressed_size, is_dir",
        "required_params": [],
        "optional_params": [],
    },
    "zip_extract_file": {
        "fn": op_zip_extract_file,
        "description": "Extract a single file from a ZIP archive, return base64 content",
        "required_params": ["filename"],
        "optional_params": [],
    },
    "gzip_compress": {
        "fn": op_gzip_compress,
        "description": "Gzip-compress input, return compressed base64 and compression ratio",
        "required_params": [],
        "optional_params": ["level (1-9, default 6)"],
    },
    "gzip_decompress": {
        "fn": op_gzip_decompress,
        "description": "Decompress gzip base64 input, return plaintext base64",
        "required_params": [],
        "optional_params": [],
    },
    "base64_encode": {
        "fn": op_base64_encode,
        "description": "Base64-encode arbitrary bytes",
        "required_params": [],
        "optional_params": [],
    },
    "base64_decode": {
        "fn": op_base64_decode,
        "description": "Base64-decode a base64 string",
        "required_params": [],
        "optional_params": [],
    },
}

LARGE_FILE_THRESHOLD = 1 * 1024 * 1024  # 1 MB

# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def _execute_operation(job_id: str, operation: str, input_bytes: bytes, params: dict):
    conn = get_conn()
    conn.execute(
        "UPDATE file_jobs SET status='running', started_at=?, progress=10 WHERE job_id=?",
        (time.time(), job_id)
    )
    conn.commit()
    try:
        fn = OPERATIONS[operation]["fn"]
        result = fn(input_bytes, params)
        result_json = json.dumps(result)
        conn.execute(
            """UPDATE file_jobs
               SET status='done', progress=100, result_data=?,
                   output_size=?, finished_at=?
               WHERE job_id=?""",
            (result_json, len(result_json), time.time(), job_id)
        )
        conn.commit()
    except Exception as exc:
        conn.execute(
            "UPDATE file_jobs SET status='failed', error=?, finished_at=? WHERE job_id=?",
            (str(exc), time.time(), job_id)
        )
        conn.commit()
    finally:
        conn.close()


def submit_job(operation: str, input_bytes: bytes, filename: str, params: dict) -> dict:
    job_id = str(uuid.uuid4())
    mime, _ = mimetypes.guess_type(filename) if filename else (None, None)
    mime = mime or "application/octet-stream"

    conn = get_conn()
    conn.execute(
        """INSERT INTO file_jobs
           (job_id, operation, input_filename, input_size, input_mime, status, progress, created_at)
           VALUES (?,?,?,?,?,'pending',0,?)""",
        (job_id, operation, filename or "", len(input_bytes), mime, time.time())
    )
    conn.commit()
    conn.close()

    if len(input_bytes) < LARGE_FILE_THRESHOLD:
        # Synchronous execution
        _execute_operation(job_id, operation, input_bytes, params)
        conn2 = get_conn()
        row = conn2.execute(
            "SELECT * FROM file_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        conn2.close()
        result_data = json.loads(row["result_data"]) if row["result_data"] else None
        return {
            "job_id": job_id,
            "status": row["status"],
            "result": result_data,
            "error": row["error"],
        }
    else:
        t = threading.Thread(
            target=_execute_operation,
            args=(job_id, operation, input_bytes, params),
            daemon=True
        )
        t.start()
        return {"job_id": job_id, "status": "pending"}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

def _json_response(handler, code: int, data):
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler) -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(length) if length else b""


def _require_admin(handler) -> bool:
    if not ADMIN_SECRET:
        return True
    auth = handler.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    if not hmac.compare_digest(token, ADMIN_SECRET):
        _json_response(handler, 401, {"error": "Unauthorized"})
        return False
    return True


class FileProcessorHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # Suppress default access log

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._handle_health()
        elif path == "/jobs":
            self._handle_list_jobs(qs)
        elif path.startswith("/jobs/"):
            job_id = path[len("/jobs/"):]
            self._handle_get_job(job_id)
        elif path == "/operations":
            self._handle_operations()
        elif path == "/files":
            self._handle_list_files()
        elif path.startswith("/files/"):
            file_hash = path[len("/files/"):]
            self._handle_get_file(file_hash)
        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/process":
            self._handle_process()
        elif path == "/process/batch":
            self._handle_batch()
        elif path == "/files":
            self._handle_store_file()
        else:
            _json_response(self, 404, {"error": "Not found"})

    # ---- GET handlers ----

    def _handle_health(self):
        conn = get_conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM file_jobs GROUP BY status"
        ).fetchall()
        conn.close()
        job_counts = {r["status"]: r["cnt"] for r in rows}
        _json_response(self, 200, {
            "service": "fm_file_processor",
            "status": "ok",
            "port": PORT,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "job_counts": job_counts,
            "supported_operations": list(OPERATIONS.keys()),
            "operation_count": len(OPERATIONS),
        })

    def _handle_list_jobs(self, qs: dict):
        status = qs.get("status", [None])[0]
        operation = qs.get("operation", [None])[0]
        limit = int((qs.get("limit", ["50"])[0]) or 50)
        limit = min(limit, 500)

        query = "SELECT * FROM file_jobs WHERE 1=1"
        args = []
        if status:
            query += " AND status=?"
            args.append(status)
        if operation:
            query += " AND operation=?"
            args.append(operation)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)

        conn = get_conn()
        rows = conn.execute(query, args).fetchall()
        conn.close()
        jobs = [dict(r) for r in rows]
        _json_response(self, 200, {"jobs": jobs, "count": len(jobs)})

    def _handle_get_job(self, job_id: str):
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM file_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        conn.close()
        if not row:
            _json_response(self, 404, {"error": "Job not found"})
            return
        job = dict(row)
        if job.get("result_data"):
            try:
                job["result_data"] = json.loads(job["result_data"])
            except Exception:
                pass
        _json_response(self, 200, job)

    def _handle_operations(self):
        ops = []
        for name, info in OPERATIONS.items():
            ops.append({
                "operation": name,
                "description": info["description"],
                "required_params": info["required_params"],
                "optional_params": info["optional_params"],
            })
        _json_response(self, 200, {"operations": ops, "count": len(ops)})

    def _handle_list_files(self):
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, file_hash, original_name, stored_name, mime_type, size_bytes, created_at FROM processed_files ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"files": [dict(r) for r in rows], "count": len(rows)})

    def _handle_get_file(self, file_hash: str):
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM processed_files WHERE file_hash=?", (file_hash,)
        ).fetchone()
        conn.close()
        if not row:
            _json_response(self, 404, {"error": "File not found"})
            return
        record = dict(row)
        try:
            file_path = Path(record["file_path"])
            if file_path.exists():
                content = file_path.read_bytes()
                record["content_base64"] = base64.b64encode(content).decode()
            else:
                record["content_base64"] = None
                record["warning"] = "File content not available on disk"
        except Exception as e:
            record["content_base64"] = None
            record["warning"] = str(e)
        _json_response(self, 200, record)

    # ---- POST handlers ----

    def _handle_process(self):
        body = _read_body(self)
        try:
            req = json.loads(body)
        except Exception:
            _json_response(self, 400, {"error": "Invalid JSON body"})
            return

        operation = req.get("operation", "").strip()
        if not operation:
            _json_response(self, 400, {"error": "operation is required"})
            return
        if operation not in OPERATIONS:
            _json_response(self, 400, {"error": f"Unknown operation '{operation}'. Valid: {list(OPERATIONS.keys())}"})
            return

        input_data_b64 = req.get("input_data", "")
        filename = req.get("filename", "input.bin")
        params = req.get("params", {}) or {}

        try:
            input_bytes = base64.b64decode(input_data_b64) if input_data_b64 else b""
        except Exception:
            _json_response(self, 400, {"error": "input_data must be valid base64"})
            return

        try:
            result = submit_job(operation, input_bytes, filename, params)
            _json_response(self, 200, result)
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _handle_batch(self):
        body = _read_body(self)
        try:
            req = json.loads(body)
        except Exception:
            _json_response(self, 400, {"error": "Invalid JSON body"})
            return

        operations_list = req.get("operations", [])
        if not isinstance(operations_list, list) or not operations_list:
            _json_response(self, 400, {"error": "operations must be a non-empty array"})
            return

        results = [None] * len(operations_list)
        errors = [None] * len(operations_list)
        lock = threading.Lock()

        def run_one(idx, op_req):
            op = op_req.get("operation", "")
            input_data_b64 = op_req.get("input_data", "")
            filename = op_req.get("filename", "input.bin")
            params = op_req.get("params", {}) or {}
            try:
                if op not in OPERATIONS:
                    raise ValueError(f"Unknown operation '{op}'")
                input_bytes = base64.b64decode(input_data_b64) if input_data_b64 else b""
                res = submit_job(op, input_bytes, filename, params)
                with lock:
                    results[idx] = res
            except Exception as exc:
                with lock:
                    errors[idx] = str(exc)
                    results[idx] = {"error": str(exc), "operation": op}

        threads = []
        for i, op_req in enumerate(operations_list):
            t = threading.Thread(target=run_one, args=(i, op_req), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=120)

        _json_response(self, 200, {"results": results, "count": len(results)})

    def _handle_store_file(self):
        body = _read_body(self)
        try:
            req = json.loads(body)
        except Exception:
            _json_response(self, 400, {"error": "Invalid JSON body"})
            return

        filename = req.get("filename", "upload.bin")
        content_b64 = req.get("content_base64", "")
        mime_type = req.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"

        try:
            content = base64.b64decode(content_b64)
        except Exception:
            _json_response(self, 400, {"error": "content_base64 must be valid base64"})
            return

        file_hash = hashlib.sha256(content).hexdigest()

        # Check if already stored
        conn = get_conn()
        existing = conn.execute(
            "SELECT * FROM processed_files WHERE file_hash=?", (file_hash,)
        ).fetchone()
        if existing:
            conn.close()
            _json_response(self, 200, {
                "file_hash": file_hash,
                "stored_name": existing["stored_name"],
                "already_existed": True,
            })
            return

        # Store file on disk
        store_dir = Path.home() / "fmsaas" / "files"
        store_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix or ""
        stored_name = f"{file_hash[:16]}{ext}"
        file_path = store_dir / stored_name
        file_path.write_bytes(content)

        metadata = json.dumps({
            "original_name": filename,
            "mime_type": mime_type,
            "size": len(content),
        })

        conn.execute(
            """INSERT OR IGNORE INTO processed_files
               (file_hash, original_name, stored_name, file_path, mime_type, size_bytes, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (file_hash, filename, stored_name, str(file_path), mime_type, len(content), metadata, time.time())
        )
        conn.commit()
        conn.close()

        _json_response(self, 201, {
            "file_hash": file_hash,
            "stored_name": stored_name,
            "size_bytes": len(content),
            "mime_type": mime_type,
            "already_existed": False,
        })


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def main():
    init_db()

    purge_thread = threading.Thread(target=_purge_old_jobs, daemon=True)
    purge_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), FileProcessorHandler)
    print(f"[fm_file_processor] Listening on port {PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
