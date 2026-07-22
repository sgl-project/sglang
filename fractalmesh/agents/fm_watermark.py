"""
FractalMesh Watermark Agent v2.0.0
Applies visible + metadata watermarks to generated documents and images.
Supports Markdown (metadata injection) and plain text (footer stamp).
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import re
import time
import signal
import sqlite3
import hashlib
from datetime import datetime

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
REPO       = os.getenv("REPO_ROOT",        os.path.expanduser("~/sglang"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL   = int(os.getenv("WATERMARK_INTERVAL", "3600"))
DRY_RUN    = os.getenv("ENABLE_WATERMARK", "false").lower() != "true"

OPERATOR   = "Samuel James Hiotis"
ABN        = "56 628 117 363"
SITE       = "https://fractalmesh.net"
PHI        = 1.6180339887

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS watermark_log (
        id INTEGER PRIMARY KEY, file_path TEXT, method TEXT,
        fingerprint TEXT, wm_hash TEXT, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log(path: str, method: str, fp: str, wm_hash: str, status: str):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO watermark_log (file_path,method,fingerprint,wm_hash,status)
        VALUES (?,?,?,?,?)""", (path, method, fp, wm_hash, status))
    conn.commit(); conn.close()


def _wm_token(path: str) -> str:
    """φ-seeded watermark token for this file."""
    raw   = hashlib.sha256(f"{path}{PHI}{OPERATOR}".encode()).hexdigest()
    return f"FM-{raw[:8].upper()}"


def _watermark_markdown(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        content = f.read()

    token   = _wm_token(path)
    ts      = datetime.utcnow().strftime("%Y-%m-%d")
    footer  = (f"\n\n---\n"
               f"*Watermark: {token} | {OPERATOR} (ABN {ABN}) | {SITE} | {ts}*\n"
               f"*This document is the intellectual property of the operator above.*\n")

    if token in content:
        return "already_watermarked"

    # Also inject HTML comment metadata (invisible in rendered MD)
    meta = (f"<!-- FRACTAL-WM: token={token} author={OPERATOR} "
            f"abn={ABN} date={ts} phi={PHI} -->")

    stamped = meta + "\n\n" + content + footer
    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(stamped)
    return "watermarked"


def _watermark_text(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        content = f.read()

    token  = _wm_token(path)
    ts     = datetime.utcnow().strftime("%Y-%m-%d")
    footer = (f"\n\n{'='*60}\n"
              f"WATERMARK: {token}\n"
              f"Owner: {OPERATOR} (ABN {ABN})\n"
              f"Website: {SITE}\n"
              f"Date: {ts}\n"
              f"{'='*60}\n")

    if token in content:
        return "already_watermarked"

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content + footer)
    return "watermarked"


def _watermark_image_available() -> bool:
    try:
        import PIL  # noqa
        return True
    except ImportError:
        return False


def _watermark_image(path: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img   = Image.open(path).convert("RGBA")
        token = _wm_token(path)
        text  = f"© {OPERATOR} | {token}"
        draw  = ImageDraw.Draw(img)
        w, h  = img.size
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        pos  = (w - tw - 10, h - th - 10)
        draw.text(pos, text, fill=(255, 255, 255, 128), font=font)
        if not DRY_RUN:
            img.save(path)
        return "watermarked"
    except Exception as e:
        return f"img_err:{e}"


def run_cycle():
    ts   = datetime.utcnow().isoformat()
    dist = os.path.join(REPO, "fractalmesh", "dist")
    print(f"[fm-watermark] {ts} | dry={DRY_RUN}")

    total = 0
    if not os.path.exists(dist):
        print("   dist/ not found")
        return

    for fname in os.listdir(dist):
        path  = os.path.join(dist, fname)
        token = _wm_token(path)
        fp    = hashlib.sha256(fname.encode()).hexdigest()[:12]

        if fname.endswith(".md"):
            status = _watermark_markdown(path)
            method = "md_footer"
        elif fname.endswith(".txt"):
            status = _watermark_text(path)
            method = "txt_footer"
        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
            status = _watermark_image(path)
            method = "pil_overlay"
        else:
            continue

        _log(path, method, fp, token, status)
        print(f"   {fname:<40} [{method}] {status}")
        total += 1

    print(f"   Processed {total} files")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-watermark] Active | interval={INTERVAL}s | dry={DRY_RUN} | "
          f"PIL={'available' if _watermark_image_available() else 'not installed'}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-watermark] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-watermark] Stopped.")
