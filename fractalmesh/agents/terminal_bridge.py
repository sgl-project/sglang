#!/usr/bin/env python3
"""
FractalMesh Terminal Bridge v1.0
WebSocket server on :5062 — /terminal (exec) + /telemetry (log broadcast)
"""

import asyncio
import json
import os
import re
import sys
import glob
from pathlib import Path

try:
    import websockets
    from websockets.server import serve
except ImportError:
    os.system(f"{sys.executable} -m pip install websockets --quiet")
    import websockets
    from websockets.server import serve

PORT     = int(os.environ.get("TERMINAL_PORT", 5062))
ROOT     = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
LOG_DIR  = os.path.join(ROOT, "logs")
FM_LOGS  = str(Path.home() / ".fm_logs")

BLOCKED  = re.compile(
    r"rm\s+-rf\s+[/~]|mkfs|dd\s+if=|>\s*/dev/sd|shutdown|reboot|"
    r"passwd|sudo\s+rm|chmod\s+777\s+/|wget\s+[^\s]+\s*\|\s*sh|"
    r"curl\s+[^\s]+\s*\|\s*sh|eval\s*\(",
    re.IGNORECASE,
)

INTERP = {
    "bash":   ["/bin/bash", "-c"],
    "python": [sys.executable, "-c"],
    "node":   ["node", "-e"],
    "json":   None,
}

terminal_clients: set = set()
telemetry_clients: set = set()


async def run_command(cmd: str, lang: str, ws):
    if BLOCKED.search(cmd):
        await ws.send(json.dumps({"type": "stderr", "data": "[BLOCKED] Dangerous pattern detected."}))
        await ws.send(json.dumps({"type": "exit", "code": 1}))
        return

    if lang == "json":
        try:
            await ws.send(json.dumps({"type": "stdout", "data": json.dumps(json.loads(cmd), indent=2)}))
        except json.JSONDecodeError as e:
            await ws.send(json.dumps({"type": "stderr", "data": f"JSON error: {e}"}))
        await ws.send(json.dumps({"type": "exit", "code": 0}))
        return

    interp = INTERP.get(lang, INTERP["bash"])
    try:
        proc = await asyncio.create_subprocess_exec(
            *interp, cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.home()),
            limit=256 * 1024,
        )

        async def stream(pipe, msg_type):
            while True:
                try:
                    line = await asyncio.wait_for(pipe.readline(), timeout=30)
                except asyncio.TimeoutError:
                    await ws.send(json.dumps({"type": "stderr", "data": "[TIMEOUT] 30s limit exceeded."}))
                    proc.kill()
                    return
                if not line:
                    break
                try:
                    await ws.send(json.dumps({"type": msg_type, "data": line.decode(errors="replace").rstrip()}))
                except websockets.ConnectionClosed:
                    proc.kill()
                    return

        await asyncio.gather(stream(proc.stdout, "stdout"), stream(proc.stderr, "stderr"))
        await proc.wait()
        await ws.send(json.dumps({"type": "exit", "code": proc.returncode}))

    except FileNotFoundError:
        await ws.send(json.dumps({"type": "stderr", "data": f"[ERR] Interpreter not found for {lang}"}))
        await ws.send(json.dumps({"type": "exit", "code": 127}))
    except websockets.ConnectionClosed:
        pass


async def terminal_handler(ws):
    terminal_clients.add(ws)
    await ws.send(json.dumps({"type": "system", "data": f"[bridge] Connected · port={PORT}"}))
    try:
        async for message in ws:
            try:
                p    = json.loads(message)
                cmd  = str(p.get("cmd", "")).strip()
                lang = str(p.get("lang", "bash")).lower()
                if lang not in INTERP:
                    lang = "bash"
                if cmd:
                    await run_command(cmd, lang, ws)
            except (json.JSONDecodeError, KeyError):
                await ws.send(json.dumps({"type": "stderr", "data": "Invalid message format"}))
    except websockets.ConnectionClosed:
        pass
    finally:
        terminal_clients.discard(ws)


async def telemetry_handler(ws):
    telemetry_clients.add(ws)
    await ws.send(json.dumps({"agent": "bridge", "text": "Telemetry feed connected", "level": "info"}))
    try:
        await ws.wait_closed()
    finally:
        telemetry_clients.discard(ws)


async def log_broadcaster():
    positions = {}
    while True:
        files = []
        for pat in [os.path.join(LOG_DIR, "*-out.log"), os.path.join(FM_LOGS, "*.log")]:
            files.extend(glob.glob(pat))

        for fpath in files:
            agent = Path(fpath).stem.replace("-out", "").replace("_out", "")
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            if fpath not in positions:
                positions[fpath] = size
                continue
            if size < positions[fpath]:
                positions[fpath] = 0
            if size <= positions[fpath]:
                continue

            try:
                with open(fpath, "r", errors="replace") as f:
                    f.seek(positions[fpath])
                    data = f.read(4096)
                    positions[fpath] = f.tell()

                for line in data.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    level = ("err"  if any(k in line for k in ("ERR", "Error", "FAIL")) else
                             "warn" if any(k in line for k in ("WARN", "401",  "404" )) else
                             "info")
                    msg  = json.dumps({"agent": agent[:16], "text": line[:160], "level": level})
                    dead = set()
                    for ws in list(telemetry_clients):
                        try:
                            await ws.send(msg)
                        except websockets.ConnectionClosed:
                            dead.add(ws)
                    telemetry_clients -= dead
            except (OSError, IOError):
                pass

        await asyncio.sleep(1)


async def router(ws):
    path = getattr(getattr(ws, "request", None), "path", "/terminal") or "/terminal"
    if "/telemetry" in path:
        await telemetry_handler(ws)
    else:
        await terminal_handler(ws)


async def main():
    print(f"[terminal_bridge] ws://0.0.0.0:{PORT}  /terminal + /telemetry")
    asyncio.ensure_future(log_broadcaster())
    async with serve(router, "0.0.0.0", PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
