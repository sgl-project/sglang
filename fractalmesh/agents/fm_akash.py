#!/usr/bin/env python3
"""
fm_akash.py — Akash Network Compute Leasing Agent (Port 7841)
Manages Akash decentralised cloud deployments via SDL templates and CLI.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import signal
import sqlite3
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT               = int(os.getenv("AKASH_PORT", "7841"))
AKASH_NODE         = os.getenv("AKASH_NODE", "https://rpc.akash.network:443")
AKASH_CHAIN_ID     = os.getenv("AKASH_CHAIN_ID", "akashnet-2")
AKASH_WALLET       = os.getenv("AKASH_WALLET_ADDRESS", "")
AKASH_PASSPHRASE   = os.getenv("AKASH_KEYRING_PASSPHRASE", "")
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY", "")
AKT_PRICE_USD      = float(os.getenv("AKT_PRICE_USD", "2.50"))

ROOT               = Path(os.path.expanduser("~/fmsaas"))
DB_PATH            = ROOT / "database" / "sovereign.db"
LOG_PATH           = ROOT / "logs" / "akash.log"
SDL_CPU_PATH       = ROOT / "akash_cpu.yaml"
SDL_GPU_PATH       = ROOT / "akash_gpu.yaml"

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AKASH] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("akash")

# ── database ──────────────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    conn = _get_db()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS akash_deployments (
                id               INTEGER PRIMARY KEY,
                dseq             TEXT,
                gseq             TEXT,
                oseq             TEXT,
                provider         TEXT,
                sdl_file         TEXT,
                status           TEXT,
                monthly_cost_uakt REAL,
                created_at       REAL,
                closed_at        REAL
            );
            CREATE TABLE IF NOT EXISTS akash_bids (
                id              INTEGER PRIMARY KEY,
                deployment_id   INTEGER,
                provider        TEXT,
                price_uakt      REAL,
                created_at      REAL
            );
            CREATE TABLE IF NOT EXISTS akash_sdls (
                id          INTEGER PRIMARY KEY,
                name        TEXT UNIQUE,
                content     TEXT,
                description TEXT,
                cpu         INTEGER,
                memory_mb   INTEGER,
                storage_gb  INTEGER,
                gpu         INTEGER,
                created_at  REAL
            );
        """)
    conn.close()

# ── SDL helpers ───────────────────────────────────────────────────────────────
def _generate_sdl(image: str, command: str, port: int,
                  cpu: int, memory_mb: int, storage_gb: int, gpu: int) -> str:
    """Build a minimal Akash SDL YAML string from resource parameters."""
    resources_block = (
        f"          cpu:\n            units: {cpu}.0\n"
        f"          memory:\n            size: {memory_mb}Mi\n"
        f"          storage:\n            size: {storage_gb}Gi"
    )
    if gpu > 0:
        resources_block += (
            f"\n          gpu:\n            units: {gpu}\n"
            f"            attributes:\n              vendor:\n                nvidia:"
        )
    sdl = (
        'version: "2.0"\n'
        "services:\n"
        "  app:\n"
        f"    image: {image}\n"
        f"    command:\n"
        f"      - {command}\n"
        "    expose:\n"
        f"      - port: {port}\n"
        "        as: 80\n"
        "        to:\n"
        "          - global: true\n"
        "profiles:\n"
        "  compute:\n"
        "    app:\n"
        "      resources:\n"
        f"{resources_block}\n"
        "  placement:\n"
        "    dcloud:\n"
        "      pricing:\n"
        "        app:\n"
        "          denom: uakt\n"
        "          amount: 1000\n"
        "deployment:\n"
        "  app:\n"
        "    dcloud:\n"
        "      profile: app\n"
        "      count: 1\n"
    )
    return sdl

def _write_sdl_file(name: str, content: str) -> Path:
    """Write SDL content to ~/fmsaas/{name}.yaml and return the path."""
    path = ROOT / f"{name}.yaml"
    path.write_text(content)
    return path

def _validate_sdl(content: str) -> tuple:
    """Parse YAML (or JSON fallback), verify required sections. Returns (valid, errors)."""
    errors = []
    parsed = None

    if _YAML_AVAILABLE:
        try:
            parsed = yaml.safe_load(content)
        except Exception as exc:
            errors.append(f"YAML parse error: {exc}")
    else:
        try:
            parsed = json.loads(content)
        except Exception:
            errors.append("YAML unavailable and content is not valid JSON")

    if parsed is None and not errors:
        errors.append("Empty or null SDL document")

    if parsed is not None:
        required = ["version", "services", "profiles", "deployment"]
        for section in required:
            if section not in parsed:
                errors.append(f"Missing required section: '{section}'")

    valid = len(errors) == 0
    return valid, errors, parsed

# ── akash CLI helper ──────────────────────────────────────────────────────────
def _run_akash(args: list, timeout: int = 30) -> tuple:
    """Run akash CLI with given args. Returns (stdout, stderr, returncode)."""
    cmd = ["akash"] + args
    env = os.environ.copy()
    if AKASH_PASSPHRASE:
        env["AKASH_KEYRING_PASSPHRASE"] = AKASH_PASSPHRASE
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "akash CLI not found", 127
    except subprocess.TimeoutExpired:
        return "", "akash CLI timed out", 1
    except Exception as exc:
        return "", str(exc), 1

def _akash_available() -> bool:
    result = subprocess.run(["which", "akash"], capture_output=True, text=True)
    return result.returncode == 0

# ── default SDL seeds ─────────────────────────────────────────────────────────
_DEFAULT_CPU_SDL = _generate_sdl(
    image="python:3.11-slim",
    command="python3 /app/server.py",
    port=8080,
    cpu=2,
    memory_mb=512,
    storage_gb=5,
    gpu=0,
)

_DEFAULT_GPU_SDL = _generate_sdl(
    image="nvcr.io/nvidia/pytorch:23.10-py3",
    command="python3 /app/inference.py",
    port=8080,
    cpu=4,
    memory_mb=8192,
    storage_gb=50,
    gpu=1,
)

def _seed_sdls():
    conn = _get_db()
    now = time.time()
    seeds = [
        {
            "name": "akash-cpu-default",
            "content": _DEFAULT_CPU_SDL,
            "description": "FractalMesh light CPU Python workload",
            "cpu": 2, "memory_mb": 512, "storage_gb": 5, "gpu": 0,
            "path": SDL_CPU_PATH,
        },
        {
            "name": "akash-gpu-default",
            "content": _DEFAULT_GPU_SDL,
            "description": "FractalMesh GPU AI inference workload (A100)",
            "cpu": 4, "memory_mb": 8192, "storage_gb": 50, "gpu": 1,
            "path": SDL_GPU_PATH,
        },
    ]
    for s in seeds:
        try:
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO akash_sdls "
                    "(name, content, description, cpu, memory_mb, storage_gb, gpu, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (s["name"], s["content"], s["description"],
                     s["cpu"], s["memory_mb"], s["storage_gb"], s["gpu"], now),
                )
            s["path"].write_text(s["content"])
            log.info("Seeded SDL: %s -> %s", s["name"], s["path"])
        except Exception as exc:
            log.warning("SDL seed skipped (%s): %s", s["name"], exc)
    conn.close()

# ── cost estimation ───────────────────────────────────────────────────────────
def _estimate_cost(cpu: int, memory_mb: int, storage_gb: int,
                   gpu: int, duration_hours: float = 720.0) -> dict:
    months = duration_hours / 720.0
    cpu_usd      = cpu * 2.0 * months
    memory_usd   = (memory_mb / 1024) * 1.0 * months
    storage_usd  = storage_gb * 0.02 * months
    gpu_usd      = gpu * 200.0 * months
    total_usd    = cpu_usd + memory_usd + storage_usd + gpu_usd
    uakt_per_akt = 1_000_000
    total_uakt   = int((total_usd / AKT_PRICE_USD) * uakt_per_akt)
    return {
        "cpu_usd":      round(cpu_usd, 4),
        "memory_usd":   round(memory_usd, 4),
        "storage_usd":  round(storage_usd, 4),
        "gpu_usd":      round(gpu_usd, 4),
        "total_usd":    round(total_usd, 4),
        "total_uakt":   total_uakt,
        "akt_price_usd": AKT_PRICE_USD,
    }

# ── HTTP handler ──────────────────────────────────────────────────────────────
class AkashHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    def _send(self, code: int, body: dict):
        data = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, code: int, text: str):
        data = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _path_parts(self) -> list:
        return [p for p in self.path.split("?")[0].split("/") if p]

    # ── routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        parts = self._path_parts()

        if not parts or parts == ["health"]:
            return self._send(200, {
                "status": "ok",
                "service": "fm-akash",
                "port": PORT,
            })

        if parts == ["sdl", "list"]:
            return self._handle_sdl_list()

        if len(parts) == 2 and parts[0] == "sdl":
            return self._handle_sdl_get(parts[1])

        if parts == ["deployments"]:
            return self._handle_deployments_list()

        if len(parts) == 2 and parts[0] == "deployments":
            return self._handle_deployment_get(parts[1])

        if len(parts) == 3 and parts[0] == "deployments" and parts[2] == "bids":
            return self._handle_deployment_bids(parts[1])

        if parts == ["analytics"]:
            return self._handle_analytics()

        self._send(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        parts = self._path_parts()
        body = self._read_body()

        if parts == ["sdl", "create"]:
            return self._handle_sdl_create(body)

        if parts == ["sdl", "validate"]:
            return self._handle_sdl_validate(body)

        if parts == ["deploy"]:
            return self._handle_deploy(body)

        if parts == ["cost", "estimate"]:
            return self._handle_cost_estimate(body)

        if len(parts) == 3 and parts[0] == "deployments" and parts[2] == "close":
            return self._handle_deployment_close(parts[1])

        if len(parts) == 3 and parts[0] == "deployments" and parts[2] == "accept_bid":
            return self._handle_accept_bid(parts[1], body)

        self._send(404, {"error": "not found", "path": self.path})

    # ── SDL handlers ───────────────────────────────────────────────────────────
    def _handle_sdl_create(self, body: dict):
        required = ["name", "cpu", "memory_mb", "storage_gb"]
        for field in required:
            if field not in body:
                return self._send(400, {"error": f"Missing field: {field}"})

        name        = str(body["name"])
        description = body.get("description", "")
        cpu         = int(body.get("cpu", 1))
        memory_mb   = int(body.get("memory_mb", 512))
        storage_gb  = int(body.get("storage_gb", 5))
        gpu         = int(body.get("gpu", 0))
        image       = body.get("image", "python:3.11-slim")
        command     = body.get("command", "python3 /app/server.py")
        port        = int(body.get("port", 8080))

        content = _generate_sdl(image, command, port, cpu, memory_mb, storage_gb, gpu)
        yaml_path = _write_sdl_file(name, content)

        conn = _get_db()
        try:
            with conn:
                cur = conn.execute(
                    "INSERT OR REPLACE INTO akash_sdls "
                    "(name, content, description, cpu, memory_mb, storage_gb, gpu, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (name, content, description, cpu, memory_mb, storage_gb, gpu, time.time()),
                )
                sdl_id = cur.lastrowid
        except Exception as exc:
            return self._send(500, {"error": str(exc)})
        finally:
            conn.close()

        log.info("SDL created: %s (id=%s)", name, sdl_id)
        return self._send(201, {
            "sdl_id":    sdl_id,
            "name":      name,
            "yaml_path": str(yaml_path),
        })

    def _handle_sdl_list(self):
        conn = _get_db()
        rows = conn.execute(
            "SELECT id, name, description, cpu, memory_mb, storage_gb, gpu, created_at "
            "FROM akash_sdls ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        result = [dict(r) for r in rows]
        return self._send(200, result)

    def _handle_sdl_get(self, sdl_id_str: str):
        try:
            sdl_id = int(sdl_id_str)
        except ValueError:
            return self._send(400, {"error": "Invalid SDL id"})

        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM akash_sdls WHERE id=?", (sdl_id,)
        ).fetchone()
        conn.close()

        if not row:
            return self._send(404, {"error": "SDL not found"})
        return self._send(200, dict(row))

    def _handle_sdl_validate(self, body: dict):
        content = None
        if "sdl_id" in body:
            conn = _get_db()
            row = conn.execute(
                "SELECT content FROM akash_sdls WHERE id=?", (body["sdl_id"],)
            ).fetchone()
            conn.close()
            if not row:
                return self._send(404, {"error": "SDL not found"})
            content = row["content"]
        elif "content" in body:
            content = body["content"]
        else:
            return self._send(400, {"error": "Provide sdl_id or content"})

        valid, errors, parsed = _validate_sdl(content)
        version  = parsed.get("version", "") if parsed else ""
        services = list(parsed.get("services", {}).keys()) if parsed else []
        return self._send(200, {
            "valid":    valid,
            "version":  version,
            "services": services,
            "errors":   errors,
        })

    # ── deployment handlers ────────────────────────────────────────────────────
    def _handle_deploy(self, body: dict):
        if "sdl_id" not in body:
            return self._send(400, {"error": "Missing field: sdl_id"})

        sdl_id = int(body["sdl_id"])
        wallet = body.get("wallet", AKASH_WALLET)
        if not wallet:
            return self._send(400, {"error": "No wallet address provided or configured"})

        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM akash_sdls WHERE id=?", (sdl_id,)
        ).fetchone()
        conn.close()
        if not row:
            return self._send(404, {"error": "SDL not found"})

        sdl_name = row["name"]
        yaml_path = ROOT / f"{sdl_name}.yaml"
        if not yaml_path.exists():
            _write_sdl_file(sdl_name, row["content"])

        if not _akash_available():
            return self._send(200, {
                "status":  "cli_not_installed",
                "message": "Install akash CLI on Termux: pkg install akash",
                "sdl_id":  sdl_id,
            })

        stdout, stderr, rc = _run_akash([
            "tx", "deployment", "create", str(yaml_path),
            "--from", wallet,
            "--node", AKASH_NODE,
            "--chain-id", AKASH_CHAIN_ID,
            "--gas", "auto",
            "--gas-adjustment", "1.4",
            "-y",
            "--output", "json",
        ], timeout=60)

        dseq   = ""
        status = "pending"
        if rc == 0 and stdout:
            try:
                tx_result = json.loads(stdout)
                raw_log = tx_result.get("raw_log", "")
                # Extract dseq from logs
                for event in tx_result.get("logs", [{}]):
                    for ev in event.get("events", []):
                        for attr in ev.get("attributes", []):
                            if attr.get("key") == "dseq":
                                dseq = attr.get("value", "")
                if not dseq and "dseq" in raw_log:
                    # fallback: scan raw_log string
                    import re
                    m = re.search(r'"dseq","value":"(\d+)"', raw_log)
                    if m:
                        dseq = m.group(1)
                status = "created" if dseq else "submitted"
            except Exception:
                status = "submitted"
        else:
            status = "failed"
            log.warning("akash deploy failed rc=%s stderr=%s", rc, stderr)

        monthly_cost = _estimate_cost(
            row["cpu"], row["memory_mb"], row["storage_gb"], row["gpu"]
        )["total_uakt"]

        conn = _get_db()
        with conn:
            cur = conn.execute(
                "INSERT INTO akash_deployments "
                "(dseq, gseq, oseq, provider, sdl_file, status, monthly_cost_uakt, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (dseq, "1", "1", "", str(yaml_path), status, monthly_cost, time.time()),
            )
            dep_id = cur.lastrowid
        conn.close()

        log.info("Deployment recorded: id=%s dseq=%s status=%s", dep_id, dseq, status)
        return self._send(201, {
            "deployment_id": dep_id,
            "dseq":          dseq,
            "status":        status,
            "sdl_id":        sdl_id,
            "monthly_cost_uakt": monthly_cost,
        })

    def _handle_deployments_list(self):
        conn = _get_db()
        rows = conn.execute(
            "SELECT * FROM akash_deployments ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return self._send(200, [dict(r) for r in rows])

    def _handle_deployment_get(self, dep_id_str: str):
        try:
            dep_id = int(dep_id_str)
        except ValueError:
            return self._send(400, {"error": "Invalid deployment id"})

        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM akash_deployments WHERE id=?", (dep_id,)
        ).fetchone()
        conn.close()
        if not row:
            return self._send(404, {"error": "Deployment not found"})
        return self._send(200, dict(row))

    def _handle_deployment_close(self, dep_id_str: str):
        try:
            dep_id = int(dep_id_str)
        except ValueError:
            return self._send(400, {"error": "Invalid deployment id"})

        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM akash_deployments WHERE id=?", (dep_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._send(404, {"error": "Deployment not found"})

        dseq   = row["dseq"]
        wallet = row["provider"] or AKASH_WALLET

        close_status = "closed"
        if _akash_available() and dseq:
            stdout, stderr, rc = _run_akash([
                "tx", "deployment", "close",
                "--dseq", dseq,
                "--from", wallet,
                "--node", AKASH_NODE,
                "--chain-id", AKASH_CHAIN_ID,
                "--gas", "auto",
                "--gas-adjustment", "1.4",
                "-y",
                "--output", "json",
            ], timeout=60)
            if rc != 0:
                log.warning("Close tx rc=%s stderr=%s", rc, stderr)
                close_status = "close_failed"

        now = time.time()
        with conn:
            conn.execute(
                "UPDATE akash_deployments SET status=?, closed_at=? WHERE id=?",
                (close_status, now, dep_id),
            )
        conn.close()
        log.info("Deployment %s status -> %s", dep_id, close_status)
        return self._send(200, {"status": close_status, "deployment_id": dep_id})

    def _handle_deployment_bids(self, dep_id_str: str):
        try:
            dep_id = int(dep_id_str)
        except ValueError:
            return self._send(400, {"error": "Invalid deployment id"})

        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM akash_deployments WHERE id=?", (dep_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._send(404, {"error": "Deployment not found"})

        dseq = row["dseq"]
        bids = []

        if _akash_available() and dseq:
            stdout, stderr, rc = _run_akash([
                "query", "market", "bid", "list",
                "--dseq", dseq,
                "--node", AKASH_NODE,
                "--output", "json",
            ], timeout=30)
            if rc == 0 and stdout:
                try:
                    data = json.loads(stdout)
                    raw_bids = data.get("bids", [])
                    now = time.time()
                    for b in raw_bids:
                        bid_obj  = b.get("bid", b)
                        provider = bid_obj.get("bid_id", {}).get("provider", "")
                        price    = 0.0
                        price_list = bid_obj.get("price", {})
                        if isinstance(price_list, dict):
                            price = float(price_list.get("amount", 0))
                        elif isinstance(price_list, list) and price_list:
                            price = float(price_list[0].get("amount", 0))

                        bids.append({"provider": provider, "price_uakt": price})
                        with conn:
                            conn.execute(
                                "INSERT OR IGNORE INTO akash_bids "
                                "(deployment_id, provider, price_uakt, created_at) "
                                "VALUES (?,?,?,?)",
                                (dep_id, provider, price, now),
                            )
                except Exception as exc:
                    log.warning("Bid parse error: %s", exc)

        if not bids:
            # Return cached bids from DB
            db_bids = conn.execute(
                "SELECT provider, price_uakt FROM akash_bids WHERE deployment_id=?",
                (dep_id,),
            ).fetchall()
            bids = [dict(b) for b in db_bids]

        conn.close()
        return self._send(200, bids)

    def _handle_accept_bid(self, dep_id_str: str, body: dict):
        try:
            dep_id = int(dep_id_str)
        except ValueError:
            return self._send(400, {"error": "Invalid deployment id"})

        provider = body.get("provider", "")
        if not provider:
            return self._send(400, {"error": "Missing field: provider"})

        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM akash_deployments WHERE id=?", (dep_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._send(404, {"error": "Deployment not found"})

        dseq   = row["dseq"]
        wallet = AKASH_WALLET

        if not _akash_available():
            conn.close()
            return self._send(200, {
                "status":  "cli_not_installed",
                "message": "Install akash CLI on Termux: pkg install akash",
            })

        stdout, stderr, rc = _run_akash([
            "tx", "market", "lease", "create",
            "--dseq", dseq,
            "--provider", provider,
            "--from", wallet,
            "--node", AKASH_NODE,
            "--chain-id", AKASH_CHAIN_ID,
            "--gas", "auto",
            "--gas-adjustment", "1.4",
            "-y",
            "--output", "json",
        ], timeout=60)

        if rc == 0:
            with conn:
                conn.execute(
                    "UPDATE akash_deployments SET provider=?, status='leased' WHERE id=?",
                    (provider, dep_id),
                )
            status = "leased"
        else:
            log.warning("Lease create rc=%s stderr=%s", rc, stderr)
            status = "lease_failed"

        conn.close()
        return self._send(200, {"status": status, "provider": provider, "deployment_id": dep_id})

    # ── cost estimation ────────────────────────────────────────────────────────
    def _handle_cost_estimate(self, body: dict):
        cpu            = int(body.get("cpu", 1))
        memory_mb      = int(body.get("memory_mb", 512))
        storage_gb     = int(body.get("storage_gb", 5))
        gpu            = int(body.get("gpu", 0))
        duration_hours = float(body.get("duration_hours", 720))
        return self._send(200, _estimate_cost(cpu, memory_mb, storage_gb, gpu, duration_hours))

    # ── analytics ─────────────────────────────────────────────────────────────
    def _handle_analytics(self):
        conn = _get_db()
        total_deps = conn.execute(
            "SELECT COUNT(*) FROM akash_deployments"
        ).fetchone()[0]
        active_deps = conn.execute(
            "SELECT COUNT(*) FROM akash_deployments WHERE status NOT IN ('closed','close_failed')"
        ).fetchone()[0]
        closed_deps = conn.execute(
            "SELECT COUNT(*) FROM akash_deployments WHERE status IN ('closed','close_failed')"
        ).fetchone()[0]
        monthly_spend = conn.execute(
            "SELECT COALESCE(SUM(monthly_cost_uakt),0) FROM akash_deployments "
            "WHERE status NOT IN ('closed','close_failed')"
        ).fetchone()[0]
        sdl_count = conn.execute(
            "SELECT COUNT(*) FROM akash_sdls"
        ).fetchone()[0]
        bid_count = conn.execute(
            "SELECT COUNT(*) FROM akash_bids"
        ).fetchone()[0]
        conn.close()

        monthly_spend_usd = round(
            (monthly_spend / 1_000_000) * AKT_PRICE_USD, 4
        ) if monthly_spend else 0.0

        return self._send(200, {
            "total_deployments":  total_deps,
            "active_deployments": active_deps,
            "closed_deployments": closed_deps,
            "total_bids":         bid_count,
            "sdl_templates":      sdl_count,
            "monthly_spend_uakt": monthly_spend,
            "monthly_spend_usd":  monthly_spend_usd,
            "akt_price_usd":      AKT_PRICE_USD,
        })


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Initialising FractalMesh Akash Agent on port %d", PORT)
    _init_db()
    _seed_sdls()

    server = HTTPServer(("0.0.0.0", PORT), AkashHandler)

    def _shutdown(sig, frame):
        log.info("Shutting down Akash Agent")
        server.server_close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Akash Agent listening on http://0.0.0.0:%d", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
