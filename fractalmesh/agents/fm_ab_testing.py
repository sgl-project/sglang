#!/usr/bin/env python3
"""
fm_ab_testing.py — A/B Testing & Experimentation Platform (Port 7890)
FractalMesh OMEGA Titan | Deterministic assignment, z-test significance, statistical analysis.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import statistics
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault loading (must be before any os.getenv calls) ────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("AB_TESTING_PORT", "7890"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

ROOT = Path(os.path.expanduser("~/fmsaas"))
DB   = ROOT / "database" / "sovereign.db"

ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)


# ── database ───────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id                  INTEGER PRIMARY KEY,
            exp_id              TEXT UNIQUE,
            name                TEXT,
            description         TEXT,
            status              TEXT DEFAULT 'draft',
            hypothesis          TEXT,
            metric              TEXT DEFAULT 'conversion_rate',
            traffic_percentage  REAL DEFAULT 1.0,
            min_sample_size     INTEGER DEFAULT 100,
            confidence_level    REAL DEFAULT 0.95,
            winner_variant      TEXT,
            started_at          REAL,
            ended_at            REAL,
            created_at          REAL,
            updated_at          REAL
        );
        CREATE TABLE IF NOT EXISTS variants (
            id          INTEGER PRIMARY KEY,
            variant_id  TEXT UNIQUE,
            exp_id      TEXT,
            name        TEXT,
            description TEXT,
            weight      REAL DEFAULT 0.5,
            config      TEXT DEFAULT '{}',
            is_control  INTEGER DEFAULT 0,
            impressions INTEGER DEFAULT 0,
            conversions INTEGER DEFAULT 0,
            revenue     REAL DEFAULT 0,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS assignments (
            id          INTEGER PRIMARY KEY,
            exp_id      TEXT,
            user_id     TEXT,
            variant_id  TEXT,
            assigned_at REAL,
            UNIQUE(exp_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY,
            event_id    TEXT UNIQUE,
            exp_id      TEXT,
            variant_id  TEXT,
            user_id     TEXT,
            event_type  TEXT,
            value       REAL DEFAULT 0,
            metadata    TEXT,
            created_at  REAL
        );
    """)
    conn.commit()
    conn.close()


# ── statistical helpers ────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """CDF of the standard normal distribution using the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _z_test(n_a: int, conv_a: int, n_b: int, conv_b: int):
    """
    Two-proportion z-test.
    Returns (z_score, p_value) where p_value is a two-tailed probability.
    """
    p_a = conv_a / n_a if n_a > 0 else 0.0
    p_b = conv_b / n_b if n_b > 0 else 0.0
    p_pool = (conv_a + conv_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0.0
    se = (
        math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        if (n_a > 0 and n_b > 0)
        else 0.0
    )
    z = (p_b - p_a) / se if se > 0 else 0.0
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return z, p_value


def _confidence_interval_95(impressions: int, conversions: int) -> dict:
    """Wilson score 95% confidence interval."""
    n = max(1, impressions)
    p = conversions / n
    z = 1.96
    margin = z * math.sqrt(p * (1 - p) / n)
    return {
        "lower": round(max(0.0, p - margin), 6),
        "upper": round(min(1.0, p + margin), 6),
    }


def _lift_percent(rate_control: float, rate_variant: float) -> float:
    """Percentage lift of variant over control."""
    if rate_control <= 0:
        return 0.0
    return round((rate_variant - rate_control) / rate_control * 100, 4)


# ── deterministic assignment ───────────────────────────────────────────────────

def _assign_variant(exp_id: str, user_id: str, variants: list) -> dict | None:
    """
    Hash user+exp to get a stable bucket 0-99, then use cumulative weights to pick variant.
    Variants are sorted by name for stability.
    """
    if not variants:
        return None
    h = int(hashlib.sha256(f"{exp_id}:{user_id}".encode()).hexdigest(), 16) % 100
    cumulative = 0.0
    for v in sorted(variants, key=lambda x: x["name"]):
        cumulative += v["weight"] * 100
        if h < cumulative:
            return v
    return variants[-1]


# ── security helpers ───────────────────────────────────────────────────────────

def _check_admin(handler: "ABTestingHandler") -> bool:
    """Return True if the X-Admin-Secret header matches ADMIN_SECRET via constant-time comparison."""
    if not ADMIN_SECRET:
        return False
    provided = handler.headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(
        ADMIN_SECRET.encode("utf-8"),
        provided.encode("utf-8"),
    )


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _read_body(handler: "ABTestingHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _send(handler: "ABTestingHandler", code: int, body: dict):
    payload = json.dumps(body, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _get_query_params(path: str) -> dict:
    """Parse query string from a path like /foo?a=1&b=2."""
    if "?" not in path:
        return {}
    qs = path.split("?", 1)[1]
    params = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return params


# ── experiment data helpers ────────────────────────────────────────────────────

def _variants_for(conn: sqlite3.Connection, exp_id: str) -> list:
    rows = conn.execute(
        "SELECT * FROM variants WHERE exp_id=? ORDER BY name",
        (exp_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _enrich_variant(v: dict) -> dict:
    """Add conversion_rate and parsed config to a variant dict."""
    n = max(1, int(v["impressions"]))
    c = int(v["conversions"])
    try:
        cfg = json.loads(v.get("config") or "{}")
    except Exception:
        cfg = {}
    return {
        **v,
        "config": cfg,
        "conversion_rate": round(c / n, 6),
        "confidence_interval": _confidence_interval_95(n, c),
    }


def _run_significance_analysis(conn: sqlite3.Connection, exp_id: str) -> dict:
    """
    Run pairwise z-test analysis across all variants.
    Returns a results dict with per-variant stats, pairwise tests, and recommended winner.
    """
    exp = conn.execute(
        "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
    ).fetchone()
    if not exp:
        return {}

    variants = _variants_for(conn, exp_id)
    enriched = [_enrich_variant(v) for v in variants]

    total_impressions = sum(int(v["impressions"]) for v in variants)
    total_conversions = sum(int(v["conversions"]) for v in variants)
    overall_rate = total_conversions / max(1, total_impressions)

    # Find control variant
    control = next((v for v in enriched if v.get("is_control")), None)

    # Pairwise z-tests
    pairwise = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            va = variants[i]
            vb = variants[j]
            z, p_val = _z_test(
                int(va["impressions"]), int(va["conversions"]),
                int(vb["impressions"]), int(vb["conversions"]),
            )
            significant = p_val < (1 - float(exp["confidence_level"]))
            pairwise.append({
                "variant_a":   va["name"],
                "variant_b":   vb["name"],
                "z_score":     round(z, 6),
                "p_value":     round(p_val, 6),
                "significant": significant,
                "confidence":  float(exp["confidence_level"]),
            })

    # Recommend winner: highest conversion rate among variants that beat control significantly,
    # or highest overall if no control.
    winner_name = None
    winner_p_value = None
    if control and len(enriched) > 1:
        challengers = [v for v in enriched if not v.get("is_control")]
        best_challenger = max(challengers, key=lambda x: x["conversion_rate"])
        z, p_val = _z_test(
            int(control["impressions"]), int(control["conversions"]),
            int(best_challenger["impressions"]), int(best_challenger["conversions"]),
        )
        significant = p_val < (1 - float(exp["confidence_level"]))
        if significant and best_challenger["conversion_rate"] > control["conversion_rate"]:
            winner_name = best_challenger["name"]
            winner_p_value = round(p_val, 6)
        elif significant and control["conversion_rate"] > best_challenger["conversion_rate"]:
            winner_name = control["name"]
            winner_p_value = round(p_val, 6)
    elif len(enriched) >= 2:
        sorted_v = sorted(enriched, key=lambda x: x["conversion_rate"], reverse=True)
        best = sorted_v[0]
        second = sorted_v[1]
        z, p_val = _z_test(
            int(second["impressions"]), int(second["conversions"]),
            int(best["impressions"]), int(best["conversions"]),
        )
        significant = p_val < (1 - float(exp["confidence_level"]))
        if significant:
            winner_name = best["name"]
            winner_p_value = round(p_val, 6)

    lift = 0.0
    if winner_name and control and winner_name != control["name"]:
        winner_v = next((v for v in enriched if v["name"] == winner_name), None)
        if winner_v:
            lift = _lift_percent(control["conversion_rate"], winner_v["conversion_rate"])

    return {
        "exp_id":                   exp_id,
        "status":                   exp["status"],
        "metric":                   exp["metric"],
        "total_impressions":        total_impressions,
        "total_conversions":        total_conversions,
        "overall_conversion_rate":  round(overall_rate, 6),
        "confidence_level":         float(exp["confidence_level"]),
        "recommended_winner":       winner_name,
        "winner_p_value":           winner_p_value,
        "lift_percent":             lift,
        "variants":                 enriched,
        "pairwise_tests":           pairwise,
    }


# ── background analysis daemon ─────────────────────────────────────────────────

def _analysis_daemon():
    """
    Daemon thread: every 3600 seconds, check running experiments that have reached
    min_sample_size. Run z-test; if p_value < (1 - confidence_level) set winner_variant
    and status='completed'.
    """
    while True:
        time.sleep(3600)
        try:
            conn = get_db()
            running = conn.execute(
                "SELECT * FROM experiments WHERE status='running'"
            ).fetchall()
            for exp in running:
                exp_id = exp["exp_id"]
                variants = _variants_for(conn, exp_id)
                total_impressions = sum(int(v["impressions"]) for v in variants)
                if total_impressions < int(exp["min_sample_size"]):
                    continue
                # Run significance analysis
                results = _run_significance_analysis(conn, exp_id)
                winner_name = results.get("recommended_winner")
                winner_p = results.get("winner_p_value")
                if winner_name and winner_p is not None:
                    threshold = 1.0 - float(exp["confidence_level"])
                    if winner_p < threshold:
                        now = time.time()
                        conn.execute(
                            """UPDATE experiments
                               SET status='completed', winner_variant=?, ended_at=?, updated_at=?
                               WHERE exp_id=?""",
                            (winner_name, now, now, exp_id),
                        )
                        conn.commit()
            conn.close()
        except Exception:
            pass


# ── request handler ────────────────────────────────────────────────────────────

class ABTestingHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-ABTesting/2.0"

    def log_message(self, fmt, *args):  # suppress default stderr logging
        pass

    # ── routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        raw_path = self.path
        path = raw_path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        # GET /health
        if path == "/health":
            return _send(self, 200, {
                "status":  "ok",
                "service": "fm-ab-testing",
                "port":    PORT,
                "db":      str(DB),
            })

        # GET /experiments
        if path == "/experiments":
            return self._list_experiments(raw_path)

        # GET /experiments/{exp_id}
        if len(parts) == 2 and parts[0] == "experiments":
            return self._get_experiment(parts[1])

        # GET /results/{exp_id}
        if len(parts) == 2 and parts[0] == "results":
            return self._get_results(parts[1])

        # GET /assignment/{exp_id}/{user_id}
        if len(parts) == 3 and parts[0] == "assignment":
            return self._get_or_create_assignment(parts[1], parts[2])

        # GET /dashboard
        if path == "/dashboard":
            if not _check_admin(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._dashboard()

        _send(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        # POST /experiments
        if path == "/experiments":
            if not _check_admin(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._create_experiment()

        # POST /assign
        if path == "/assign":
            return self._assign()

        # POST /track
        if path == "/track":
            return self._track()

        # POST /experiments/{exp_id}/variants
        if (len(parts) == 3 and parts[0] == "experiments"
                and parts[2] == "variants"):
            if not _check_admin(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._add_variant(parts[1])

        # POST /experiments/{exp_id}/start
        if (len(parts) == 3 and parts[0] == "experiments"
                and parts[2] == "start"):
            if not _check_admin(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._start_experiment(parts[1])

        # POST /experiments/{exp_id}/stop
        if (len(parts) == 3 and parts[0] == "experiments"
                and parts[2] == "stop"):
            if not _check_admin(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._stop_experiment(parts[1])

        _send(self, 404, {"error": "not found"})

    # ── experiment management ──────────────────────────────────────────────────

    def _create_experiment(self):
        body = _read_body(self)
        name = str(body.get("name", "")).strip()
        if not name:
            return _send(self, 400, {"error": "name is required"})

        description        = str(body.get("description", ""))
        hypothesis         = str(body.get("hypothesis", ""))
        metric             = str(body.get("metric", "conversion_rate"))
        traffic_percentage = float(body.get("traffic_percentage", 1.0))
        min_sample_size    = int(body.get("min_sample_size", 100))
        confidence_level   = float(body.get("confidence_level", 0.95))

        if not (0.0 < traffic_percentage <= 1.0):
            return _send(self, 400, {"error": "traffic_percentage must be between 0 and 1"})
        if not (0.0 < confidence_level < 1.0):
            return _send(self, 400, {"error": "confidence_level must be between 0 and 1"})

        now    = time.time()
        exp_id = secrets.token_hex(8)

        try:
            conn = get_db()
            conn.execute(
                """INSERT INTO experiments
                   (exp_id, name, description, status, hypothesis, metric,
                    traffic_percentage, min_sample_size, confidence_level,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (exp_id, name, description, "draft", hypothesis, metric,
                 traffic_percentage, min_sample_size, confidence_level,
                 now, now),
            )
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            try:
                conn.close()
            except Exception:
                pass
            return _send(self, 409, {"error": "experiment already exists"})

        return _send(self, 201, {
            "exp_id":             exp_id,
            "name":               name,
            "status":             "draft",
            "metric":             metric,
            "traffic_percentage": traffic_percentage,
            "min_sample_size":    min_sample_size,
            "confidence_level":   confidence_level,
            "created_at":         now,
        })

    def _list_experiments(self, raw_path: str):
        params = _get_query_params(raw_path)
        status_filter = params.get("status", "")

        conn = get_db()
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status=? ORDER BY created_at DESC",
                (status_filter,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        conn.close()

        experiments = []
        for r in rows:
            experiments.append(dict(r))

        return _send(self, 200, {
            "experiments": experiments,
            "count":       len(experiments),
        })

    def _get_experiment(self, exp_id: str):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        variants = _variants_for(conn, exp_id)
        conn.close()

        enriched = [_enrich_variant(v) for v in variants]
        return _send(self, 200, {
            "experiment": dict(exp),
            "variants":   enriched,
        })

    def _add_variant(self, exp_id: str):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})
        if exp["status"] not in ("draft",):
            conn.close()
            return _send(self, 400, {
                "error": f"cannot add variants to experiment in status '{exp['status']}'"
            })

        body       = _read_body(self)
        name       = str(body.get("name", "")).strip()
        if not name:
            conn.close()
            return _send(self, 400, {"error": "name is required"})

        description = str(body.get("description", ""))
        weight      = float(body.get("weight", 0.5))
        config      = json.dumps(body.get("config", {}))
        is_control  = int(bool(body.get("is_control", False)))
        now         = time.time()
        variant_id  = secrets.token_hex(8)

        conn.execute(
            """INSERT INTO variants
               (variant_id, exp_id, name, description, weight, config, is_control, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (variant_id, exp_id, name, description, weight, config, is_control, now),
        )
        conn.commit()
        conn.close()

        return _send(self, 201, {
            "variant_id": variant_id,
            "exp_id":     exp_id,
            "name":       name,
            "weight":     weight,
            "is_control": bool(is_control),
            "created_at": now,
        })

    def _start_experiment(self, exp_id: str):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})
        if exp["status"] != "draft":
            conn.close()
            return _send(self, 400, {
                "error": f"experiment is already in status '{exp['status']}'"
            })

        variants = _variants_for(conn, exp_id)
        if len(variants) < 2:
            conn.close()
            return _send(self, 400, {"error": "at least 2 variants required to start"})

        now = time.time()
        conn.execute(
            "UPDATE experiments SET status='running', started_at=?, updated_at=? WHERE exp_id=?",
            (now, now, exp_id),
        )
        conn.commit()
        conn.close()
        return _send(self, 200, {"exp_id": exp_id, "status": "running", "started_at": now})

    def _stop_experiment(self, exp_id: str):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})
        if exp["status"] not in ("running",):
            conn.close()
            return _send(self, 400, {
                "error": f"experiment is not running (status: '{exp['status']}')"
            })

        # Run significance analysis before stopping
        results = _run_significance_analysis(conn, exp_id)
        winner_name = results.get("recommended_winner")

        now = time.time()
        conn.execute(
            """UPDATE experiments
               SET status='stopped', ended_at=?, updated_at=?, winner_variant=?
               WHERE exp_id=?""",
            (now, now, winner_name, exp_id),
        )
        conn.commit()
        conn.close()

        return _send(self, 200, {
            "exp_id":  exp_id,
            "status":  "stopped",
            "ended_at": now,
            "winner_variant": winner_name,
            "analysis": results,
        })

    # ── assignment ─────────────────────────────────────────────────────────────

    def _assign(self):
        body   = _read_body(self)
        exp_id = str(body.get("exp_id", "")).strip()
        user_id = str(body.get("user_id", "")).strip()
        if not exp_id or not user_id:
            return _send(self, 400, {"error": "exp_id and user_id are required"})

        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=? AND status='running'",
            (exp_id,),
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "running experiment not found"})

        variants = _variants_for(conn, exp_id)
        if not variants:
            conn.close()
            return _send(self, 400, {"error": "no variants configured"})

        # Check traffic_percentage gating
        traffic_h = int(
            hashlib.sha256(f"traffic:{exp_id}:{user_id}".encode()).hexdigest(), 16
        ) % 100
        if traffic_h >= float(exp["traffic_percentage"]) * 100:
            conn.close()
            return _send(self, 200, {"assigned": False, "reason": "outside traffic percentage"})

        variant = _assign_variant(exp_id, user_id, variants)
        if not variant:
            conn.close()
            return _send(self, 500, {"error": "could not assign variant"})

        # Upsert assignment; only increment impressions on first assignment
        existing = conn.execute(
            "SELECT variant_id FROM assignments WHERE exp_id=? AND user_id=?",
            (exp_id, user_id),
        ).fetchone()

        if not existing:
            conn.execute(
                """INSERT INTO assignments (exp_id, user_id, variant_id, assigned_at)
                   VALUES (?,?,?,?)""",
                (exp_id, user_id, variant["variant_id"], time.time()),
            )
            conn.execute(
                "UPDATE variants SET impressions = impressions + 1 WHERE variant_id=?",
                (variant["variant_id"],),
            )
            conn.commit()
        else:
            # Return the previously assigned variant for consistency
            assigned_vid = existing["variant_id"]
            variant = dict(conn.execute(
                "SELECT * FROM variants WHERE variant_id=?", (assigned_vid,)
            ).fetchone())

        try:
            cfg = json.loads(variant.get("config") or "{}")
        except Exception:
            cfg = {}

        conn.close()
        return _send(self, 200, {
            "assigned":   True,
            "exp_id":     exp_id,
            "user_id":    user_id,
            "variant_id": variant["variant_id"],
            "name":       variant["name"],
            "is_control": bool(variant.get("is_control")),
            "config":     cfg,
        })

    def _get_or_create_assignment(self, exp_id: str, user_id: str):
        """GET /assignment/{exp_id}/{user_id} — return or create assignment."""
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        existing = conn.execute(
            "SELECT * FROM assignments WHERE exp_id=? AND user_id=?",
            (exp_id, user_id),
        ).fetchone()

        if existing:
            variant = conn.execute(
                "SELECT * FROM variants WHERE variant_id=?", (existing["variant_id"],)
            ).fetchone()
            conn.close()
            if not variant:
                return _send(self, 404, {"error": "assigned variant not found"})
            try:
                cfg = json.loads(variant["config"] or "{}")
            except Exception:
                cfg = {}
            return _send(self, 200, {
                "assigned":   True,
                "exp_id":     exp_id,
                "user_id":    user_id,
                "variant_id": variant["variant_id"],
                "name":       variant["name"],
                "is_control": bool(variant["is_control"]),
                "config":     cfg,
                "assigned_at": existing["assigned_at"],
            })

        # No existing assignment — create if experiment is running
        if exp["status"] != "running":
            conn.close()
            return _send(self, 400, {
                "error": f"experiment is not running (status: '{exp['status']}')"
            })

        variants = _variants_for(conn, exp_id)
        if not variants:
            conn.close()
            return _send(self, 400, {"error": "no variants configured"})

        variant = _assign_variant(exp_id, user_id, variants)
        if not variant:
            conn.close()
            return _send(self, 500, {"error": "could not assign variant"})

        now = time.time()
        try:
            conn.execute(
                """INSERT INTO assignments (exp_id, user_id, variant_id, assigned_at)
                   VALUES (?,?,?,?)""",
                (exp_id, user_id, variant["variant_id"], now),
            )
            conn.execute(
                "UPDATE variants SET impressions = impressions + 1 WHERE variant_id=?",
                (variant["variant_id"],),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Race condition; assignment was just created by another thread

        try:
            cfg = json.loads(variant.get("config") or "{}")
        except Exception:
            cfg = {}

        conn.close()
        return _send(self, 200, {
            "assigned":    True,
            "exp_id":      exp_id,
            "user_id":     user_id,
            "variant_id":  variant["variant_id"],
            "name":        variant["name"],
            "is_control":  bool(variant.get("is_control")),
            "config":      cfg,
            "assigned_at": now,
        })

    # ── event tracking ─────────────────────────────────────────────────────────

    def _track(self):
        body       = _read_body(self)
        exp_id     = str(body.get("exp_id", "")).strip()
        user_id    = str(body.get("user_id", "")).strip()
        event_type = str(body.get("event_type", "")).strip()
        value      = float(body.get("value", 0.0))
        metadata   = json.dumps(body.get("metadata", {}))

        if not exp_id or not user_id or not event_type:
            return _send(self, 400, {"error": "exp_id, user_id, event_type are required"})

        conn = get_db()
        assignment = conn.execute(
            "SELECT variant_id FROM assignments WHERE exp_id=? AND user_id=?",
            (exp_id, user_id),
        ).fetchone()
        if not assignment:
            conn.close()
            return _send(self, 400, {"error": "no assignment found for this user/experiment"})

        variant_id = assignment["variant_id"]
        event_id   = secrets.token_hex(12)
        now        = time.time()

        conn.execute(
            """INSERT INTO events (event_id, exp_id, variant_id, user_id, event_type, value, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (event_id, exp_id, variant_id, user_id, event_type, value, metadata, now),
        )

        # Increment conversions and revenue when event_type='conversion'
        if event_type == "conversion":
            conn.execute(
                """UPDATE variants
                   SET conversions = conversions + 1,
                       revenue = revenue + ?
                   WHERE variant_id=?""",
                (value, variant_id),
            )

        conn.commit()
        conn.close()

        return _send(self, 200, {
            "tracked":    True,
            "event_id":   event_id,
            "exp_id":     exp_id,
            "variant_id": variant_id,
            "event_type": event_type,
            "value":      value,
            "created_at": now,
        })

    # ── results ────────────────────────────────────────────────────────────────

    def _get_results(self, exp_id: str):
        """GET /results/{exp_id} — full statistical analysis."""
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        results = _run_significance_analysis(conn, exp_id)
        conn.close()
        return _send(self, 200, results)

    # ── dashboard ──────────────────────────────────────────────────────────────

    def _dashboard(self):
        """GET /dashboard — admin: running experiments, total assignments, experiments with winners."""
        conn = get_db()

        running_exps = conn.execute(
            "SELECT * FROM experiments WHERE status='running' ORDER BY started_at DESC"
        ).fetchall()
        running_exps_list = [dict(r) for r in running_exps]

        total_assignments = conn.execute(
            "SELECT COUNT(*) FROM assignments"
        ).fetchone()[0]

        experiments_with_winners = conn.execute(
            """SELECT * FROM experiments
               WHERE winner_variant IS NOT NULL AND winner_variant != ''
               ORDER BY ended_at DESC"""
        ).fetchall()
        winners_list = [dict(r) for r in experiments_with_winners]

        # Aggregate stats
        total_experiments = conn.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0]

        completed_experiments = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='completed'"
        ).fetchone()[0]

        stopped_experiments = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='stopped'"
        ).fetchone()[0]

        draft_experiments = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='draft'"
        ).fetchone()[0]

        total_impressions = conn.execute(
            "SELECT COALESCE(SUM(impressions), 0) FROM variants"
        ).fetchone()[0]

        total_conversions = conn.execute(
            "SELECT COALESCE(SUM(conversions), 0) FROM variants"
        ).fetchone()[0]

        total_revenue = conn.execute(
            "SELECT COALESCE(SUM(revenue), 0.0) FROM variants"
        ).fetchone()[0]

        # Recent assignments (last 24h)
        day_start = time.time() - 86400
        recent_assignments = conn.execute(
            "SELECT COUNT(*) FROM assignments WHERE assigned_at >= ?",
            (day_start,),
        ).fetchone()[0]

        # Recent events (last 24h)
        recent_events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE created_at >= ?",
            (day_start,),
        ).fetchone()[0]

        conn.close()

        overall_rate = total_conversions / max(1, total_impressions)

        return _send(self, 200, {
            "summary": {
                "total_experiments":      total_experiments,
                "running_experiments":    len(running_exps_list),
                "completed_experiments":  completed_experiments,
                "stopped_experiments":    stopped_experiments,
                "draft_experiments":      draft_experiments,
                "total_assignments":      total_assignments,
                "recent_assignments_24h": recent_assignments,
                "recent_events_24h":      recent_events,
                "total_impressions":      total_impressions,
                "total_conversions":      total_conversions,
                "overall_conversion_rate": round(overall_rate, 6),
                "total_revenue":          round(float(total_revenue), 4),
                "experiments_with_winners": len(winners_list),
            },
            "running_experiments":      running_exps_list,
            "experiments_with_winners": winners_list,
        })


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main():
    init_db()

    # Start background analysis daemon
    daemon = threading.Thread(target=_analysis_daemon, daemon=True, name="ab-analysis-daemon")
    daemon.start()

    server = HTTPServer(("0.0.0.0", PORT), ABTestingHandler)
    print(f"[fm-ab-testing] Listening on port {PORT} | DB={DB}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[fm-ab-testing] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
