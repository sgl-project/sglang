#!/usr/bin/env python3
"""
fm_ab_testing.py — A/B Testing Engine (Port 7849)
FractalMesh OMEGA Titan | Deterministic assignment, z-test significance, Claude-generated variants.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import math
import random
import hashlib
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── vault ──────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT             = int(os.getenv("AB_TESTING_PORT", "7849"))
ADMIN_SECRET     = os.getenv("ADMIN_SECRET", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

ROOT = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
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
            id              INTEGER PRIMARY KEY,
            name            TEXT UNIQUE,
            description     TEXT,
            status          TEXT DEFAULT 'draft',
            hypothesis      TEXT,
            metric          TEXT,
            winning_variant TEXT,
            started_at      REAL,
            ended_at        REAL,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS variants (
            id              INTEGER PRIMARY KEY,
            experiment_id   INTEGER,
            name            TEXT,
            description     TEXT,
            traffic_pct     REAL DEFAULT 50.0,
            content         TEXT,
            metadata        TEXT,
            impressions     INTEGER DEFAULT 0,
            conversions     INTEGER DEFAULT 0,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS assignments (
            id              INTEGER PRIMARY KEY,
            experiment_id   INTEGER,
            variant_id      INTEGER,
            user_id         TEXT,
            assigned_at     REAL
        );
        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY,
            experiment_id   INTEGER,
            variant_id      INTEGER,
            user_id         TEXT,
            event_type      TEXT,
            value           REAL DEFAULT 1.0,
            created_at      REAL
        );
    """)
    conn.commit()
    _seed_example(conn)
    conn.close()


def _seed_example(conn: sqlite3.Connection):
    """Seed a pricing_page_test experiment with two variants if not present."""
    existing = conn.execute(
        "SELECT id FROM experiments WHERE name='pricing_page_test'"
    ).fetchone()
    if existing:
        return

    now = time.time()
    cur = conn.execute(
        """INSERT INTO experiments (name, description, status, hypothesis, metric, created_at)
           VALUES (?, ?, 'draft', ?, 'plan_upgrade', ?)""",
        (
            "pricing_page_test",
            "Test whether showing monthly or annual pricing first increases upgrades",
            "Showing annual pricing first will increase plan upgrades by 15%",
            now,
        ),
    )
    exp_id = cur.lastrowid
    conn.execute(
        """INSERT INTO variants (experiment_id, name, description, traffic_pct,
                                 content, metadata, created_at)
           VALUES (?, 'monthly_first', 'Show monthly pricing tab by default', 50,
                   'Monthly billing — cancel anytime', '{}', ?)""",
        (exp_id, now),
    )
    conn.execute(
        """INSERT INTO variants (experiment_id, name, description, traffic_pct,
                                 content, metadata, created_at)
           VALUES (?, 'annual_first', 'Show annual pricing tab by default (save badge)', 50,
                   'Annual billing — save 20%', '{"badge":"Save 20%"}', ?)""",
        (exp_id, now),
    )
    conn.commit()


# ── helpers ────────────────────────────────────────────────────────────────────

def _check_auth(handler: "ABTestingHandler") -> bool:
    """Return True if the request carries a valid admin secret."""
    auth = handler.headers.get("Authorization", "")
    secret = auth.replace("Bearer ", "").strip()
    return bool(ADMIN_SECRET) and secret == ADMIN_SECRET


def _assign_variant(experiment_id: int, user_id: str, variants: list) -> dict | None:
    """Deterministically pick a variant for a user using MD5 hash bucket."""
    if not variants:
        return None
    bucket = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16) % 100
    cumulative = 0.0
    for v in variants:
        cumulative += float(v["traffic_pct"])
        if bucket < cumulative:
            return v
    return variants[-1]


def _significance_test(v_a: dict, v_b: dict) -> dict:
    """z-test between two variants; returns z_score, p_value label, significant, winner."""
    n_a = max(1, int(v_a["impressions"]))
    n_b = max(1, int(v_b["impressions"]))
    c_a = int(v_a["conversions"])
    c_b = int(v_b["conversions"])

    rate_a = c_a / n_a
    rate_b = c_b / n_b
    pooled = (c_a + c_b) / max(1, n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (rate_a - rate_b) / max(se, 1e-10)

    abs_z = abs(z)
    if abs_z > 2.576:
        p_value = "<0.01"
        significance = "99%"
        significant = True
    elif abs_z > 1.96:
        p_value = "<0.05"
        significance = "95%"
        significant = True
    elif abs_z > 1.645:
        p_value = "<0.10"
        significance = "90%"
        significant = True
    else:
        p_value = ">=0.10"
        significance = "not significant"
        significant = False

    winner = v_a["name"] if rate_a >= rate_b else v_b["name"]
    return {
        "z_score": round(z, 4),
        "p_value": p_value,
        "significance": significance,
        "significant": significant,
        "winner": winner if significant else None,
        "rate_a": round(rate_a, 6),
        "rate_b": round(rate_b, 6),
    }


def _confidence_interval(variant: dict) -> dict:
    """95% Wilson-like CI approximation."""
    n = max(1, int(variant["impressions"]))
    p = int(variant["conversions"]) / n
    z = 1.96
    margin = z * math.sqrt(p * (1 - p) / n)
    return {
        "lower": round(max(0.0, p - margin), 6),
        "upper": round(min(1.0, p + margin), 6),
    }


def _claude_generate(prompt: str, max_tokens: int = 500) -> str:
    """Call Anthropic Messages API and return the assistant text."""
    if not ANTHROPIC_API_KEY:
        return ""
    payload = json.dumps({
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception:
        return ""


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
    payload = json.dumps(body).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _variants_for(conn: sqlite3.Connection, experiment_id: int) -> list:
    rows = conn.execute(
        "SELECT * FROM variants WHERE experiment_id=? ORDER BY id",
        (experiment_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── request handler ────────────────────────────────────────────────────────────

class ABTestingHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-ABTesting/1.0"

    def log_message(self, fmt, *args):  # silence default stderr logs
        pass

    # ── routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            return _send(self, 200, {
                "status": "ok",
                "service": "fm-ab-testing",
                "port": PORT,
            })

        if path == "/experiments":
            return self._list_experiments()

        if path == "/analytics":
            return self._analytics()

        # /experiments/{id}
        parts = path.split("/")
        if len(parts) == 3 and parts[1] == "experiments" and parts[2].isdigit():
            return self._get_experiment(int(parts[2]))

        # /experiments/{id}/results
        if (len(parts) == 4 and parts[1] == "experiments"
                and parts[2].isdigit() and parts[3] == "results"):
            return self._experiment_results(int(parts[2]))

        _send(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/assign":
            return self._assign()

        if path == "/assign/multi":
            return self._assign_multi()

        if path == "/track":
            return self._track()

        if path == "/experiments/create":
            if not _check_auth(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._create_experiment()

        parts = path.split("/")

        # /experiments/{id}/variants/add
        if (len(parts) == 5 and parts[1] == "experiments"
                and parts[2].isdigit() and parts[3] == "variants" and parts[4] == "add"):
            if not _check_auth(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._add_variant(int(parts[2]))

        # /experiments/{id}/start
        if (len(parts) == 4 and parts[1] == "experiments"
                and parts[2].isdigit() and parts[3] == "start"):
            if not _check_auth(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._start_experiment(int(parts[2]))

        # /experiments/{id}/stop
        if (len(parts) == 4 and parts[1] == "experiments"
                and parts[2].isdigit() and parts[3] == "stop"):
            if not _check_auth(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._stop_experiment(int(parts[2]))

        # /experiments/{id}/generate_variants
        if (len(parts) == 4 and parts[1] == "experiments"
                and parts[2].isdigit() and parts[3] == "generate_variants"):
            if not _check_auth(self):
                return _send(self, 401, {"error": "unauthorized"})
            return self._generate_variants(int(parts[2]))

        _send(self, 404, {"error": "not found"})

    # ── experiment endpoints ───────────────────────────────────────────────────

    def _create_experiment(self):
        body = _read_body(self)
        name = body.get("name", "").strip()
        if not name:
            return _send(self, 400, {"error": "name is required"})
        description = body.get("description", "")
        hypothesis  = body.get("hypothesis", "")
        metric      = body.get("metric", "")
        now = time.time()
        try:
            conn = get_db()
            cur = conn.execute(
                """INSERT INTO experiments (name, description, status, hypothesis, metric, created_at)
                   VALUES (?, ?, 'draft', ?, ?, ?)""",
                (name, description, hypothesis, metric, now),
            )
            exp_id = cur.lastrowid
            conn.commit()
            conn.close()
            return _send(self, 201, {"experiment_id": exp_id})
        except sqlite3.IntegrityError:
            conn.close()
            return _send(self, 409, {"error": f"experiment '{name}' already exists"})

    def _add_variant(self, experiment_id: int):
        body = _read_body(self)
        name = body.get("name", "").strip()
        if not name:
            return _send(self, 400, {"error": "name is required"})
        description = body.get("description", "")
        traffic_pct = float(body.get("traffic_pct", 50.0))
        content     = body.get("content", "")
        metadata    = json.dumps(body.get("metadata", {}))
        now = time.time()

        conn = get_db()
        exp = conn.execute(
            "SELECT id FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        existing_pct = conn.execute(
            "SELECT COALESCE(SUM(traffic_pct), 0) FROM variants WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()[0]
        if existing_pct + traffic_pct > 100.0 + 0.01:
            conn.close()
            return _send(self, 400, {
                "error": f"traffic_pct total would exceed 100 ({existing_pct + traffic_pct:.1f})"
            })

        cur = conn.execute(
            """INSERT INTO variants (experiment_id, name, description, traffic_pct,
                                     content, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, name, description, traffic_pct, content, metadata, now),
        )
        variant_id = cur.lastrowid
        conn.commit()
        conn.close()
        return _send(self, 201, {"variant_id": variant_id})

    def _start_experiment(self, experiment_id: int):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        variants = _variants_for(conn, experiment_id)
        if len(variants) < 2:
            conn.close()
            return _send(self, 400, {"error": "at least 2 variants required to start"})

        total_pct = sum(float(v["traffic_pct"]) for v in variants)
        if abs(total_pct - 100.0) > 1.0:
            conn.close()
            return _send(self, 400, {
                "error": f"traffic_pct must sum to ~100 (currently {total_pct:.1f})"
            })

        conn.execute(
            "UPDATE experiments SET status='running', started_at=? WHERE id=?",
            (time.time(), experiment_id),
        )
        conn.commit()
        conn.close()
        return _send(self, 200, {"started": True})

    def _stop_experiment(self, experiment_id: int):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        variants = _variants_for(conn, experiment_id)
        winner_name = None
        confidence  = "not significant"

        if len(variants) >= 2:
            # Find the best performer via pairwise comparison
            best = variants[0]
            for v in variants[1:]:
                result = _significance_test(best, v)
                if result["significant"]:
                    rate_best = int(best["conversions"]) / max(1, int(best["impressions"]))
                    rate_v    = int(v["conversions"]) / max(1, int(v["impressions"]))
                    if rate_v > rate_best:
                        best = v
                    winner_name = best["name"]
                    confidence  = result["significance"]

        conn.execute(
            """UPDATE experiments SET status='completed', ended_at=?, winning_variant=?
               WHERE id=?""",
            (time.time(), winner_name, experiment_id),
        )
        conn.commit()
        conn.close()
        return _send(self, 200, {
            "stopped": True,
            "winner": winner_name,
            "confidence": confidence,
        })

    # ── assignment endpoints ───────────────────────────────────────────────────

    def _assign(self):
        body = _read_body(self)
        experiment_id = body.get("experiment_id")
        user_id       = str(body.get("user_id", "")).strip()
        if experiment_id is None or not user_id:
            return _send(self, 400, {"error": "experiment_id and user_id are required"})
        experiment_id = int(experiment_id)

        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id=? AND status='running'",
            (experiment_id,),
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "running experiment not found"})

        variants = _variants_for(conn, experiment_id)
        if not variants:
            conn.close()
            return _send(self, 400, {"error": "no variants configured"})

        variant = _assign_variant(experiment_id, user_id, variants)
        if not variant:
            conn.close()
            return _send(self, 500, {"error": "could not assign variant"})

        # Record if new assignment
        existing = conn.execute(
            "SELECT id FROM assignments WHERE experiment_id=? AND user_id=?",
            (experiment_id, user_id),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO assignments (experiment_id, variant_id, user_id, assigned_at) VALUES (?,?,?,?)",
                (experiment_id, variant["id"], user_id, time.time()),
            )
            conn.execute(
                "UPDATE variants SET impressions = impressions + 1 WHERE id=?",
                (variant["id"],),
            )
            conn.commit()

        try:
            metadata = json.loads(variant["metadata"] or "{}")
        except Exception:
            metadata = {}

        conn.close()
        return _send(self, 200, {
            "variant_id":   variant["id"],
            "variant_name": variant["name"],
            "content":      variant["content"],
            "metadata":     metadata,
        })

    def _assign_multi(self):
        body = _read_body(self)
        user_id        = str(body.get("user_id", "")).strip()
        experiment_ids = body.get("experiment_ids", [])
        if not user_id or not experiment_ids:
            return _send(self, 400, {"error": "user_id and experiment_ids are required"})

        conn = get_db()
        assignments = []
        now = time.time()
        for exp_id in experiment_ids:
            exp_id = int(exp_id)
            exp = conn.execute(
                "SELECT * FROM experiments WHERE id=? AND status='running'", (exp_id,)
            ).fetchone()
            if not exp:
                continue
            variants = _variants_for(conn, exp_id)
            if not variants:
                continue
            variant = _assign_variant(exp_id, user_id, variants)
            if not variant:
                continue
            existing = conn.execute(
                "SELECT id FROM assignments WHERE experiment_id=? AND user_id=?",
                (exp_id, user_id),
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO assignments (experiment_id, variant_id, user_id, assigned_at) VALUES (?,?,?,?)",
                    (exp_id, variant["id"], user_id, now),
                )
                conn.execute(
                    "UPDATE variants SET impressions = impressions + 1 WHERE id=?",
                    (variant["id"],),
                )
            try:
                metadata = json.loads(variant["metadata"] or "{}")
            except Exception:
                metadata = {}
            assignments.append({
                "experiment_id":   exp_id,
                "experiment_name": exp["name"],
                "variant_id":      variant["id"],
                "variant":         variant["name"],
                "content":         variant["content"],
                "metadata":        metadata,
            })
        conn.commit()
        conn.close()
        return _send(self, 200, {"assignments": assignments})

    # ── tracking ───────────────────────────────────────────────────────────────

    def _track(self):
        body = _read_body(self)
        experiment_id = body.get("experiment_id")
        user_id       = str(body.get("user_id", "")).strip()
        event_type    = str(body.get("event_type", "")).strip()
        value         = float(body.get("value", 1.0))
        if experiment_id is None or not user_id or not event_type:
            return _send(self, 400, {"error": "experiment_id, user_id, event_type required"})
        experiment_id = int(experiment_id)

        conn = get_db()
        # Find the variant this user is assigned to
        assignment = conn.execute(
            "SELECT variant_id FROM assignments WHERE experiment_id=? AND user_id=?",
            (experiment_id, user_id),
        ).fetchone()
        if not assignment:
            conn.close()
            return _send(self, 400, {"error": "user has no assignment for this experiment"})
        variant_id = assignment["variant_id"]

        cur = conn.execute(
            "INSERT INTO events (experiment_id, variant_id, user_id, event_type, value, created_at) VALUES (?,?,?,?,?,?)",
            (experiment_id, variant_id, user_id, event_type, value, time.time()),
        )
        event_id = cur.lastrowid

        # Increment conversions on the variant
        exp = conn.execute("SELECT metric FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp and exp["metric"] == event_type:
            conn.execute(
                "UPDATE variants SET conversions = conversions + 1 WHERE id=?",
                (variant_id,),
            )

        conn.commit()
        conn.close()
        return _send(self, 200, {"tracked": True, "event_id": event_id})

    # ── listing / detail ───────────────────────────────────────────────────────

    def _list_experiments(self):
        conn = get_db()
        rows = conn.execute(
            "SELECT e.*, (SELECT COUNT(*) FROM variants v WHERE v.experiment_id=e.id) AS variant_count FROM experiments e ORDER BY e.created_at DESC"
        ).fetchall()
        conn.close()
        return _send(self, 200, {"experiments": [dict(r) for r in rows]})

    def _get_experiment(self, experiment_id: int):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})
        variants = _variants_for(conn, experiment_id)
        conn.close()
        enriched_variants = []
        for v in variants:
            n = max(1, int(v["impressions"]))
            c = int(v["conversions"])
            try:
                meta = json.loads(v["metadata"] or "{}")
            except Exception:
                meta = {}
            enriched_variants.append({
                **v,
                "metadata": meta,
                "conversion_rate": round(c / n, 6),
            })
        return _send(self, 200, {
            "experiment": dict(exp),
            "variants":   enriched_variants,
        })

    # ── results / analytics ────────────────────────────────────────────────────

    def _experiment_results(self, experiment_id: int):
        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        variants = _variants_for(conn, experiment_id)
        conn.close()

        total_impressions = sum(int(v["impressions"]) for v in variants)
        total_conversions = sum(int(v["conversions"]) for v in variants)
        overall_rate = total_conversions / max(1, total_impressions)

        results = []
        for v in variants:
            n = max(1, int(v["impressions"]))
            c = int(v["conversions"])
            rate = c / n
            ci = _confidence_interval(v)
            try:
                meta = json.loads(v["metadata"] or "{}")
            except Exception:
                meta = {}
            results.append({
                "variant_id":        v["id"],
                "name":              v["name"],
                "impressions":       int(v["impressions"]),
                "conversions":       c,
                "conversion_rate":   round(rate, 6),
                "confidence_interval": ci,
                "traffic_pct":       float(v["traffic_pct"]),
                "metadata":          meta,
            })

        # Pairwise significance tests
        pairwise = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                test = _significance_test(variants[i], variants[j])
                pairwise.append({
                    "variant_a": variants[i]["name"],
                    "variant_b": variants[j]["name"],
                    **test,
                })

        # Recommended winner: highest conversion rate among significant winners
        winner = None
        significance = "not significant"
        lift_pct = 0.0
        if len(results) >= 2:
            sorted_r = sorted(results, key=lambda x: x["conversion_rate"], reverse=True)
            best = sorted_r[0]
            second = sorted_r[1]
            test = _significance_test(
                next(v for v in variants if v["id"] == best["variant_id"]),
                next(v for v in variants if v["id"] == second["variant_id"]),
            )
            if test["significant"]:
                winner      = best["name"]
                significance = test["significance"]
                base_rate    = second["conversion_rate"]
                lift_pct     = round(
                    (best["conversion_rate"] - base_rate) / max(base_rate, 1e-10) * 100, 2
                )

        return _send(self, 200, {
            "experiment_id": experiment_id,
            "status":        exp["status"],
            "metric":        exp["metric"],
            "sample_size":   total_impressions,
            "overall_conversion_rate": round(overall_rate, 6),
            "significance":  significance,
            "winner":        winner,
            "lift_pct":      lift_pct,
            "results":       results,
            "pairwise_tests": pairwise,
        })

    def _analytics(self):
        conn = get_db()
        active_experiments = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='running'"
        ).fetchone()[0]

        day_start = time.time() - 86400
        total_assignments_today = conn.execute(
            "SELECT COUNT(*) FROM assignments WHERE assigned_at >= ?", (day_start,)
        ).fetchone()[0]

        agg = conn.execute(
            "SELECT COALESCE(SUM(impressions),0), COALESCE(SUM(conversions),0) FROM variants"
        ).fetchone()
        total_impressions = int(agg[0])
        total_conversions = int(agg[1])
        overall_conversion_rate = round(
            total_conversions / max(1, total_impressions), 6
        )

        # Best performing experiment by conversion rate
        best_exp = conn.execute(
            """SELECT e.name, SUM(v.conversions)*1.0/MAX(SUM(v.impressions),1) AS cr
               FROM experiments e JOIN variants v ON v.experiment_id=e.id
               WHERE e.status='running'
               GROUP BY e.id ORDER BY cr DESC LIMIT 1"""
        ).fetchone()

        conn.close()
        return _send(self, 200, {
            "active_experiments":      active_experiments,
            "total_assignments_today": total_assignments_today,
            "overall_conversion_rate": overall_conversion_rate,
            "best_experiment":         best_exp["name"] if best_exp else None,
        })

    # ── variant generation via Claude ──────────────────────────────────────────

    def _generate_variants(self, experiment_id: int):
        body = _read_body(self)
        base_content = body.get("base_content", "").strip()
        goal         = body.get("goal", "increase_clicks").strip()
        n_variants   = max(1, min(10, int(body.get("n_variants", 3))))

        if not base_content:
            return _send(self, 400, {"error": "base_content is required"})

        conn = get_db()
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if not exp:
            conn.close()
            return _send(self, 404, {"error": "experiment not found"})

        # Check available traffic headroom
        existing_pct = conn.execute(
            "SELECT COALESCE(SUM(traffic_pct), 0) FROM variants WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()[0]
        available_pct = 100.0 - float(existing_pct)
        per_variant_pct = round(available_pct / n_variants, 2)

        prompt = (
            f"You are a conversion rate optimization expert. "
            f"Generate {n_variants} distinct A/B test variant copy alternatives for the following:\n\n"
            f"Base content: {base_content}\n"
            f"Goal: {goal}\n\n"
            f"Return ONLY a JSON array of objects with keys 'name' (short slug), "
            f"'content' (the variant copy), and 'description' (one sentence rationale). "
            f"No markdown, no explanation, just the JSON array."
        )

        raw = _claude_generate(prompt, max_tokens=600)

        # Parse Claude's response
        generated = []
        try:
            # Strip any markdown code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            generated = json.loads(cleaned.strip())
            if not isinstance(generated, list):
                generated = []
        except Exception:
            # Fallback: create simple numbered variants
            generated = [
                {
                    "name": f"variant_{i+1}",
                    "content": f"{base_content} (variant {i+1})",
                    "description": f"Auto-generated variant {i+1}",
                }
                for i in range(n_variants)
            ]

        now = time.time()
        created_variants = []
        for item in generated[:n_variants]:
            name        = str(item.get("name", f"variant_{random.randint(1000,9999)}"))
            content     = str(item.get("content", base_content))
            description = str(item.get("description", ""))
            cur = conn.execute(
                """INSERT INTO variants (experiment_id, name, description, traffic_pct,
                                         content, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, '{}', ?)""",
                (experiment_id, name, description, per_variant_pct, content, now),
            )
            created_variants.append({
                "variant_id":  cur.lastrowid,
                "name":        name,
                "content":     content,
                "description": description,
                "traffic_pct": per_variant_pct,
            })

        conn.commit()
        conn.close()
        return _send(self, 201, {
            "variants_created": len(created_variants),
            "variants":         created_variants,
        })


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), ABTestingHandler)
    print(f"[fm-ab-testing] Listening on port {PORT} | DB={DB}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[fm-ab-testing] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
