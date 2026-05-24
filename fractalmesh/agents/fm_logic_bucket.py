#!/usr/bin/env python3
"""
fm_logic_bucket.py — Logic Bucket & Committee Training Agent (Port 7836)
FractalMesh OMEGA Titan — multi-model committee reasoning, logic rule engine,
and training example management.
All credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import sqlite3
import logging
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT            = int(os.getenv("LOGIC_BUCKET_PORT", "7836"))
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")
MCP_SECRET      = os.getenv("MCP_SECRET", "")
ROOT            = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB              = ROOT / "database" / "sovereign.db"
LOG             = ROOT / "logs" / "fm_logic_bucket.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FM-LOGIC-BUCKET] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("fm_logic_bucket")

# ── committee members ─────────────────────────────────────────────────────────
COMMITTEE_MEMBERS = {
    "claude_haiku": {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
    "mistral_7b":   {"provider": "openrouter", "model": "mistralai/mistral-7b-instruct"},
    "llama_70b":    {"provider": "openrouter", "model": "meta-llama/llama-3-70b-instruct"},
    "gemma_9b":     {"provider": "openrouter", "model": "google/gemma-2-9b-it"},
}

# ── global thread pool ─────────────────────────────────────────────────────────
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# ── database ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS logic_rules (
                id            INTEGER PRIMARY KEY,
                name          TEXT UNIQUE,
                description   TEXT,
                rule_type     TEXT,
                conditions    TEXT,
                actions       TEXT,
                priority      INTEGER DEFAULT 5,
                enabled       INTEGER DEFAULT 1,
                trigger_count INTEGER DEFAULT 0,
                created_at    REAL
            );

            CREATE TABLE IF NOT EXISTS committee_sessions (
                id           INTEGER PRIMARY KEY,
                question     TEXT,
                context      TEXT,
                members      TEXT,
                strategy     TEXT,
                consensus    TEXT,
                responses    TEXT,
                final_answer TEXT,
                confidence   REAL,
                created_at   REAL
            );

            CREATE TABLE IF NOT EXISTS decision_log (
                id           INTEGER PRIMARY KEY,
                rule_id      INTEGER,
                input_data   TEXT,
                matched      INTEGER,
                action_taken TEXT,
                result       TEXT,
                created_at   REAL
            );

            CREATE TABLE IF NOT EXISTS training_examples (
                id              INTEGER PRIMARY KEY,
                input_text      TEXT,
                expected_output TEXT,
                actual_output   TEXT,
                model           TEXT,
                score           REAL,
                feedback        TEXT,
                created_at      REAL
            );
        """)
    _seed_rules()


def _seed_rules() -> None:
    defaults = [
        {
            "name": "auto_tag_high_score",
            "description": "Tag lead as qualified if score >= 0.7",
            "rule_type": "threshold",
            "conditions": {"field": "score", "operator": "gte", "value": 0.7},
            "actions": [{"type": "tag", "value": "qualified"}],
            "priority": 1,
        },
        {
            "name": "flag_dental_leads",
            "description": "Tag leads from dental sector",
            "rule_type": "keyword",
            "conditions": {"field": "company", "operator": "contains", "value": "dental"},
            "actions": [{"type": "tag", "value": "dental-sector"}],
            "priority": 2,
        },
        {
            "name": "email_required",
            "description": "Tag lead as contactable if email exists",
            "rule_type": "existence",
            "conditions": {"field": "email", "operator": "exists", "value": True},
            "actions": [{"type": "tag", "value": "contactable"}],
            "priority": 3,
        },
    ]
    with _db() as conn:
        for rule in defaults:
            existing = conn.execute(
                "SELECT id FROM logic_rules WHERE name=?", (rule["name"],)
            ).fetchone()
            if not existing:
                conn.execute(
                    """INSERT INTO logic_rules
                       (name, description, rule_type, conditions, actions, priority, enabled, created_at)
                       VALUES (?,?,?,?,?,?,1,?)""",
                    (
                        rule["name"],
                        rule.get("description", ""),
                        rule.get("rule_type", "custom"),
                        json.dumps(rule["conditions"]),
                        json.dumps(rule["actions"]),
                        rule["priority"],
                        time.time(),
                    ),
                )


# ── model caller ──────────────────────────────────────────────────────────────
def _call_model(provider: str, model: str, messages: list, max_tokens: int = 500) -> str:
    """Call an AI model and return the text response."""
    try:
        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["content"][0]["text"].strip()

        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer": "https://fractalmesh.net",
                "X-Title": "FractalMesh OMEGA Titan",
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()

        else:
            return f"[ERROR] Unknown provider: {provider}"

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        log.error("HTTP %s calling %s/%s: %s", e.code, provider, model, body[:200])
        return f"[ERROR] HTTP {e.code}: {body[:200]}"
    except Exception as exc:
        log.error("Error calling %s/%s: %s", provider, model, exc)
        return f"[ERROR] {exc}"


# ── condition evaluator ───────────────────────────────────────────────────────
def _eval_condition(conditions: dict, data: dict) -> bool:
    """Evaluate a conditions dict against a data dict. Supports compound and/or."""
    if "and" in conditions:
        return all(_eval_condition(c, data) for c in conditions["and"])
    if "or" in conditions:
        return any(_eval_condition(c, data) for c in conditions["or"])

    field    = conditions.get("field", "")
    operator = conditions.get("operator", "eq")
    value    = conditions.get("value")

    field_val = data.get(field)

    if operator == "exists":
        return (field_val is not None) if value else (field_val is None)
    if field_val is None:
        return False

    try:
        if operator == "eq":
            return field_val == value
        elif operator == "gt":
            return float(field_val) > float(value)
        elif operator == "lt":
            return float(field_val) < float(value)
        elif operator == "gte":
            return float(field_val) >= float(value)
        elif operator == "lte":
            return float(field_val) <= float(value)
        elif operator == "contains":
            return str(value).lower() in str(field_val).lower()
        elif operator == "in":
            return field_val in value
        elif operator == "not_in":
            return field_val not in value
        else:
            log.warning("Unknown operator: %s", operator)
            return False
    except (TypeError, ValueError) as exc:
        log.warning("Condition eval error (field=%s op=%s): %s", field, operator, exc)
        return False


# ── rule evaluator ─────────────────────────────────────────────────────────────
def _evaluate_rules(input_data: dict) -> dict:
    """Evaluate all enabled rules against input_data. Returns match summary."""
    with _db() as conn:
        rules = conn.execute(
            "SELECT * FROM logic_rules WHERE enabled=1 ORDER BY priority ASC"
        ).fetchall()

    matched_rules = []
    total_evaluated = len(rules)

    with _db() as conn:
        for rule in rules:
            try:
                conditions = json.loads(rule["conditions"])
            except (json.JSONDecodeError, TypeError):
                conditions = {}

            try:
                actions = json.loads(rule["actions"])
            except (json.JSONDecodeError, TypeError):
                actions = []

            matched = _eval_condition(conditions, input_data)
            actions_taken = []

            if matched:
                for action in actions:
                    actions_taken.append(action)
                    conn.execute(
                        """INSERT INTO decision_log
                           (rule_id, input_data, matched, action_taken, result, created_at)
                           VALUES (?,?,?,?,?,?)""",
                        (
                            rule["id"],
                            json.dumps(input_data),
                            1,
                            json.dumps(action),
                            f"action_type={action.get('type','unknown')}",
                            time.time(),
                        ),
                    )
                conn.execute(
                    "UPDATE logic_rules SET trigger_count=trigger_count+1 WHERE id=?",
                    (rule["id"],),
                )
                matched_rules.append({
                    "rule_id": rule["id"],
                    "name": rule["name"],
                    "priority": rule["priority"],
                    "actions_taken": actions_taken,
                })
            else:
                conn.execute(
                    """INSERT INTO decision_log
                       (rule_id, input_data, matched, action_taken, result, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        rule["id"],
                        json.dumps(input_data),
                        0,
                        "none",
                        "no_match",
                        time.time(),
                    ),
                )

    return {"matched_rules": matched_rules, "total_evaluated": total_evaluated}


# ── committee logic ────────────────────────────────────────────────────────────
def _ask_member(member_key: str, question: str, context: str) -> tuple[str, str]:
    """Ask a single committee member. Returns (member_key, response)."""
    member = COMMITTEE_MEMBERS.get(member_key)
    if not member:
        return member_key, f"[ERROR] Unknown member: {member_key}"

    messages = []
    if context:
        messages.append({
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide a concise, direct answer.",
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nProvide a concise, direct answer.",
        })

    response = _call_model(member["provider"], member["model"], messages, max_tokens=500)
    return member_key, response


def _strategy_vote(responses: dict) -> tuple[str, float]:
    """Simple majority vote by response similarity bucketing."""
    if not responses:
        return "No responses", 0.0

    # Normalise responses to first sentence for comparison
    buckets: dict[str, list[str]] = {}
    for key, resp in responses.items():
        first_sent = resp.split(".")[0].strip().lower()[:80]
        matched = False
        for bucket_key in buckets:
            if _text_similarity(first_sent, bucket_key) > 0.5:
                buckets[bucket_key].append(key)
                matched = True
                break
        if not matched:
            buckets[first_sent] = [key]

    winner_bucket = max(buckets, key=lambda k: len(buckets[k]))
    winner_member = buckets[winner_bucket][0]
    vote_count = len(buckets[winner_bucket])
    total = len(responses)
    confidence = vote_count / total if total > 0 else 0.0

    return responses[winner_member], confidence


def _text_similarity(a: str, b: str) -> float:
    """Rough word overlap similarity between two strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    return len(overlap) / max(len(words_a), len(words_b))


def _strategy_consensus(responses: dict) -> tuple[str, float, str]:
    """Check if all responses agree; synthesise if not. Returns (answer, confidence, consensus)."""
    if not responses:
        return "No responses", 0.0, "none"

    values = list(responses.values())
    # Check similarity between all pairs
    all_agree = True
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            sim = _text_similarity(
                values[i].lower()[:100], values[j].lower()[:100]
            )
            if sim < 0.3:
                all_agree = False
                break
        if not all_agree:
            break

    if all_agree:
        return values[0], 1.0, "full_consensus"

    # Synthesise via claude_haiku
    synthesis_prompt = (
        "Multiple AI models were asked the same question. Their responses are:\n\n"
        + "\n\n".join(f"Model {k}:\n{v}" for k, v in responses.items())
        + "\n\nSynthesize these into a single, balanced final answer."
    )
    messages = [{"role": "user", "content": synthesis_prompt}]
    member = COMMITTEE_MEMBERS["claude_haiku"]
    final = _call_model(member["provider"], member["model"], messages, max_tokens=600)
    return final, 0.6, "synthesized"


def _strategy_synthesis(responses: dict, question: str) -> tuple[str, float]:
    """Send all responses to Claude Haiku for synthesis."""
    synthesis_prompt = (
        f"Original question: {question}\n\n"
        "Multiple AI committee members answered this question. Their responses:\n\n"
        + "\n\n".join(f"Member {k}:\n{v}" for k, v in responses.items())
        + "\n\nProvide the best possible final synthesized answer."
    )
    messages = [{"role": "user", "content": synthesis_prompt}]
    member = COMMITTEE_MEMBERS["claude_haiku"]
    final = _call_model(member["provider"], member["model"], messages, max_tokens=700)
    return final, 0.8


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    def _send(self, code: int, body: Any) -> None:
        data = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
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
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]

        # ── /health ──────────────────────────────────────────────────────────
        if path == "/health":
            self._send(200, {"status": "ok", "service": "fm-logic-bucket", "port": PORT})
            return

        # ── /rules ───────────────────────────────────────────────────────────
        if path == "/rules":
            with _db() as conn:
                rows = conn.execute(
                    "SELECT * FROM logic_rules ORDER BY priority ASC"
                ).fetchall()
            rules = []
            for r in rows:
                rules.append({
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "rule_type": r["rule_type"],
                    "conditions": json.loads(r["conditions"] or "{}"),
                    "actions": json.loads(r["actions"] or "[]"),
                    "priority": r["priority"],
                    "enabled": bool(r["enabled"]),
                    "trigger_count": r["trigger_count"],
                    "created_at": r["created_at"],
                })
            self._send(200, rules)
            return

        # ── /sessions ────────────────────────────────────────────────────────
        if path == "/sessions":
            with _db() as conn:
                rows = conn.execute(
                    """SELECT id, substr(question,1,80) as question_summary,
                       strategy, confidence, created_at
                       FROM committee_sessions ORDER BY created_at DESC LIMIT 100"""
                ).fetchall()
            sessions = [dict(r) for r in rows]
            self._send(200, sessions)
            return

        # ── /sessions/{id} ───────────────────────────────────────────────────
        if path.startswith("/sessions/"):
            try:
                session_id = int(path.split("/")[-1])
            except ValueError:
                self._send(400, {"error": "Invalid session id"})
                return
            with _db() as conn:
                row = conn.execute(
                    "SELECT * FROM committee_sessions WHERE id=?", (session_id,)
                ).fetchone()
            if not row:
                self._send(404, {"error": "Session not found"})
                return
            s = dict(row)
            for field in ("members", "responses"):
                if s.get(field):
                    try:
                        s[field] = json.loads(s[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            self._send(200, s)
            return

        # ── /training/examples ───────────────────────────────────────────────
        if path == "/training/examples":
            with _db() as conn:
                rows = conn.execute(
                    "SELECT * FROM training_examples ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
            self._send(200, [dict(r) for r in rows])
            return

        # ── /analytics ───────────────────────────────────────────────────────
        if path == "/analytics":
            with _db() as conn:
                rules_total = conn.execute(
                    "SELECT COUNT(*) FROM logic_rules"
                ).fetchone()[0]
                rules_enabled = conn.execute(
                    "SELECT COUNT(*) FROM logic_rules WHERE enabled=1"
                ).fetchone()[0]
                sessions_total = conn.execute(
                    "SELECT COUNT(*) FROM committee_sessions"
                ).fetchone()[0]
                avg_conf = conn.execute(
                    "SELECT AVG(confidence) FROM committee_sessions"
                ).fetchone()[0]
                model_acc = conn.execute(
                    """SELECT model,
                          AVG(score) as avg_score,
                          COUNT(*) as total
                       FROM training_examples
                       GROUP BY model"""
                ).fetchall()

            training_by_model = {
                r["model"]: {"avg_score": round(r["avg_score"] or 0, 4), "total": r["total"]}
                for r in model_acc
            }
            self._send(200, {
                "rules": {
                    "total": rules_total,
                    "enabled": rules_enabled,
                    "disabled": rules_total - rules_enabled,
                },
                "committee_sessions": {
                    "total": sessions_total,
                    "avg_confidence": round(avg_conf or 0, 4),
                },
                "training_by_model": training_by_model,
            })
            return

        self._send(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_body()

        # ── /committee/ask ───────────────────────────────────────────────────
        if path == "/committee/ask":
            question  = body.get("question", "").strip()
            context   = body.get("context", "")
            members   = body.get("members", list(COMMITTEE_MEMBERS.keys()))
            strategy  = body.get("strategy", "synthesis")

            if not question:
                self._send(400, {"error": "question is required"})
                return

            valid_members = [m for m in members if m in COMMITTEE_MEMBERS]
            if not valid_members:
                self._send(400, {"error": "No valid committee members specified"})
                return

            # Ask all members in parallel
            futures = {
                _pool.submit(_ask_member, m, question, context): m
                for m in valid_members
            }
            responses: dict[str, str] = {}
            for fut in concurrent.futures.as_completed(futures, timeout=45):
                member_key, response = fut.result()
                responses[member_key] = response

            # Apply strategy
            consensus_label = ""
            if strategy == "vote":
                final_answer, confidence = _strategy_vote(responses)
            elif strategy == "consensus":
                final_answer, confidence, consensus_label = _strategy_consensus(responses)
            else:  # synthesis (default)
                final_answer, confidence = _strategy_synthesis(responses, question)

            # Store session
            with _db() as conn:
                cursor = conn.execute(
                    """INSERT INTO committee_sessions
                       (question, context, members, strategy, consensus, responses, final_answer, confidence, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        question,
                        context,
                        json.dumps(valid_members),
                        strategy,
                        consensus_label,
                        json.dumps(responses),
                        final_answer,
                        confidence,
                        time.time(),
                    ),
                )
                session_id = cursor.lastrowid

            self._send(200, {
                "session_id": session_id,
                "question": question,
                "responses": responses,
                "final_answer": final_answer,
                "confidence": round(confidence, 4),
                "strategy": strategy,
            })
            return

        # ── /committee/vote ──────────────────────────────────────────────────
        if path == "/committee/vote":
            question = body.get("question", "").strip()
            options  = body.get("options", [])
            members  = body.get("members", list(COMMITTEE_MEMBERS.keys()))

            if not question or not options:
                self._send(400, {"error": "question and options are required"})
                return

            valid_members = [m for m in members if m in COMMITTEE_MEMBERS]
            options_str = ", ".join(f'"{o}"' for o in options)
            vote_question = (
                f"{question}\n\nYou must pick EXACTLY one of these options: [{options_str}]. "
                "Reply with only the option text, nothing else."
            )

            futures = {
                _pool.submit(_ask_member, m, vote_question, ""): m
                for m in valid_members
            }
            votes: dict[str, int] = {o: 0 for o in options}
            member_votes: dict[str, str] = {}

            for fut in concurrent.futures.as_completed(futures, timeout=45):
                member_key, response = fut.result()
                # Match response to closest option
                matched_option = None
                response_lower = response.strip().lower()
                for opt in options:
                    if opt.lower() in response_lower or response_lower in opt.lower():
                        matched_option = opt
                        break
                if matched_option is None:
                    # Pick best overlap
                    best_sim = -1.0
                    for opt in options:
                        sim = _text_similarity(response_lower, opt.lower())
                        if sim > best_sim:
                            best_sim = sim
                            matched_option = opt
                if matched_option:
                    votes[matched_option] = votes.get(matched_option, 0) + 1
                    member_votes[member_key] = matched_option

            max_votes = max(votes.values()) if votes else 0
            winners = [o for o, v in votes.items() if v == max_votes]
            winner = winners[0] if winners else (options[0] if options else "")
            tied = len(winners) > 1

            self._send(200, {
                "winner": winner,
                "votes": votes,
                "member_votes": member_votes,
                "tied": tied,
            })
            return

        # ── /rules/create ────────────────────────────────────────────────────
        if path == "/rules/create":
            name        = body.get("name", "").strip()
            description = body.get("description", "")
            rule_type   = body.get("rule_type", "custom")
            conditions  = body.get("conditions", {})
            actions     = body.get("actions", [])
            priority    = body.get("priority", 5)

            if not name:
                self._send(400, {"error": "name is required"})
                return

            try:
                with _db() as conn:
                    cursor = conn.execute(
                        """INSERT INTO logic_rules
                           (name, description, rule_type, conditions, actions, priority, enabled, created_at)
                           VALUES (?,?,?,?,?,?,1,?)""",
                        (
                            name,
                            description,
                            rule_type,
                            json.dumps(conditions),
                            json.dumps(actions),
                            priority,
                            time.time(),
                        ),
                    )
                    rule_id = cursor.lastrowid
                self._send(201, {"rule_id": rule_id})
            except sqlite3.IntegrityError:
                self._send(409, {"error": f"Rule name '{name}' already exists"})
            return

        # ── /rules/evaluate ──────────────────────────────────────────────────
        if path == "/rules/evaluate":
            input_data = body.get("data", body)
            result = _evaluate_rules(input_data)
            self._send(200, result)
            return

        # ── /rules/batch_evaluate ────────────────────────────────────────────
        if path == "/rules/batch_evaluate":
            dataset = body.get("dataset", "leads")
            limit   = int(body.get("limit", 100))

            try:
                with _db() as conn:
                    rows = conn.execute(
                        f"SELECT * FROM {dataset} LIMIT ?", (limit,)
                    ).fetchall()
            except sqlite3.OperationalError as exc:
                self._send(400, {"error": f"Cannot read dataset '{dataset}': {exc}"})
                return

            matches_by_rule: dict[str, int] = {}
            evaluated = 0

            for row in rows:
                row_data = dict(row)
                result = _evaluate_rules(row_data)
                evaluated += 1
                for matched in result["matched_rules"]:
                    rule_name = matched["name"]
                    matches_by_rule[rule_name] = matches_by_rule.get(rule_name, 0) + 1

            self._send(200, {
                "evaluated": evaluated,
                "matches_by_rule": matches_by_rule,
            })
            return

        # ── /training/add ────────────────────────────────────────────────────
        if path == "/training/add":
            input_text      = body.get("input_text", "").strip()
            expected_output = body.get("expected_output", "").strip()
            model_key       = body.get("model", "claude_haiku")
            feedback        = body.get("feedback", "")

            if not input_text:
                self._send(400, {"error": "input_text is required"})
                return

            # Optionally query model and score
            actual_output = ""
            score = 0.0

            member = COMMITTEE_MEMBERS.get(model_key)
            if member and expected_output:
                messages = [{"role": "user", "content": input_text}]
                actual_output = _call_model(
                    member["provider"], member["model"], messages, max_tokens=200
                )
                score = 1.0 if (
                    expected_output.lower().strip() in actual_output.lower()
                    or actual_output.lower().strip() == expected_output.lower().strip()
                ) else 0.0

            with _db() as conn:
                cursor = conn.execute(
                    """INSERT INTO training_examples
                       (input_text, expected_output, actual_output, model, score, feedback, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        input_text,
                        expected_output,
                        actual_output,
                        model_key,
                        score,
                        feedback,
                        time.time(),
                    ),
                )
                example_id = cursor.lastrowid

            self._send(201, {
                "example_id": example_id,
                "score": score,
                "actual_output": actual_output,
            })
            return

        # ── /training/eval ───────────────────────────────────────────────────
        if path == "/training/eval":
            model_key = body.get("model", "claude_haiku")
            limit     = int(body.get("limit", 20))

            member = COMMITTEE_MEMBERS.get(model_key)
            if not member:
                self._send(400, {"error": f"Unknown model: {model_key}"})
                return

            with _db() as conn:
                rows = conn.execute(
                    """SELECT * FROM training_examples WHERE model=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (model_key, limit),
                ).fetchall()

            if not rows:
                self._send(200, {
                    "model": model_key,
                    "accuracy": 0.0,
                    "avg_score": 0.0,
                    "examples_tested": 0,
                })
                return

            exact_matches = 0
            scores = []

            for row in rows:
                messages = [{"role": "user", "content": row["input_text"]}]
                actual = _call_model(
                    member["provider"], member["model"], messages, max_tokens=200
                )
                expected = (row["expected_output"] or "").lower().strip()
                actual_lower = actual.lower().strip()
                hit = 1.0 if (expected in actual_lower or actual_lower == expected) else 0.0
                exact_matches += int(hit == 1.0)
                scores.append(hit)

                # Update record
                with _db() as conn:
                    conn.execute(
                        "UPDATE training_examples SET actual_output=?, score=? WHERE id=?",
                        (actual, hit, row["id"]),
                    )

            n = len(rows)
            accuracy = exact_matches / n if n > 0 else 0.0
            avg_score = sum(scores) / n if n > 0 else 0.0

            self._send(200, {
                "model": model_key,
                "accuracy": round(accuracy, 4),
                "avg_score": round(avg_score, 4),
                "examples_tested": n,
            })
            return

        self._send(404, {"error": "Not found"})

    def do_PUT(self):
        path = self.path.split("?")[0]

        # ── PUT /rules/{id} ──────────────────────────────────────────────────
        if path.startswith("/rules/"):
            try:
                rule_id = int(path.split("/")[-1])
            except ValueError:
                self._send(400, {"error": "Invalid rule id"})
                return

            body = self._read_body()
            with _db() as conn:
                existing = conn.execute(
                    "SELECT * FROM logic_rules WHERE id=?", (rule_id,)
                ).fetchone()
                if not existing:
                    self._send(404, {"error": "Rule not found"})
                    return

                fields = []
                values = []
                for field in ("name", "description", "rule_type", "priority", "enabled"):
                    if field in body:
                        fields.append(f"{field}=?")
                        values.append(body[field])
                for field in ("conditions", "actions"):
                    if field in body:
                        fields.append(f"{field}=?")
                        values.append(json.dumps(body[field]))

                if not fields:
                    self._send(400, {"error": "No fields to update"})
                    return

                values.append(rule_id)
                conn.execute(
                    f"UPDATE logic_rules SET {', '.join(fields)} WHERE id=?",
                    values,
                )

            self._send(200, {"updated": True, "rule_id": rule_id})
            return

        self._send(404, {"error": "Not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0]

        # ── DELETE /rules/{id} ───────────────────────────────────────────────
        if path.startswith("/rules/"):
            try:
                rule_id = int(path.split("/")[-1])
            except ValueError:
                self._send(400, {"error": "Invalid rule id"})
                return

            with _db() as conn:
                existing = conn.execute(
                    "SELECT id FROM logic_rules WHERE id=?", (rule_id,)
                ).fetchone()
                if not existing:
                    self._send(404, {"error": "Rule not found"})
                    return
                conn.execute("DELETE FROM logic_rules WHERE id=?", (rule_id,))

            self._send(200, {"deleted": True, "rule_id": rule_id})
            return

        self._send(404, {"error": "Not found"})


# ── entrypoint ────────────────────────────────────────────────────────────────
def main():
    log.info("Initialising database at %s", DB)
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    log.info("fm-logic-bucket listening on port %s", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
