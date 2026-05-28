"""
FractalMesh OMEGA Titan — Gamification & Achievement Engine
Port: 7895
Samuel James Hiotis | ABN 56 628 117 363

Full gamification engine: badges, achievements, challenges, leaderboards.
Tracks player progress, awards XP and badges automatically when conditions
are met. Supports events (login, purchase, review, referral, etc.).
"""

import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Vault loading — MUST be before any os.getenv calls
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / "fmsaas"
DB_PATH  = BASE_DIR / "database" / "sovereign.db"
LOG_PATH = BASE_DIR / "logs" / "fm_gamification.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

PORT               = int(os.environ.get("GAMIFICATION_PORT", "7895"))
SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM      = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "")

# ---------------------------------------------------------------------------
# XP map — points awarded per event type
# ---------------------------------------------------------------------------
XP_MAP = {
    "login":           5,
    "purchase":        50,
    "review":          20,
    "referral":        100,
    "streak":          10,
    "lesson_complete": 30,
    "course_complete": 200,
}

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    line = f"[{ts}] [{level}] {msg}\n"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as fh:
                fh.write(line)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# XP → Level formula
# ---------------------------------------------------------------------------
def xp_to_level(xp: int) -> int:
    """level = 1 + int(sqrt(xp / 100))"""
    return 1 + int(math.sqrt(max(xp, 0) / 100))

def level_xp_bounds(level: int):
    """Return (xp_start, xp_end) for a given level."""
    if level <= 1:
        start = 0
    else:
        start = (level - 1) ** 2 * 100
    end = level ** 2 * 100
    return start, end

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            id           INTEGER PRIMARY KEY,
            player_id    TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            email        TEXT,
            xp           INTEGER DEFAULT 0,
            level        INTEGER DEFAULT 1,
            total_badges INTEGER DEFAULT 0,
            rank         INTEGER DEFAULT 0,
            streak_days  INTEGER DEFAULT 0,
            last_active  REAL,
            joined_at    REAL,
            updated_at   REAL
        );

        CREATE TABLE IF NOT EXISTS badges (
            id              INTEGER PRIMARY KEY,
            badge_id        TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            description     TEXT,
            icon            TEXT,
            category        TEXT,
            rarity          TEXT DEFAULT 'common',
            xp_reward       INTEGER DEFAULT 0,
            condition_type  TEXT,
            condition_value INTEGER,
            active          INTEGER DEFAULT 1,
            created_at      REAL
        );

        CREATE TABLE IF NOT EXISTS player_badges (
            id         INTEGER PRIMARY KEY,
            player_id  TEXT NOT NULL,
            badge_id   TEXT NOT NULL,
            awarded_at REAL,
            UNIQUE(player_id, badge_id)
        );

        CREATE TABLE IF NOT EXISTS challenges (
            id              INTEGER PRIMARY KEY,
            challenge_id    TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            description     TEXT,
            xp_reward       INTEGER DEFAULT 0,
            badge_id        TEXT,
            start_time      REAL,
            end_time        REAL,
            condition_type  TEXT,
            condition_value INTEGER,
            active          INTEGER DEFAULT 1,
            created_at      REAL
        );

        CREATE TABLE IF NOT EXISTS player_challenges (
            id           INTEGER PRIMARY KEY,
            player_id    TEXT NOT NULL,
            challenge_id TEXT NOT NULL,
            progress     INTEGER DEFAULT 0,
            completed    INTEGER DEFAULT 0,
            completed_at REAL,
            UNIQUE(player_id, challenge_id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY,
            event_id     TEXT UNIQUE NOT NULL,
            player_id    TEXT NOT NULL,
            event_type   TEXT NOT NULL,
            value        INTEGER DEFAULT 1,
            metadata     TEXT,
            xp_awarded   INTEGER DEFAULT 0,
            processed_at REAL,
            created_at   REAL
        );
    """)

    conn.commit()
    conn.close()
    _log("INFO", "Database initialized")

# ---------------------------------------------------------------------------
# Seed default badges
# ---------------------------------------------------------------------------
SEED_BADGES = [
    {
        "badge_id":        "first_steps",
        "name":            "First Steps",
        "description":     "Earn your first 10 XP",
        "icon":            "🥾",
        "category":        "general",
        "rarity":          "common",
        "xp_reward":       10,
        "condition_type":  "xp_threshold",
        "condition_value": 10,
    },
    {
        "badge_id":        "regular_user",
        "name":            "Regular User",
        "description":     "Login 10 times",
        "icon":            "📅",
        "category":        "login",
        "rarity":          "common",
        "xp_reward":       25,
        "condition_type":  "event_count",
        "condition_value": 10,
    },
    {
        "badge_id":        "spender",
        "name":            "Spender",
        "description":     "Make 5 purchases",
        "icon":            "💳",
        "category":        "purchase",
        "rarity":          "rare",
        "xp_reward":       100,
        "condition_type":  "purchase_count",
        "condition_value": 5,
    },
    {
        "badge_id":        "week_warrior",
        "name":            "Week Warrior",
        "description":     "Maintain a 7-day streak",
        "icon":            "🔥",
        "category":        "streak",
        "rarity":          "uncommon",
        "xp_reward":       50,
        "condition_type":  "streak",
        "condition_value": 7,
    },
    {
        "badge_id":        "century_club",
        "name":            "Century Club",
        "description":     "Accumulate 1000 XP",
        "icon":            "💯",
        "category":        "general",
        "rarity":          "epic",
        "xp_reward":       500,
        "condition_type":  "xp_threshold",
        "condition_value": 1000,
    },
    {
        "badge_id":        "course_graduate",
        "name":            "Course Graduate",
        "description":     "Complete your first course",
        "icon":            "🎓",
        "category":        "course_complete",
        "rarity":          "rare",
        "xp_reward":       150,
        "condition_type":  "event_count",
        "condition_value": 1,
    },
]

def seed_badges() -> None:
    conn = get_db()
    cur = conn.cursor()
    count = cur.execute("SELECT COUNT(*) FROM badges").fetchone()[0]
    if count == 0:
        now = time.time()
        for b in SEED_BADGES:
            bid = b["badge_id"]
            cur.execute("""
                INSERT OR IGNORE INTO badges
                    (badge_id, name, description, icon, category, rarity,
                     xp_reward, condition_type, condition_value, active, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,1,?)
            """, (
                bid, b["name"], b["description"], b["icon"], b["category"],
                b["rarity"], b["xp_reward"], b["condition_type"],
                b["condition_value"], now,
            ))
        conn.commit()
        _log("INFO", f"Seeded {len(SEED_BADGES)} default badges")
    conn.close()

# ---------------------------------------------------------------------------
# SendGrid email helper
# ---------------------------------------------------------------------------
def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    if not SENDGRID_API_KEY or not to_email:
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from":             {"email": SENDGRID_FROM},
        "subject":          subject,
        "content":          [{"type": "text/html", "value": html_body}],
    }).encode()
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except urllib.error.URLError as exc:
        _log("WARN", f"SendGrid error: {exc}")
        return False

def send_welcome_email(display_name: str, email: str) -> None:
    subject = "Welcome to FractalMesh — Your Adventure Begins!"
    body = f"""
    <h2>Welcome, {display_name}!</h2>
    <p>You've joined the FractalMesh gamification platform. Start earning XP,
    collecting badges, and climbing the leaderboard.</p>
    <p>Good luck on your journey!</p>
    <p>— The FractalMesh Team</p>
    """
    _send_email(email, subject, body)

def send_badge_email(display_name: str, email: str, badge_name: str, badge_icon: str, xp_reward: int) -> None:
    subject = f"Badge Unlocked: {badge_name}!"
    body = f"""
    <h2>Congratulations, {display_name}!</h2>
    <p>You've earned the <strong>{badge_icon} {badge_name}</strong> badge!</p>
    <p>This badge comes with a reward of <strong>{xp_reward} XP</strong>.</p>
    <p>Keep up the great work!</p>
    <p>— The FractalMesh Team</p>
    """
    _send_email(email, subject, body)

def send_level_up_email(display_name: str, email: str, new_level: int) -> None:
    subject = f"Level Up! You're now Level {new_level}!"
    body = f"""
    <h2>Level Up, {display_name}!</h2>
    <p>You've reached <strong>Level {new_level}</strong> on FractalMesh!</p>
    <p>Keep earning XP to unlock even more rewards and recognition.</p>
    <p>— The FractalMesh Team</p>
    """
    _send_email(email, subject, body)

def send_challenge_complete_email(display_name: str, email: str, challenge_name: str, xp_reward: int) -> None:
    subject = f"Challenge Complete: {challenge_name}!"
    body = f"""
    <h2>Challenge Complete, {display_name}!</h2>
    <p>You've completed the challenge: <strong>{challenge_name}</strong>!</p>
    <p>You've been awarded <strong>{xp_reward} XP</strong>.</p>
    <p>— The FractalMesh Team</p>
    """
    _send_email(email, subject, body)

# ---------------------------------------------------------------------------
# Core game logic helpers
# ---------------------------------------------------------------------------
def _get_player(cur: sqlite3.Cursor, player_id: str) -> sqlite3.Row | None:
    return cur.execute(
        "SELECT * FROM players WHERE player_id=?", (player_id,)
    ).fetchone()

def _award_xp(conn: sqlite3.Connection, cur: sqlite3.Cursor, player_id: str, xp: int) -> dict:
    """Add XP to player. Returns {'old_level': int, 'new_level': int, 'leveled_up': bool}."""
    player = _get_player(cur, player_id)
    if not player:
        return {"old_level": 1, "new_level": 1, "leveled_up": False}

    old_xp    = player["xp"]
    new_xp    = old_xp + xp
    old_level = xp_to_level(old_xp)
    new_level = xp_to_level(new_xp)

    cur.execute(
        "UPDATE players SET xp=?, level=?, updated_at=? WHERE player_id=?",
        (new_xp, new_level, time.time(), player_id),
    )
    conn.commit()

    return {
        "old_level":  old_level,
        "new_level":  new_level,
        "leveled_up": new_level > old_level,
    }

def _check_and_award_badges(
    conn: sqlite3.Connection,
    cur: sqlite3.Cursor,
    player_id: str,
) -> list[dict]:
    """Check all active badges and award any that the player qualifies for."""
    player = _get_player(cur, player_id)
    if not player:
        return []

    earned = cur.execute(
        "SELECT badge_id FROM player_badges WHERE player_id=?", (player_id,)
    ).fetchall()
    earned_ids = {row["badge_id"] for row in earned}

    badges = cur.execute(
        "SELECT * FROM badges WHERE active=1"
    ).fetchall()

    newly_awarded = []
    xp_bonus      = 0

    for badge in badges:
        if badge["badge_id"] in earned_ids:
            continue

        qualifies = False
        ct = badge["condition_type"]
        cv = badge["condition_value"]

        if ct == "xp_threshold":
            qualifies = player["xp"] >= cv

        elif ct == "event_count":
            category = badge["category"]
            count = cur.execute(
                "SELECT COUNT(*) FROM events WHERE player_id=? AND event_type=?",
                (player_id, category),
            ).fetchone()[0]
            qualifies = count >= cv

        elif ct == "streak":
            qualifies = player["streak_days"] >= cv

        elif ct == "purchase_count":
            count = cur.execute(
                "SELECT COUNT(*) FROM events WHERE player_id=? AND event_type='purchase'",
                (player_id,),
            ).fetchone()[0]
            qualifies = count >= cv

        if qualifies:
            now = time.time()
            cur.execute(
                "INSERT OR IGNORE INTO player_badges (player_id, badge_id, awarded_at) VALUES (?,?,?)",
                (player_id, badge["badge_id"], now),
            )
            # Update total_badges count
            cur.execute(
                "UPDATE players SET total_badges=total_badges+1, updated_at=? WHERE player_id=?",
                (now, player_id),
            )
            xp_bonus += badge["xp_reward"]
            newly_awarded.append(dict(badge))
            _log("INFO", f"Badge awarded: {badge['badge_id']} -> {player_id}")

    conn.commit()

    # Award XP for badges (if any) and re-check levels
    if xp_bonus > 0:
        _award_xp(conn, cur, player_id, xp_bonus)

    return newly_awarded

def _check_challenges(
    conn: sqlite3.Connection,
    cur: sqlite3.Cursor,
    player_id: str,
    event_type: str,
) -> list[dict]:
    """Update challenge progress and complete any that qualify."""
    now = time.time()
    active_challenges = cur.execute(
        "SELECT * FROM challenges WHERE active=1 AND (end_time IS NULL OR end_time > ?)",
        (now,),
    ).fetchall()

    completed_challenges = []

    for ch in active_challenges:
        row = cur.execute(
            "SELECT * FROM player_challenges WHERE player_id=? AND challenge_id=?",
            (player_id, ch["challenge_id"]),
        ).fetchone()

        if row and row["completed"]:
            continue  # already done

        # Determine if this event counts toward the challenge
        ct = ch["condition_type"]
        cv = ch["condition_value"]
        contributes = False

        if ct == "event_count":
            contributes = (event_type == ch["condition_type"] or True)
            # Count relevant events for this challenge's condition_type
            # Use challenge category as event_type filter if exists
            count = cur.execute(
                "SELECT COUNT(*) FROM events WHERE player_id=? AND event_type=?",
                (player_id, event_type),
            ).fetchone()[0]
            if not row:
                cur.execute(
                    """INSERT OR IGNORE INTO player_challenges
                       (player_id, challenge_id, progress, completed) VALUES (?,?,?,0)""",
                    (player_id, ch["challenge_id"], 0),
                )
                conn.commit()

            # Refresh row
            row = cur.execute(
                "SELECT * FROM player_challenges WHERE player_id=? AND challenge_id=?",
                (player_id, ch["challenge_id"]),
            ).fetchone()

            new_progress = (row["progress"] if row else 0) + 1
            cur.execute(
                "UPDATE player_challenges SET progress=? WHERE player_id=? AND challenge_id=?",
                (new_progress, player_id, ch["challenge_id"]),
            )

            if new_progress >= cv:
                contributes = True
                cur.execute(
                    """UPDATE player_challenges
                       SET completed=1, completed_at=?, progress=?
                       WHERE player_id=? AND challenge_id=?""",
                    (now, new_progress, player_id, ch["challenge_id"]),
                )
                _award_xp(conn, cur, player_id, ch["xp_reward"])

                # Award badge if linked
                if ch["badge_id"]:
                    cur.execute(
                        "INSERT OR IGNORE INTO player_badges (player_id, badge_id, awarded_at) VALUES (?,?,?)",
                        (player_id, ch["badge_id"], now),
                    )

                completed_challenges.append(dict(ch))
                _log("INFO", f"Challenge completed: {ch['challenge_id']} -> {player_id}")

        elif ct == "xp_threshold":
            player = _get_player(cur, player_id)
            if player and player["xp"] >= cv:
                if not row:
                    cur.execute(
                        """INSERT OR IGNORE INTO player_challenges
                           (player_id, challenge_id, progress, completed, completed_at)
                           VALUES (?,?,?,1,?)""",
                        (player_id, ch["challenge_id"], cv, now),
                    )
                else:
                    cur.execute(
                        """UPDATE player_challenges SET completed=1, completed_at=?, progress=?
                           WHERE player_id=? AND challenge_id=?""",
                        (now, cv, player_id, ch["challenge_id"]),
                    )
                _award_xp(conn, cur, player_id, ch["xp_reward"])
                completed_challenges.append(dict(ch))
                _log("INFO", f"Challenge completed (xp): {ch['challenge_id']} -> {player_id}")

        elif ct == "purchase_count":
            if event_type == "purchase":
                count = cur.execute(
                    "SELECT COUNT(*) FROM events WHERE player_id=? AND event_type='purchase'",
                    (player_id,),
                ).fetchone()[0]

                if not row:
                    cur.execute(
                        "INSERT OR IGNORE INTO player_challenges (player_id, challenge_id, progress) VALUES (?,?,?)",
                        (player_id, ch["challenge_id"], count),
                    )
                else:
                    cur.execute(
                        "UPDATE player_challenges SET progress=? WHERE player_id=? AND challenge_id=?",
                        (count, player_id, ch["challenge_id"]),
                    )

                if count >= cv:
                    cur.execute(
                        """UPDATE player_challenges SET completed=1, completed_at=?
                           WHERE player_id=? AND challenge_id=? AND completed=0""",
                        (now, player_id, ch["challenge_id"]),
                    )
                    _award_xp(conn, cur, player_id, ch["xp_reward"])
                    completed_challenges.append(dict(ch))

        elif ct == "streak":
            player = _get_player(cur, player_id)
            if player and player["streak_days"] >= cv:
                if not row:
                    cur.execute(
                        """INSERT OR IGNORE INTO player_challenges
                           (player_id, challenge_id, progress, completed, completed_at)
                           VALUES (?,?,?,1,?)""",
                        (player_id, ch["challenge_id"], player["streak_days"], now),
                    )
                else:
                    cur.execute(
                        """UPDATE player_challenges SET completed=1, completed_at=?, progress=?
                           WHERE player_id=? AND challenge_id=?""",
                        (now, player["streak_days"], player_id, ch["challenge_id"]),
                    )
                _award_xp(conn, cur, player_id, ch["xp_reward"])
                completed_challenges.append(dict(ch))

    conn.commit()
    return completed_challenges

# ---------------------------------------------------------------------------
# Security helper
# ---------------------------------------------------------------------------
def _verify_admin(headers) -> bool:
    secret = headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return False
    return hmac.compare_digest(
        secret.encode("utf-8"),
        ADMIN_SECRET.encode("utf-8"),
    )

# ---------------------------------------------------------------------------
# Background daemon — update ranks, expire challenges
# ---------------------------------------------------------------------------
def _background_daemon() -> None:
    while True:
        try:
            _update_ranks()
            _expire_challenges()
        except Exception as exc:
            _log("ERROR", f"Background daemon error: {exc}")
        time.sleep(3600)

def _update_ranks() -> None:
    conn = get_db()
    cur  = conn.cursor()
    players = cur.execute(
        "SELECT player_id FROM players ORDER BY xp DESC"
    ).fetchall()
    now = time.time()
    for rank, row in enumerate(players, start=1):
        cur.execute(
            "UPDATE players SET rank=?, updated_at=? WHERE player_id=?",
            (rank, now, row["player_id"]),
        )
    conn.commit()
    conn.close()
    _log("INFO", f"Leaderboard ranks updated for {len(players)} players")

def _expire_challenges() -> None:
    conn = get_db()
    cur  = conn.cursor()
    now = time.time()
    result = cur.execute(
        "UPDATE challenges SET active=0 WHERE end_time IS NOT NULL AND end_time <= ? AND active=1",
        (now,),
    )
    expired = result.rowcount
    conn.commit()
    conn.close()
    if expired:
        _log("INFO", f"Expired {expired} challenge(s)")

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class GamificationHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        pass

    # ------------------------------------------------------------------ util
    def _send_json(self, status: int, data: dict | list) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _parse_path(self):
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        path, qs = self._parse_path()
        parts = [p for p in path.split("/") if p]

        try:
            # GET /health
            if path == "/health":
                self._send_json(200, {"status": "ok", "service": "fm_gamification", "port": PORT})

            # GET /badges
            elif path == "/badges":
                self._handle_list_badges()

            # GET /challenges
            elif path == "/challenges":
                self._handle_list_challenges()

            # GET /leaderboard/weekly
            elif path == "/leaderboard/weekly":
                self._handle_leaderboard_weekly()

            # GET /leaderboard
            elif path == "/leaderboard":
                self._handle_leaderboard()

            # GET /stats
            elif path == "/stats":
                self._handle_stats()

            # GET /players/{player_id}
            elif len(parts) == 2 and parts[0] == "players":
                self._handle_get_player(parts[1])

            # GET /players/{player_id}/badges
            elif len(parts) == 3 and parts[0] == "players" and parts[2] == "badges":
                self._handle_player_badges(parts[1])

            # GET /players/{player_id}/challenges
            elif len(parts) == 3 and parts[0] == "players" and parts[2] == "challenges":
                self._handle_player_challenges(parts[1])

            else:
                self._send_json(404, {"error": "Not found"})

        except Exception as exc:
            _log("ERROR", f"GET {path} error: {exc}")
            self._send_json(500, {"error": str(exc)})

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        path, _ = self._parse_path()
        parts   = [p for p in path.split("/") if p]

        try:
            # POST /players
            if path == "/players":
                self._handle_create_player()

            # POST /events
            elif path == "/events":
                self._handle_create_event()

            # POST /badges  (admin)
            elif path == "/badges":
                self._handle_create_badge()

            # POST /challenges  (admin)
            elif path == "/challenges":
                self._handle_create_challenge()

            # POST /streak
            elif path == "/streak":
                self._handle_streak()

            else:
                self._send_json(404, {"error": "Not found"})

        except Exception as exc:
            _log("ERROR", f"POST {path} error: {exc}")
            self._send_json(500, {"error": str(exc)})

    # ================================================================ handlers

    # ------------------------------------------------------------------ GET /health
    # (handled inline above)

    # ------------------------------------------------------------------ GET /badges
    def _handle_list_badges(self):
        conn = get_db()
        cur  = conn.cursor()
        rows = cur.execute("SELECT * FROM badges ORDER BY rarity, name").fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    # ------------------------------------------------------------------ GET /challenges
    def _handle_list_challenges(self):
        now  = time.time()
        conn = get_db()
        cur  = conn.cursor()
        rows = cur.execute(
            "SELECT * FROM challenges WHERE active=1 AND (end_time IS NULL OR end_time>?) ORDER BY created_at DESC",
            (now,),
        ).fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    # ------------------------------------------------------------------ GET /leaderboard
    def _handle_leaderboard(self):
        conn = get_db()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT player_id, display_name, xp, level, total_badges, rank
               FROM players ORDER BY xp DESC LIMIT 50"""
        ).fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    # ------------------------------------------------------------------ GET /leaderboard/weekly
    def _handle_leaderboard_weekly(self):
        week_start = time.time() - 7 * 86400
        conn = get_db()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT e.player_id, p.display_name, SUM(e.xp_awarded) AS weekly_xp
               FROM events e
               JOIN players p ON p.player_id = e.player_id
               WHERE e.created_at >= ?
               GROUP BY e.player_id
               ORDER BY weekly_xp DESC
               LIMIT 20""",
            (week_start,),
        ).fetchall()
        conn.close()
        result = []
        for rank, row in enumerate(rows, start=1):
            d = dict(row)
            d["rank"] = rank
            result.append(d)
        self._send_json(200, result)

    # ------------------------------------------------------------------ GET /stats
    def _handle_stats(self):
        if not _verify_admin(self.headers):
            self._send_json(403, {"error": "Forbidden"})
            return
        conn = get_db()
        cur  = conn.cursor()
        total_players = cur.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        total_xp      = cur.execute("SELECT COALESCE(SUM(xp), 0) FROM players").fetchone()[0]
        total_badges  = cur.execute("SELECT COUNT(*) FROM player_badges").fetchone()[0]
        popular_row   = cur.execute(
            """SELECT b.name, COUNT(pb.badge_id) AS cnt
               FROM player_badges pb JOIN badges b ON b.badge_id=pb.badge_id
               GROUP BY pb.badge_id ORDER BY cnt DESC LIMIT 1"""
        ).fetchone()
        popular_badge = dict(popular_row) if popular_row else None
        conn.close()
        self._send_json(200, {
            "total_players":    total_players,
            "total_xp_distributed": total_xp,
            "total_badges_awarded": total_badges,
            "most_popular_badge": popular_badge,
        })

    # ------------------------------------------------------------------ GET /players/{id}
    def _handle_get_player(self, player_id: str):
        conn = get_db()
        cur  = conn.cursor()
        player = _get_player(cur, player_id)
        if not player:
            conn.close()
            self._send_json(404, {"error": "Player not found"})
            return

        badges = cur.execute(
            """SELECT b.*, pb.awarded_at FROM badges b
               JOIN player_badges pb ON pb.badge_id=b.badge_id
               WHERE pb.player_id=? ORDER BY pb.awarded_at DESC""",
            (player_id,),
        ).fetchall()

        now = time.time()
        active_challenges = cur.execute(
            """SELECT c.*, pc.progress, pc.completed, pc.completed_at
               FROM challenges c
               LEFT JOIN player_challenges pc
                   ON pc.challenge_id=c.challenge_id AND pc.player_id=?
               WHERE c.active=1 AND (c.end_time IS NULL OR c.end_time>?)
               ORDER BY c.created_at DESC""",
            (player_id, now),
        ).fetchall()

        p = dict(player)
        lv  = p["level"]
        xp  = p["xp"]
        xp_start, xp_end = level_xp_bounds(lv)
        p["level_progress"] = {
            "current_level": lv,
            "xp_for_level":  xp - xp_start,
            "xp_needed":     xp_end - xp_start,
            "next_level_xp": xp_end,
        }
        p["badges"]           = [dict(b) for b in badges]
        p["active_challenges"] = [dict(c) for c in active_challenges]

        conn.close()
        self._send_json(200, p)

    # ------------------------------------------------------------------ GET /players/{id}/badges
    def _handle_player_badges(self, player_id: str):
        conn = get_db()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT b.*, pb.awarded_at FROM badges b
               JOIN player_badges pb ON pb.badge_id=b.badge_id
               WHERE pb.player_id=? ORDER BY pb.awarded_at DESC""",
            (player_id,),
        ).fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    # ------------------------------------------------------------------ GET /players/{id}/challenges
    def _handle_player_challenges(self, player_id: str):
        conn = get_db()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT c.*, pc.progress, pc.completed, pc.completed_at
               FROM challenges c
               LEFT JOIN player_challenges pc
                   ON pc.challenge_id=c.challenge_id AND pc.player_id=?
               ORDER BY c.created_at DESC""",
            (player_id,),
        ).fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    # ------------------------------------------------------------------ POST /players
    def _handle_create_player(self):
        body = self._read_json()
        player_id    = body.get("player_id", "").strip()
        display_name = body.get("display_name", "").strip()
        email        = body.get("email", "").strip()

        if not player_id or not display_name:
            self._send_json(400, {"error": "player_id and display_name required"})
            return

        now  = time.time()
        conn = get_db()
        cur  = conn.cursor()

        existing = _get_player(cur, player_id)
        if existing:
            conn.close()
            self._send_json(409, {"error": "Player already exists"})
            return

        cur.execute(
            """INSERT INTO players
               (player_id, display_name, email, xp, level, total_badges, rank,
                streak_days, last_active, joined_at, updated_at)
               VALUES (?,?,?,0,1,0,0,0,?,?,?)""",
            (player_id, display_name, email or None, now, now, now),
        )
        conn.commit()
        player = _get_player(cur, player_id)
        conn.close()

        if email:
            threading.Thread(
                target=send_welcome_email,
                args=(display_name, email),
                daemon=True,
            ).start()

        _log("INFO", f"New player enrolled: {player_id} ({display_name})")
        self._send_json(201, dict(player))

    # ------------------------------------------------------------------ POST /events
    def _handle_create_event(self):
        body = self._read_json()
        player_id  = body.get("player_id", "").strip()
        event_type = body.get("event_type", "").strip()
        value      = int(body.get("value", 1))
        metadata   = body.get("metadata")

        if not player_id or not event_type:
            self._send_json(400, {"error": "player_id and event_type required"})
            return

        conn = get_db()
        cur  = conn.cursor()
        player = _get_player(cur, player_id)
        if not player:
            conn.close()
            self._send_json(404, {"error": "Player not found"})
            return

        xp_award  = XP_MAP.get(event_type, 0) * value
        event_id  = secrets.token_hex(16)
        now       = time.time()

        cur.execute(
            """INSERT INTO events
               (event_id, player_id, event_type, value, metadata, xp_awarded,
                processed_at, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                event_id, player_id, event_type, value,
                json.dumps(metadata) if metadata else None,
                xp_award, now, now,
            ),
        )
        conn.commit()

        # Update last_active
        cur.execute(
            "UPDATE players SET last_active=?, updated_at=? WHERE player_id=?",
            (now, now, player_id),
        )
        conn.commit()

        # Award XP
        level_info = {"leveled_up": False, "new_level": player["level"]}
        if xp_award > 0:
            level_info = _award_xp(conn, cur, player_id, xp_award)

        # Check badges
        newly_awarded_badges = _check_and_award_badges(conn, cur, player_id)

        # Check challenges
        completed_challenges = _check_challenges(conn, cur, player_id, event_type)

        # Re-fetch player for updated state
        player = _get_player(cur, player_id)
        conn.close()

        # Fire emails in background
        if player and player["email"]:
            email = player["email"]
            name  = player["display_name"]
            for badge in newly_awarded_badges:
                threading.Thread(
                    target=send_badge_email,
                    args=(name, email, badge["name"], badge.get("icon", ""), badge["xp_reward"]),
                    daemon=True,
                ).start()
            if level_info["leveled_up"]:
                threading.Thread(
                    target=send_level_up_email,
                    args=(name, email, level_info["new_level"]),
                    daemon=True,
                ).start()
            for ch in completed_challenges:
                threading.Thread(
                    target=send_challenge_complete_email,
                    args=(name, email, ch["name"], ch["xp_reward"]),
                    daemon=True,
                ).start()

        _log("INFO", f"Event recorded: {event_type} for {player_id}, xp={xp_award}, badges={len(newly_awarded_badges)}")

        self._send_json(201, {
            "event_id":             event_id,
            "xp_awarded":           xp_award,
            "badges_earned":        newly_awarded_badges,
            "level_up":             level_info["leveled_up"],
            "new_level":            level_info["new_level"],
            "challenges_completed": completed_challenges,
        })

    # ------------------------------------------------------------------ POST /badges
    def _handle_create_badge(self):
        if not _verify_admin(self.headers):
            self._send_json(403, {"error": "Forbidden"})
            return

        body = self._read_json()
        name            = body.get("name", "").strip()
        description     = body.get("description", "")
        icon            = body.get("icon", "")
        category        = body.get("category", "general")
        rarity          = body.get("rarity", "common")
        xp_reward       = int(body.get("xp_reward", 0))
        condition_type  = body.get("condition_type", "")
        condition_value = int(body.get("condition_value", 0))

        if not name or not condition_type:
            self._send_json(400, {"error": "name and condition_type required"})
            return

        badge_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        now      = time.time()

        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            """INSERT INTO badges
               (badge_id, name, description, icon, category, rarity, xp_reward,
                condition_type, condition_value, active, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,1,?)""",
            (badge_id, name, description, icon, category, rarity, xp_reward,
             condition_type, condition_value, now),
        )
        conn.commit()
        badge = cur.execute("SELECT * FROM badges WHERE badge_id=?", (badge_id,)).fetchone()
        conn.close()

        _log("INFO", f"Badge created: {badge_id} ({name})")
        self._send_json(201, dict(badge))

    # ------------------------------------------------------------------ POST /challenges
    def _handle_create_challenge(self):
        if not _verify_admin(self.headers):
            self._send_json(403, {"error": "Forbidden"})
            return

        body = self._read_json()
        name            = body.get("name", "").strip()
        description     = body.get("description", "")
        xp_reward       = int(body.get("xp_reward", 0))
        badge_id        = body.get("badge_id")
        start_time      = float(body.get("start_time", time.time()))
        end_time        = body.get("end_time")
        condition_type  = body.get("condition_type", "")
        condition_value = int(body.get("condition_value", 0))

        if not name or not condition_type:
            self._send_json(400, {"error": "name and condition_type required"})
            return

        challenge_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        now          = time.time()

        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            """INSERT INTO challenges
               (challenge_id, name, description, xp_reward, badge_id, start_time,
                end_time, condition_type, condition_value, active, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,1,?)""",
            (
                challenge_id, name, description, xp_reward, badge_id,
                start_time, float(end_time) if end_time else None,
                condition_type, condition_value, now,
            ),
        )
        conn.commit()
        ch = cur.execute(
            "SELECT * FROM challenges WHERE challenge_id=?", (challenge_id,)
        ).fetchone()
        conn.close()

        _log("INFO", f"Challenge created: {challenge_id} ({name})")
        self._send_json(201, dict(ch))

    # ------------------------------------------------------------------ POST /streak
    def _handle_streak(self):
        body      = self._read_json()
        player_id = body.get("player_id", "").strip()

        if not player_id:
            self._send_json(400, {"error": "player_id required"})
            return

        conn   = get_db()
        cur    = conn.cursor()
        player = _get_player(cur, player_id)
        if not player:
            conn.close()
            self._send_json(404, {"error": "Player not found"})
            return

        now         = time.time()
        last_active = player["last_active"] or 0
        elapsed     = now - last_active
        DAY         = 86400

        streak_days = player["streak_days"]

        if elapsed < DAY:
            # Already checked in today — no change
            conn.close()
            self._send_json(200, {
                "streak_days": streak_days,
                "xp_awarded":  0,
                "message":     "Already checked in today",
            })
            return
        elif elapsed < 2 * DAY:
            # Consecutive day — extend streak
            streak_days += 1
        else:
            # Streak broken
            streak_days = 1

        xp_award = streak_days * 5
        level_info = _award_xp(conn, cur, player_id, xp_award)
        cur.execute(
            """UPDATE players SET streak_days=?, last_active=?, updated_at=?
               WHERE player_id=?""",
            (streak_days, now, now, player_id),
        )
        conn.commit()

        # Check badges after streak update
        newly_awarded_badges = _check_and_award_badges(conn, cur, player_id)
        player_upd = _get_player(cur, player_id)
        conn.close()

        if player_upd and player_upd["email"] and newly_awarded_badges:
            for badge in newly_awarded_badges:
                threading.Thread(
                    target=send_badge_email,
                    args=(player_upd["display_name"], player_upd["email"],
                          badge["name"], badge.get("icon", ""), badge["xp_reward"]),
                    daemon=True,
                ).start()

        _log("INFO", f"Streak updated: {player_id} -> {streak_days} days, xp={xp_award}")
        self._send_json(200, {
            "streak_days":   streak_days,
            "xp_awarded":    xp_award,
            "level_up":      level_info["leveled_up"],
            "new_level":     level_info["new_level"],
            "badges_earned": newly_awarded_badges,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_db()
    seed_badges()

    daemon = threading.Thread(target=_background_daemon, daemon=True, name="gamification-daemon")
    daemon.start()

    server = HTTPServer(("0.0.0.0", PORT), GamificationHandler)
    _log("INFO", f"Gamification engine listening on port {PORT}")
    print(f"[fm_gamification] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("INFO", "Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
