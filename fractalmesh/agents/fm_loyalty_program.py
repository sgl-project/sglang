"""
FractalMesh OMEGA Titan — Customer Loyalty & Rewards Program
Port: 7886
Samuel James Hiotis | ABN 56 628 117 363

Customer loyalty and rewards system. Customers earn points for purchases,
referrals, reviews, and social shares. Redeem points for discounts.
Tier system: Bronze / Silver / Gold / Platinum.
"""

import hashlib
import hmac
import json
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
BASE_DIR  = Path.home() / "fmsaas"
DB_PATH   = BASE_DIR / "database" / "sovereign.db"
LOG_PATH  = BASE_DIR / "logs" / "fm_loyalty_program.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

PORT               = int(os.environ.get("LOYALTY_PROGRAM_PORT", "7886"))
SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM      = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "")

# ---------------------------------------------------------------------------
# Tier thresholds (lifetime_points)
# ---------------------------------------------------------------------------
TIERS = [
    ("platinum", 10000),
    ("gold",     2000),
    ("silver",   500),
    ("bronze",   0),
]

TIER_BONUS = {
    "silver":   100,
    "gold":     500,
    "platinum": 2000,
}

# ---------------------------------------------------------------------------
# Points earning rules
# ---------------------------------------------------------------------------
EARN_RULES = {
    "review":       50,
    "social_share": 25,
    "birthday":     100,
    "referral":     200,
    # purchase: 1 pt per $1 AUD — calculated separately
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    line = f"[{ts}] [{level.upper()}] {msg}"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

def log_info(msg: str)  -> None: _log("INFO",  msg)
def log_error(msg: str) -> None: _log("ERROR", msg)
def log_warn(msg: str)  -> None: _log("WARN",  msg)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db()
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS members (
            id              INTEGER PRIMARY KEY,
            member_id       TEXT    UNIQUE NOT NULL,
            email           TEXT    UNIQUE NOT NULL,
            name            TEXT    NOT NULL,
            tier            TEXT    NOT NULL DEFAULT 'bronze',
            points          INTEGER NOT NULL DEFAULT 0,
            lifetime_points INTEGER NOT NULL DEFAULT 0,
            referral_code   TEXT    UNIQUE NOT NULL,
            referred_by     TEXT,
            status          TEXT    NOT NULL DEFAULT 'active',
            joined_at       REAL    NOT NULL,
            updated_at      REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id            INTEGER PRIMARY KEY,
            txn_id        TEXT    UNIQUE NOT NULL,
            member_id     TEXT    NOT NULL,
            type          TEXT    NOT NULL,
            points        INTEGER NOT NULL,
            description   TEXT,
            order_ref     TEXT,
            balance_after INTEGER NOT NULL,
            created_at    REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rewards (
            id             INTEGER PRIMARY KEY,
            reward_id      TEXT    UNIQUE NOT NULL,
            name           TEXT    NOT NULL,
            description    TEXT,
            points_cost    INTEGER NOT NULL,
            reward_type    TEXT    NOT NULL,
            reward_value   REAL    NOT NULL DEFAULT 0.0,
            active         INTEGER NOT NULL DEFAULT 1,
            quantity       INTEGER NOT NULL DEFAULT -1,
            redeemed_count INTEGER NOT NULL DEFAULT 0,
            created_at     REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS redemptions (
            id             INTEGER PRIMARY KEY,
            redemption_id  TEXT    UNIQUE NOT NULL,
            member_id      TEXT    NOT NULL,
            reward_id      TEXT    NOT NULL,
            points_spent   INTEGER NOT NULL,
            coupon_code    TEXT    UNIQUE NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'active',
            used_at        REAL,
            created_at     REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_members_email        ON members(email);
        CREATE INDEX IF NOT EXISTS idx_members_referral     ON members(referral_code);
        CREATE INDEX IF NOT EXISTS idx_transactions_member  ON transactions(member_id);
        CREATE INDEX IF NOT EXISTS idx_redemptions_member   ON redemptions(member_id);
        CREATE INDEX IF NOT EXISTS idx_redemptions_coupon   ON redemptions(coupon_code);
    """)

    conn.commit()
    conn.close()
    log_info("Database initialised")

def seed_rewards() -> None:
    """Insert default rewards if rewards table is empty."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rewards")
    count = cur.fetchone()[0]
    if count > 0:
        conn.close()
        return

    now = time.time()
    defaults = [
        ("5% Discount",  "Get 5% off your next order",     100,  "discount", 5.0),
        ("10% Discount", "Get 10% off your next order",    200,  "discount", 10.0),
        ("$10 Credit",   "Receive $10 account credit",     400,  "credit",   10.0),
        ("Free Shipping","Free shipping on your next order",150, "shipping",  0.0),
        ("VIP Access",   "Exclusive VIP member access",    1000, "access",    0.0),
    ]
    for name, desc, cost, rtype, rvalue in defaults:
        rid = "RWD-" + secrets.token_urlsafe(8).upper()
        cur.execute("""
            INSERT OR IGNORE INTO rewards
                (reward_id, name, description, points_cost, reward_type,
                 reward_value, active, quantity, redeemed_count, created_at)
            VALUES (?,?,?,?,?,?,1,-1,0,?)
        """, (rid, name, desc, cost, rtype, rvalue, now))

    conn.commit()
    conn.close()
    log_info("Default rewards seeded")

# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------
def calc_tier(lifetime_points: int) -> str:
    for tier_name, threshold in TIERS:
        if lifetime_points >= threshold:
            return tier_name
    return "bronze"

# ---------------------------------------------------------------------------
# SendGrid email
# ---------------------------------------------------------------------------
def send_email(to_email: str, subject: str, body_html: str) -> bool:
    if not SENDGRID_API_KEY:
        log_warn(f"SendGrid not configured — skipping email to {to_email}")
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from":             {"email": SENDGRID_FROM},
        "subject":          subject,
        "content":          [{"type": "text/html", "value": body_html}],
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
            ok = resp.status in (200, 201, 202)
            if ok:
                log_info(f"Email sent to {to_email}: {subject}")
            return ok
    except urllib.error.URLError as exc:
        log_error(f"SendGrid error for {to_email}: {exc}")
        return False

def email_welcome(member: dict) -> None:
    subject = "Welcome to FractalMesh Rewards!"
    body    = f"""
    <h2>Welcome, {member['name']}!</h2>
    <p>You have successfully enrolled in the FractalMesh Loyalty Program.</p>
    <p>Your personal referral code is: <strong>{member['referral_code']}</strong></p>
    <p>Share it with friends — you'll earn <strong>200 bonus points</strong> each time a referred
    friend makes their first purchase.</p>
    <p>Current tier: <strong>{member['tier'].upper()}</strong></p>
    <p>Start earning points today for purchases, reviews, and social shares!</p>
    <br><p>— The FractalMesh Team</p>
    """
    send_email(member["email"], subject, body)

def email_tier_upgrade(member: dict, old_tier: str, new_tier: str, bonus_pts: int) -> None:
    subject = f"Congratulations! You've reached {new_tier.upper()} tier!"
    body    = f"""
    <h2>Tier Upgrade — Congratulations, {member['name']}!</h2>
    <p>You have been upgraded from <strong>{old_tier.upper()}</strong>
    to <strong>{new_tier.upper()}</strong>!</p>
    <p>As a welcome bonus you've received <strong>{bonus_pts} extra points</strong>.</p>
    <p>Your current balance: <strong>{member['points']} points</strong></p>
    <p>Keep earning to unlock even greater rewards!</p>
    <br><p>— The FractalMesh Team</p>
    """
    send_email(member["email"], subject, body)

def email_redemption(member: dict, reward: dict, coupon_code: str) -> None:
    subject = f"Your Reward: {reward['name']}"
    body    = f"""
    <h2>Redemption Confirmed!</h2>
    <p>Hi {member['name']},</p>
    <p>You have successfully redeemed <strong>{reward['name']}</strong>
    ({reward['points_cost']} points).</p>
    <p>Your coupon code is: <strong>{coupon_code}</strong></p>
    <p>Remaining points balance: <strong>{member['points']}</strong></p>
    <p>Use your coupon at checkout. This code is valid until used.</p>
    <br><p>— The FractalMesh Team</p>
    """
    send_email(member["email"], subject, body)

def email_birthday(member: dict, points_awarded: int) -> None:
    subject = "Happy Birthday from FractalMesh Rewards!"
    body    = f"""
    <h2>Happy Birthday, {member['name']}! 🎉</h2>
    <p>As a thank-you for being a valued loyalty member, we've credited your account
    with <strong>{points_awarded} birthday bonus points</strong>!</p>
    <p>Current balance: <strong>{member['points']} points</strong></p>
    <p>Enjoy your special day!</p>
    <br><p>— The FractalMesh Team</p>
    """
    send_email(member["email"], subject, body)

# ---------------------------------------------------------------------------
# Core business logic
# ---------------------------------------------------------------------------
def _new_member_id() -> str:
    return "MBR-" + secrets.token_urlsafe(8).upper()

def _new_txn_id() -> str:
    return "TXN-" + secrets.token_urlsafe(8).upper()

def _new_redemption_id() -> str:
    return "RDM-" + secrets.token_urlsafe(8).upper()

def enroll_member(email: str, name: str, referred_by_code: str = None) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    try:
        # Check for duplicate email
        cur.execute("SELECT member_id FROM members WHERE email=?", (email,))
        existing = cur.fetchone()
        if existing:
            return {"error": "Email already enrolled", "member_id": existing["member_id"]}

        # Validate referral code if provided
        referrer_id = None
        if referred_by_code:
            cur.execute("SELECT member_id FROM members WHERE referral_code=?", (referred_by_code,))
            ref_row = cur.fetchone()
            if ref_row:
                referrer_id = ref_row["member_id"]

        member_id     = _new_member_id()
        referral_code = secrets.token_urlsafe(6).upper()
        now           = time.time()

        cur.execute("""
            INSERT INTO members
                (member_id, email, name, tier, points, lifetime_points,
                 referral_code, referred_by, status, joined_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (member_id, email, name, "bronze", 0, 0,
              referral_code, referrer_id, "active", now, now))

        conn.commit()

        member = {
            "member_id":      member_id,
            "email":          email,
            "name":           name,
            "tier":           "bronze",
            "points":         0,
            "lifetime_points":0,
            "referral_code":  referral_code,
            "referred_by":    referrer_id,
            "status":         "active",
            "joined_at":      now,
        }

        log_info(f"Member enrolled: {member_id} ({email})")
        threading.Thread(target=email_welcome, args=(member,), daemon=True).start()
        return {"success": True, "member": member}

    except sqlite3.IntegrityError as exc:
        conn.rollback()
        log_error(f"Enroll error for {email}: {exc}")
        return {"error": str(exc)}
    finally:
        conn.close()

def get_member(member_id: str) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM members WHERE member_id=?", (member_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def earn_points(member_id: str, earn_type: str,
                order_ref: str = None, order_value: float = None) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT * FROM members WHERE member_id=? AND status='active'",
                    (member_id,))
        member = cur.fetchone()
        if not member:
            return {"error": "Member not found or inactive"}

        # Calculate points
        if earn_type == "purchase":
            if order_value is None or order_value <= 0:
                return {"error": "order_value required for purchase type"}
            points = int(order_value)  # 1 pt per $1 AUD
        elif earn_type in EARN_RULES:
            points = EARN_RULES[earn_type]
        else:
            return {"error": f"Unknown earn type: {earn_type}"}

        if points <= 0:
            return {"error": "No points to award"}

        new_points          = member["points"] + points
        new_lifetime_points = member["lifetime_points"] + points
        now                 = time.time()

        # Insert transaction
        txn_id       = _new_txn_id()
        description  = f"{earn_type.replace('_', ' ').title()} points"
        if earn_type == "purchase" and order_value:
            description = f"Purchase ${order_value:.2f} AUD"
        elif earn_type == "referral":
            description = "Referral bonus"

        cur.execute("""
            INSERT INTO transactions
                (txn_id, member_id, type, points, description,
                 order_ref, balance_after, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (txn_id, member_id, earn_type, points, description,
              order_ref, new_points, now))

        # Check tier upgrade
        old_tier = member["tier"]
        new_tier = calc_tier(new_lifetime_points)
        tier_changed = (new_tier != old_tier)
        bonus_pts = 0

        if tier_changed:
            bonus_pts       = TIER_BONUS.get(new_tier, 0)
            new_points     += bonus_pts
            new_lifetime_points += bonus_pts

            # Insert tier_upgrade transaction
            tu_txn_id = _new_txn_id()
            cur.execute("""
                INSERT INTO transactions
                    (txn_id, member_id, type, points, description,
                     order_ref, balance_after, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (tu_txn_id, member_id, "tier_upgrade", bonus_pts,
                  f"Tier upgrade bonus: {new_tier.title()}",
                  None, new_points, now))

        # Update member
        cur.execute("""
            UPDATE members
            SET points=?, lifetime_points=?, tier=?, updated_at=?
            WHERE member_id=?
        """, (new_points, new_lifetime_points, new_tier, now, member_id))

        conn.commit()

        result = {
            "success":      True,
            "txn_id":       txn_id,
            "points_earned":points,
            "balance":      new_points,
            "tier":         new_tier,
            "tier_upgraded":tier_changed,
        }

        if tier_changed:
            result["bonus_points"] = bonus_pts
            result["old_tier"]     = old_tier
            updated_member = {
                "name":   member["name"],
                "email":  member["email"],
                "points": new_points,
                "tier":   new_tier,
            }
            threading.Thread(
                target=email_tier_upgrade,
                args=(updated_member, old_tier, new_tier, bonus_pts),
                daemon=True,
            ).start()

        # Award referral points to referrer on first purchase
        if earn_type == "purchase" and member["referred_by"]:
            cur2 = conn.cursor() if False else get_db().cursor()
            _maybe_award_referral(member["referred_by"], member_id)

        log_info(f"Points earned: {member_id} +{points} ({earn_type})")
        return result

    except Exception as exc:
        conn.rollback()
        log_error(f"Earn points error: {exc}")
        return {"error": str(exc)}
    finally:
        conn.close()

def _maybe_award_referral(referrer_id: str, new_member_id: str) -> None:
    """Award referral bonus to referrer if this is the new member's first purchase."""
    conn = get_db()
    cur  = conn.cursor()
    try:
        # Check if this member has had a previous purchase transaction
        cur.execute("""
            SELECT COUNT(*) FROM transactions
            WHERE member_id=? AND type='purchase'
        """, (new_member_id,))
        purchase_count = cur.fetchone()[0]

        # Only award if this is exactly the first purchase (count = 1 after insertion)
        if purchase_count != 1:
            conn.close()
            return

        # Check referrer exists
        cur.execute("SELECT * FROM members WHERE member_id=? AND status='active'",
                    (referrer_id,))
        referrer = cur.fetchone()
        if not referrer:
            conn.close()
            return

        ref_pts              = EARN_RULES["referral"]
        new_points          = referrer["points"] + ref_pts
        new_lifetime_points = referrer["lifetime_points"] + ref_pts
        now                  = time.time()
        txn_id               = _new_txn_id()

        cur.execute("""
            INSERT INTO transactions
                (txn_id, member_id, type, points, description,
                 order_ref, balance_after, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (txn_id, referrer_id, "referral", ref_pts,
              f"Referral bonus for member {new_member_id}",
              None, new_points, now))

        old_tier = referrer["tier"]
        new_tier = calc_tier(new_lifetime_points)
        tier_changed = (new_tier != old_tier)
        bonus_pts = 0

        if tier_changed:
            bonus_pts            = TIER_BONUS.get(new_tier, 0)
            new_points          += bonus_pts
            new_lifetime_points += bonus_pts
            tu_txn_id = _new_txn_id()
            cur.execute("""
                INSERT INTO transactions
                    (txn_id, member_id, type, points, description,
                     order_ref, balance_after, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (tu_txn_id, referrer_id, "tier_upgrade", bonus_pts,
                  f"Tier upgrade bonus: {new_tier.title()}",
                  None, new_points, now))

        cur.execute("""
            UPDATE members
            SET points=?, lifetime_points=?, tier=?, updated_at=?
            WHERE member_id=?
        """, (new_points, new_lifetime_points, new_tier, now, referrer_id))

        conn.commit()
        log_info(f"Referral bonus awarded to {referrer_id}: +{ref_pts}")

        if tier_changed:
            updated_ref = dict(referrer)
            updated_ref["points"] = new_points
            updated_ref["tier"]   = new_tier
            threading.Thread(
                target=email_tier_upgrade,
                args=(updated_ref, old_tier, new_tier, bonus_pts),
                daemon=True,
            ).start()

    except Exception as exc:
        conn.rollback()
        log_error(f"Referral bonus error: {exc}")
    finally:
        conn.close()

def redeem_points(member_id: str, reward_id: str) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT * FROM members WHERE member_id=? AND status='active'",
                    (member_id,))
        member = cur.fetchone()
        if not member:
            return {"error": "Member not found or inactive"}

        cur.execute("SELECT * FROM rewards WHERE reward_id=? AND active=1", (reward_id,))
        reward = cur.fetchone()
        if not reward:
            return {"error": "Reward not found or inactive"}

        # Check quantity
        if reward["quantity"] >= 0 and reward["redeemed_count"] >= reward["quantity"]:
            return {"error": "Reward is out of stock"}

        # Check sufficient points
        if member["points"] < reward["points_cost"]:
            return {"error": "Insufficient points",
                    "required": reward["points_cost"],
                    "balance":  member["points"]}

        new_balance  = member["points"] - reward["points_cost"]
        now          = time.time()
        coupon_code  = secrets.token_urlsafe(10).upper()
        redemption_id = _new_redemption_id()
        txn_id        = _new_txn_id()

        # Insert redemption
        cur.execute("""
            INSERT INTO redemptions
                (redemption_id, member_id, reward_id, points_spent,
                 coupon_code, status, used_at, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (redemption_id, member_id, reward_id, reward["points_cost"],
              coupon_code, "active", None, now))

        # Insert debit transaction
        cur.execute("""
            INSERT INTO transactions
                (txn_id, member_id, type, points, description,
                 order_ref, balance_after, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (txn_id, member_id, "redemption", -reward["points_cost"],
              f"Redeemed: {reward['name']}",
              redemption_id, new_balance, now))

        # Deduct points from member
        cur.execute("""
            UPDATE members SET points=?, updated_at=? WHERE member_id=?
        """, (new_balance, now, member_id))

        # Increment redeemed_count
        cur.execute("""
            UPDATE rewards SET redeemed_count=redeemed_count+1 WHERE reward_id=?
        """, (reward_id,))

        conn.commit()

        result = {
            "success":        True,
            "redemption_id":  redemption_id,
            "coupon_code":    coupon_code,
            "reward":         dict(reward),
            "points_spent":   reward["points_cost"],
            "new_balance":    new_balance,
        }

        updated_member = dict(member)
        updated_member["points"] = new_balance
        threading.Thread(
            target=email_redemption,
            args=(updated_member, dict(reward), coupon_code),
            daemon=True,
        ).start()

        log_info(f"Redemption: {member_id} redeemed {reward_id} → coupon {coupon_code}")
        return result

    except Exception as exc:
        conn.rollback()
        log_error(f"Redeem points error: {exc}")
        return {"error": str(exc)}
    finally:
        conn.close()

def get_transactions(member_id: str) -> list:
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT * FROM transactions
        WHERE member_id=?
        ORDER BY created_at DESC
    """, (member_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def list_rewards() -> list:
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM rewards WHERE active=1 ORDER BY points_cost ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def create_reward(name: str, description: str, points_cost: int,
                  reward_type: str, reward_value: float, quantity: int = -1) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    try:
        reward_id = "RWD-" + secrets.token_urlsafe(8).upper()
        now       = time.time()
        cur.execute("""
            INSERT INTO rewards
                (reward_id, name, description, points_cost, reward_type,
                 reward_value, active, quantity, redeemed_count, created_at)
            VALUES (?,?,?,?,?,?,1,?,0,?)
        """, (reward_id, name, description, points_cost, reward_type,
              reward_value, quantity, now))
        conn.commit()
        log_info(f"Reward created: {reward_id} ({name})")
        return {"success": True, "reward_id": reward_id}
    except Exception as exc:
        conn.rollback()
        log_error(f"Create reward error: {exc}")
        return {"error": str(exc)}
    finally:
        conn.close()

def get_redemptions(member_id: str) -> list:
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT r.*, rw.name AS reward_name, rw.description AS reward_description,
               rw.reward_type, rw.reward_value
        FROM redemptions r
        JOIN rewards rw ON r.reward_id = rw.reward_id
        WHERE r.member_id=?
        ORDER BY r.created_at DESC
    """, (member_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def validate_coupon(coupon_code: str) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT r.*, rw.name AS reward_name, rw.description AS reward_description,
               rw.reward_type, rw.reward_value
        FROM redemptions r
        JOIN rewards rw ON r.reward_id = rw.reward_id
        WHERE r.coupon_code=?
    """, (coupon_code,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"valid": False, "error": "Coupon not found"}
    data = dict(row)
    if data["status"] != "active":
        return {"valid": False, "error": f"Coupon already {data['status']}", "coupon": data}
    return {"valid": True, "coupon": data}

def use_coupon(coupon_code: str) -> dict:
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT * FROM redemptions WHERE coupon_code=?", (coupon_code,))
        row = cur.fetchone()
        if not row:
            return {"error": "Coupon not found"}
        if row["status"] != "active":
            return {"error": f"Coupon already {row['status']}"}

        now = time.time()
        cur.execute("""
            UPDATE redemptions SET status='used', used_at=? WHERE coupon_code=?
        """, (now, coupon_code))
        conn.commit()
        log_info(f"Coupon used: {coupon_code}")
        return {"success": True, "coupon_code": coupon_code, "used_at": now}
    except Exception as exc:
        conn.rollback()
        log_error(f"Use coupon error: {exc}")
        return {"error": str(exc)}
    finally:
        conn.close()

def get_leaderboard() -> list:
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT name, tier, lifetime_points
        FROM members
        WHERE status='active'
        ORDER BY lifetime_points DESC
        LIMIT 20
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_analytics() -> dict:
    conn = get_db()
    cur  = conn.cursor()

    # Members per tier
    cur.execute("""
        SELECT tier, COUNT(*) AS count
        FROM members WHERE status='active'
        GROUP BY tier
    """)
    tiers_data = {row["tier"]: row["count"] for row in cur.fetchall()}

    # Total points issued (positive transactions)
    cur.execute("SELECT COALESCE(SUM(points),0) FROM transactions WHERE points > 0")
    total_issued = cur.fetchone()[0]

    # Total points redeemed
    cur.execute("SELECT COALESCE(SUM(points_spent),0) FROM redemptions")
    total_redeemed = cur.fetchone()[0]

    # Active redemptions
    cur.execute("SELECT COUNT(*) FROM redemptions WHERE status='active'")
    active_redemptions = cur.fetchone()[0]

    # Total members
    cur.execute("SELECT COUNT(*) FROM members WHERE status='active'")
    total_members = cur.fetchone()[0]

    conn.close()
    return {
        "total_members":      total_members,
        "members_per_tier":   tiers_data,
        "total_points_issued":total_issued,
        "total_points_redeemed": total_redeemed,
        "active_redemptions": active_redemptions,
    }

# ---------------------------------------------------------------------------
# Birthday background thread
# ---------------------------------------------------------------------------
def _birthday_worker() -> None:
    """Daemon thread — runs every hour, checks for birthday anniversaries."""
    log_info("Birthday worker started")
    while True:
        try:
            _check_birthdays()
        except Exception as exc:
            log_error(f"Birthday worker error: {exc}")
        time.sleep(3600)

def _check_birthdays() -> None:
    conn = get_db()
    cur  = conn.cursor()
    now  = time.time()

    # Find active members whose join anniversary is within the next 24-hour window
    # i.e. today's day-of-year matches their join day-of-year
    cur.execute("SELECT * FROM members WHERE status='active'")
    members = cur.fetchall()

    today_tm = time.gmtime(now)
    today_yd = today_tm.tm_yday
    today_yr = today_tm.tm_year

    for member in members:
        try:
            joined_tm = time.gmtime(member["joined_at"])
            # Same month and day (use month+day for leap-year safety)
            if joined_tm.tm_mon != today_tm.tm_mon:
                continue
            if joined_tm.tm_mday != today_tm.tm_mday:
                continue

            # Already awarded this calendar year?
            cur.execute("""
                SELECT COUNT(*) FROM transactions
                WHERE member_id=? AND type='birthday'
                  AND created_at >= ? AND created_at < ?
            """, (
                member["member_id"],
                time.mktime(time.struct_time((today_yr, 1, 1, 0, 0, 0, 0, 0, 0))),
                time.mktime(time.struct_time((today_yr + 1, 1, 1, 0, 0, 0, 0, 0, 0))),
            ))
            already_awarded = cur.fetchone()[0] > 0
            if already_awarded:
                continue

            # Award birthday points
            pts              = EARN_RULES["birthday"]
            new_points       = member["points"] + pts
            new_lifetime     = member["lifetime_points"] + pts
            txn_id           = _new_txn_id()

            cur.execute("""
                INSERT INTO transactions
                    (txn_id, member_id, type, points, description,
                     order_ref, balance_after, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (txn_id, member["member_id"], "birthday", pts,
                  "Birthday bonus points", None, new_points, now))

            old_tier  = member["tier"]
            new_tier  = calc_tier(new_lifetime)
            bonus_pts = 0
            if new_tier != old_tier:
                bonus_pts     = TIER_BONUS.get(new_tier, 0)
                new_points   += bonus_pts
                new_lifetime += bonus_pts
                tu_txn_id = _new_txn_id()
                cur.execute("""
                    INSERT INTO transactions
                        (txn_id, member_id, type, points, description,
                         order_ref, balance_after, created_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (tu_txn_id, member["member_id"], "tier_upgrade", bonus_pts,
                      f"Tier upgrade bonus: {new_tier.title()}",
                      None, new_points, now))

            cur.execute("""
                UPDATE members SET points=?, lifetime_points=?, tier=?, updated_at=?
                WHERE member_id=?
            """, (new_points, new_lifetime, new_tier, now, member["member_id"]))

            conn.commit()
            log_info(f"Birthday bonus awarded: {member['member_id']} +{pts}")

            updated_member = dict(member)
            updated_member["points"] = new_points
            threading.Thread(
                target=email_birthday,
                args=(updated_member, pts),
                daemon=True,
            ).start()

            if new_tier != old_tier:
                email_tier_upgrade(updated_member, old_tier, new_tier, bonus_pts)

        except Exception as exc:
            log_error(f"Birthday check error for {member['member_id']}: {exc}")

    conn.close()

# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
def _parse_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}

def _send_json(handler, status: int, data: dict) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

def _verify_admin(handler) -> bool:
    if not ADMIN_SECRET:
        return True  # Not configured — open (dev mode)
    provided = handler.headers.get("X-Admin-Secret", "")
    try:
        return hmac.compare_digest(
            provided.encode("utf-8"),
            ADMIN_SECRET.encode("utf-8"),
        )
    except Exception:
        return False

class LoyaltyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        log_info(f"HTTP {fmt % args}")

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def _route_get(self, path: str, parts: list) -> None:
        if path == "/health":
            _send_json(self, 200, {"status": "ok", "service": "fm_loyalty_program",
                                    "port": PORT, "ts": time.time()})

        elif path == "/rewards":
            _send_json(self, 200, {"rewards": list_rewards()})

        elif path == "/leaderboard":
            _send_json(self, 200, {"leaderboard": get_leaderboard()})

        elif path == "/analytics":
            if not _verify_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            _send_json(self, 200, get_analytics())

        elif len(parts) == 3 and parts[1] == "members":
            member_id = parts[2]
            m = get_member(member_id)
            if not m:
                _send_json(self, 404, {"error": "Member not found"})
            else:
                _send_json(self, 200, {"member": m})

        elif len(parts) == 3 and parts[1] == "transactions":
            member_id = parts[2]
            m = get_member(member_id)
            if not m:
                _send_json(self, 404, {"error": "Member not found"})
            else:
                txns = get_transactions(member_id)
                _send_json(self, 200, {"member_id": member_id, "transactions": txns})

        elif len(parts) == 3 and parts[1] == "redemptions":
            member_id = parts[2]
            m = get_member(member_id)
            if not m:
                _send_json(self, 404, {"error": "Member not found"})
            else:
                redemptions = get_redemptions(member_id)
                _send_json(self, 200, {"member_id": member_id, "redemptions": redemptions})

        else:
            _send_json(self, 404, {"error": "Not found"})

    def _route_post(self, path: str, parts: list) -> None:
        body = _parse_body(self)

        if path == "/members/enroll":
            email = body.get("email", "").strip().lower()
            name  = body.get("name", "").strip()
            if not email or not name:
                _send_json(self, 400, {"error": "email and name required"})
                return
            ref_code = body.get("referred_by_code", "").strip().upper() or None
            result   = enroll_member(email, name, ref_code)
            status   = 200 if result.get("success") else 409 if "already enrolled" in result.get("error","") else 400
            _send_json(self, status, result)

        elif path == "/points/earn":
            member_id   = body.get("member_id", "").strip()
            earn_type   = body.get("type", "").strip()
            order_ref   = body.get("order_ref")
            order_value = body.get("order_value")
            if not member_id or not earn_type:
                _send_json(self, 400, {"error": "member_id and type required"})
                return
            if order_value is not None:
                try:
                    order_value = float(order_value)
                except (TypeError, ValueError):
                    _send_json(self, 400, {"error": "order_value must be numeric"})
                    return
            result = earn_points(member_id, earn_type, order_ref, order_value)
            status = 200 if result.get("success") else 400
            _send_json(self, status, result)

        elif path == "/points/redeem":
            member_id = body.get("member_id", "").strip()
            reward_id = body.get("reward_id", "").strip()
            if not member_id or not reward_id:
                _send_json(self, 400, {"error": "member_id and reward_id required"})
                return
            result = redeem_points(member_id, reward_id)
            status = 200 if result.get("success") else 400
            _send_json(self, status, result)

        elif path == "/rewards":
            if not _verify_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            name        = body.get("name", "").strip()
            description = body.get("description", "").strip()
            points_cost = body.get("points_cost")
            reward_type = body.get("reward_type", "").strip()
            reward_value= body.get("reward_value", 0.0)
            quantity    = body.get("quantity", -1)
            if not name or points_cost is None or not reward_type:
                _send_json(self, 400, {"error": "name, points_cost, and reward_type required"})
                return
            try:
                points_cost  = int(points_cost)
                reward_value = float(reward_value)
                quantity     = int(quantity)
            except (TypeError, ValueError):
                _send_json(self, 400, {"error": "Invalid numeric fields"})
                return
            result = create_reward(name, description, points_cost, reward_type,
                                   reward_value, quantity)
            status = 201 if result.get("success") else 400
            _send_json(self, status, result)

        elif path == "/coupon/validate":
            code = body.get("coupon_code", "").strip().upper()
            if not code:
                _send_json(self, 400, {"error": "coupon_code required"})
                return
            result = validate_coupon(code)
            _send_json(self, 200, result)

        elif path == "/coupon/use":
            code = body.get("coupon_code", "").strip().upper()
            if not code:
                _send_json(self, 400, {"error": "coupon_code required"})
                return
            result = use_coupon(code)
            status = 200 if result.get("success") else 400
            _send_json(self, status, result)

        else:
            _send_json(self, 404, {"error": "Not found"})

    # ------------------------------------------------------------------
    # do_GET / do_POST
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        parts  = [p for p in path.split("/") if p]
        parts.insert(0, "")  # maintain leading empty for index consistency
        try:
            self._route_get(path, parts)
        except Exception as exc:
            log_error(f"GET {path} error: {exc}")
            _send_json(self, 500, {"error": "Internal server error"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        parts  = [p for p in path.split("/") if p]
        parts.insert(0, "")
        try:
            self._route_post(path, parts)
        except Exception as exc:
            log_error(f"POST {path} error: {exc}")
            _send_json(self, 500, {"error": "Internal server error"})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log_info("FractalMesh Loyalty Program starting up")
    init_db()
    seed_rewards()

    # Start birthday background thread
    birthday_thread = threading.Thread(target=_birthday_worker, daemon=True)
    birthday_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), LoyaltyHandler)
    log_info(f"Loyalty Program listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("Loyalty Program shutting down")
        server.server_close()

if __name__ == "__main__":
    main()
