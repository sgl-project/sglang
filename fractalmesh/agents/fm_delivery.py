#!/usr/bin/env python3
"""
fm_delivery.py — Automated product delivery engine
Watches delivery_queue table → sends product via Gmail → marks delivered
Samuel James Hiotis | ABN 56628117363 | Albury NSW
"""
import os, time, json, smtplib, sqlite3, logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

# Load vault
for vault in [Path(os.path.expanduser("~/.secrets/fractal.env")),
              Path(os.path.expanduser("~/fmsaas/.env"))]:
    if vault.exists():
        for line in vault.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DELIVERY] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.environ.get("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")),
                         "logs", "delivery.log"), mode="a"
        )
    ]
)
log = logging.getLogger("delivery")

ROOT   = os.environ.get("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB     = os.path.join(ROOT, "db", "sovereign.db")
GMAIL  = os.environ.get("GMAIL_USER", "")
GPASS  = os.environ.get("GMAIL_APP_PASS", "")

# Stripe price ID → product info mapping
PRODUCTS = {
    # Fractal Signal Feed
    "price_SIGNAL_499": {
        "name":     "FractalMesh Fractal Signal Feed",
        "price":    "$499 AUD",
        "features": ["5 live pairs (BTC/ETH/SOL/XRP/BNB)", "Sub-second RL updates",
                     "Fractal confidence scores", "Telegram alerts"],
    },
    # Sovereign AI Dashboard
    "price_DASH_299": {
        "name":     "FractalMesh Sovereign AI Dashboard",
        "price":    "$299 AUD",
        "features": ["Real-time leads pipeline", "Order management",
                     "Neural AI chat", "PM2 monitoring"],
    },
    # NFT Genesis Pack
    "price_NFT_199": {
        "name":     "FractalMesh NFT Genesis Pack",
        "price":    "$199 AUD",
        "features": ["On-chain royalties", "Fractal RL-generated art",
                     "Solana mainnet", "Transferable token"],
    },
    # Enterprise Bundle
    "price_ENT_899": {
        "name":     "FractalMesh Enterprise Bundle",
        "price":    "$899 AUD",
        "features": ["All products included", "White-label option",
                     "Priority support", "ABN invoiced"],
    },
}

def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS delivery_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session_id TEXT,
            customer_email TEXT,
            product_name TEXT,
            price_id TEXT,
            amount_aud REAL,
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            created_at TEXT,
            delivered_at TEXT
        )
    """)
    conn.commit()

def send_delivery(email: str, product: dict, session_id: str, amount: float) -> bool:
    if not GPASS:
        log.warning("No GMAIL_APP_PASS — cannot send delivery email")
        return False

    features = "\n".join(f"  • {f}" for f in product["features"])
    body = f"""Welcome to FractalMesh!

Your order is confirmed and active.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORDER CONFIRMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Product:    {product["name"]}
Amount:     {product["price"]}
Session:    {session_id}
Confirmed:  {datetime.now().strftime("%Y-%m-%d %H:%M AEST")}

INCLUDED IN YOUR PLAN:
{features}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GETTING STARTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Access credentials and onboarding instructions will follow
within 24 hours. For questions, reply to this email.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Samuel James Hiotis
FractalMesh — Autonomous Intelligence Systems
ABN 56628117363 | Albury NSW 2640
"""
    msg = MIMEMultipart()
    msg["From"]    = f"FractalMesh <{GMAIL}>"
    msg["To"]      = email
    msg["Subject"] = f"FractalMesh — {product['name']} Active"
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as srv:
            srv.login(GMAIL, GPASS)
            srv.send_message(msg)
        log.info("Delivered → %s | %s", email, product["name"])
        return True
    except Exception as e:
        log.error("SMTP error → %s: %s", email, e)
        return False

def process_queue():
    if not Path(DB).exists():
        return
    try:
        conn = sqlite3.connect(DB, timeout=10)
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        pending = conn.execute(
            "SELECT * FROM delivery_queue WHERE status='pending' AND attempts < 3"
        ).fetchall()
        for row in pending:
            conn.execute("UPDATE delivery_queue SET attempts=attempts+1 WHERE id=?", (row["id"],))
            conn.commit()
            product = PRODUCTS.get(row["price_id"] or "", {
                "name":     row["product_name"] or "FractalMesh Product",
                "price":    f"${row['amount_aud']:.2f} AUD",
                "features": ["Access confirmed — details to follow"],
            })
            ok = send_delivery(row["customer_email"], product,
                               row["stripe_session_id"] or "MANUAL", row["amount_aud"] or 0)
            conn.execute(
                "UPDATE delivery_queue SET status=?, delivered_at=? WHERE id=?",
                ("delivered" if ok else "failed",
                 datetime.now().isoformat() if ok else None, row["id"])
            )
            conn.commit()
        conn.close()
    except Exception as e:
        log.error("Queue error: %s", e)

def main():
    log.info("fm-delivery started | DB=%s", DB)
    while True:
        try:
            process_queue()
        except Exception as e:
            log.error("Cycle error: %s", e)
        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
