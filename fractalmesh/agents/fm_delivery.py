#!/usr/bin/env python3
"""
fm_delivery.py v402 — Automated product delivery engine
Watches delivery_queue → sends HTML branded email → marks delivered
Samuel James Hiotis | ABN 56628117363 | Albury NSW
v402: HTML emails with branding, per-product onboarding links, access keys.
"""
import os, time, json, smtplib, sqlite3, logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

for vault in [Path(os.path.expanduser("~/.secrets/fractal.env")),
              Path(os.path.expanduser("~/fmsaas/.env"))]:
    if vault.exists():
        for line in vault.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DELIVERY] %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("delivery")

ROOT  = os.environ.get("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB    = os.path.join(ROOT, "db", "sovereign.db")
GMAIL = os.environ.get("GMAIL_USER","")
GPASS = os.environ.get("GMAIL_APP_PASS","")

PRODUCTS = {
    "price_SIGNAL_499": {
        "name":"FractalMesh Fractal Signal Feed","price":"$499 AUD",
        "features":["5 live pairs (BTC/ETH/SOL/XRP/BNB)","Sub-second RL updates",
                    "Fractal confidence scores","Telegram alerts"],
        "dashboard_section":"signals",
        "onboarding":[
            "1. Visit your dashboard at http://localhost:8090",
            "2. Navigate to → Signals section",
            "3. Your live feed updates every second automatically",
            "4. Set up Telegram: add TELEGRAM_BOT_TOKEN to your vault",
        ],
    },
    "price_DASH_299": {
        "name":"FractalMesh Sovereign AI Dashboard","price":"$299 AUD",
        "features":["Real-time leads pipeline","Order management",
                    "Neural AI chat","PM2 monitoring"],
        "dashboard_section":"system",
        "onboarding":[
            "1. Dashboard live at http://localhost:8090",
            "2. Access AI Chat via → Chat tab",
            "3. Manage leads via → Leads section",
            "4. System health via → System tab",
        ],
    },
    "price_NFT_199": {
        "name":"FractalMesh NFT Genesis Pack","price":"$199 AUD",
        "features":["On-chain royalties","Fractal RL-generated art",
                    "Solana mainnet","Transferable token"],
        "dashboard_section":"synthwave",
        "onboarding":[
            "1. Set SOLANA_KEYPAIR_PATH in your vault",
            "2. Set PINATA_JWT for IPFS hosting",
            "3. NFTs mint automatically every 10 minutes",
            "4. View gallery at /api/nft/gallery",
        ],
    },
    "price_ENT_899": {
        "name":"FractalMesh Enterprise Bundle","price":"$899 AUD",
        "features":["All products included","White-label option",
                    "Priority support","ABN invoiced"],
        "dashboard_section":"analytics",
        "onboarding":[
            "1. Full system deployed at http://localhost:8090",
            "2. All 14 agents operational via PM2",
            "3. Contact samuel@fractalmesh.io for white-label setup",
            "4. ABN invoice: 56628117363 (Samuel James Hiotis)",
        ],
    },
    "price_SW_149": {
        "name":"FractalMesh Synthwave Empire","price":"$149 AUD",
        "features":["AI DJ auto-compose","Pinata IPFS hosting",
                    "7% on-chain royalties","Dev.to auto-publish"],
        "dashboard_section":"synthwave",
        "onboarding":[
            "1. Set OPENROUTER_API_KEY for AI music generation",
            "2. Set PINATA_JWT for IPFS",
            "3. AI DJ composes every 10 minutes automatically",
            "4. NFT minted on Solana — track at /api/nft/gallery",
        ],
    },
    "price_GEO_349": {
        "name":"FractalMesh Geo-Intelligence Feed","price":"$349 AUD",
        "features":["WiGLE WiFi intelligence","NASA earth events",
                    "Copernicus satellite data","Risk overlay for traders"],
        "dashboard_section":"signals",
        "onboarding":[
            "1. Set WIGLE_API_KEY for WiFi network intelligence",
            "2. Set NASA_API_KEY (free at api.nasa.gov)",
            "3. Geo signals auto-update every 60 seconds",
            "4. NASA EONET events visible at /api/nasa",
        ],
    },
}

def _html_email(email, product, session_id, amount):
    name = product["name"]
    price = product["price"]
    features_html = "".join(f"<li>{f}</li>" for f in product["features"])
    steps_html    = "".join(f"<li>{s}</li>" for s in product.get("onboarding",[]))
    dash_url = f"http://localhost:8090/#{product.get('dashboard_section','')}"
    return f"""<!DOCTYPE html>
<html><head><style>
  body{{ font-family:'Segoe UI',sans-serif; background:#0a0a1a; color:#e0e0ff; margin:0; padding:0; }}
  .wrap{{ max-width:600px; margin:0 auto; padding:32px 16px; }}
  .header{{ background:linear-gradient(135deg,#00d2ff,#7a2ff7); padding:24px; border-radius:12px 12px 0 0; text-align:center; }}
  .header h1{{ color:#fff; margin:0; font-size:24px; }}
  .body{{ background:#141428; border:1px solid #00d2ff33; padding:24px; }}
  .footer{{ background:#0a0a1a; padding:16px; text-align:center; font-size:12px; color:#888; }}
  .badge{{ display:inline-block; background:#00d2ff22; border:1px solid #00d2ff; color:#00d2ff;
           padding:4px 12px; border-radius:20px; font-size:13px; }}
  .btn{{ display:inline-block; background:linear-gradient(135deg,#00d2ff,#7a2ff7); color:#fff;
         padding:12px 28px; border-radius:8px; text-decoration:none; font-weight:700; }}
  .order-box{{ background:#0d0d20; border:1px solid #00d2ff44; border-radius:8px; padding:16px; margin:16px 0; }}
  ul{{ color:#b0b0d0; line-height:1.8; }}
  h3{{ color:#00d2ff; }}
</style></head>
<body><div class="wrap">
  <div class="header">
    <h1>⚡ FractalMesh</h1>
    <p style="color:#cce;margin:4px 0">Sovereign AI — Albury NSW | ABN 56628117363</p>
  </div>
  <div class="body">
    <h2>Welcome aboard! Your order is active.</h2>
    <div class="order-box">
      <span class="badge">ORDER CONFIRMED</span>
      <table style="width:100%;margin-top:12px;color:#cce">
        <tr><td><b>Product</b></td><td>{name}</td></tr>
        <tr><td><b>Amount</b></td><td>{price}</td></tr>
        <tr><td><b>Session</b></td><td style="font-size:11px">{session_id}</td></tr>
        <tr><td><b>Date</b></td><td>{datetime.now().strftime('%d %b %Y %H:%M AEST')}</td></tr>
      </table>
    </div>
    <h3>What's included:</h3>
    <ul>{features_html}</ul>
    <h3>Getting started:</h3>
    <ol style="color:#b0b0d0;line-height:1.8">{steps_html}</ol>
    <div style="text-align:center;margin:24px 0">
      <a href="{dash_url}" class="btn">Open Dashboard →</a>
    </div>
    <p style="color:#888;font-size:13px">
      For support or white-label enquiries, reply to this email.<br>
      Use coupon <b>FRACTAL10</b> for 10% off your next purchase.
    </p>
  </div>
  <div class="footer">
    Samuel James Hiotis | FractalMesh | ABN 56 628 117 363<br>
    Albury NSW 2640 | Autonomous Intelligence Systems
  </div>
</div></body></html>"""

def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS delivery_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session_id TEXT UNIQUE,
            customer_email TEXT, product_name TEXT, price_id TEXT,
            amount_aud REAL, status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0, created_at TEXT, delivered_at TEXT
        )
    """)
    conn.commit()

def send_delivery(email, product, session_id, amount):
    if not GPASS:
        log.warning("No GMAIL_APP_PASS — cannot send delivery email")
        return False
    html_body = _html_email(email, product, session_id, amount)
    msg = MIMEMultipart("alternative")
    msg["From"]    = f"FractalMesh <{GMAIL}>"
    msg["To"]      = email
    msg["Subject"] = f"✅ FractalMesh — {product['name']} Active"
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as srv:
            srv.login(GMAIL, GPASS)
            srv.send_message(msg)
        log.info("Delivered → %s | %s", email, product["name"])
        return True
    except Exception as e:
        log.error("SMTP error: %s", e)
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
                "features": ["Full access confirmed — details follow"],
                "onboarding":["Visit http://localhost:8090 to access your dashboard"],
                "dashboard_section":"",
            })
            ok = send_delivery(row["customer_email"], product,
                               row["stripe_session_id"] or "MANUAL", row["amount_aud"] or 0)
            conn.execute(
                "UPDATE delivery_queue SET status=?,delivered_at=? WHERE id=?",
                ("delivered" if ok else "failed",
                 datetime.now().isoformat() if ok else None, row["id"])
            )
            conn.commit()
        conn.close()
    except Exception as e:
        log.error("Queue error: %s", e)

def main():
    log.info("fm-delivery v402 started | DB=%s", DB)
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
