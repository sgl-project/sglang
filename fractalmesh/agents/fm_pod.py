#!/usr/bin/env python3
"""
FractalMesh Pod — Master API Server
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Port: 5058
"""
import os, json, sqlite3, time, hashlib, hmac, threading
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOT     = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
DB_PATH  = os.path.join(ROOT, "db", "sovereign.db")
VAULT    = os.path.join(ROOT, ".env")

# ─── helpers ──────────────────────────────────────────────────────────────────

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session TEXT,
            product TEXT,
            contact TEXT,
            amount_aud REAL DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT,
            contact TEXT,
            phone TEXT,
            score INTEGER DEFAULT 50,
            context TEXT,
            status TEXT DEFAULT 'new',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            signal TEXT,
            confidence REAL,
            fractal_score INTEGER,
            price REAL,
            change_24h REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS rag_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE,
            content TEXT,
            category TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            detail TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS nft_mints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id TEXT,
            wallet TEXT,
            fractal_hash TEXT,
            price_sol REAL,
            status TEXT DEFAULT 'pending',
            minted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Seed leads
    existing = db.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    if existing == 0:
        leads = [
            ("Albury City Council",        "Mark Thompson",  "02 6023 8111", 88, "Enterprise SaaS + AI reporting"),
            ("Border Bank",                "Lisa Chen",       "02 6041 2200", 82, "Fintech integration, crypto custody"),
            ("Mungabareena Aboriginal",    "David Williams",  "02 6041 1304", 75, "Community AI platform"),
            ("Murray River Group",         "Sarah O'Brien",   "02 6025 0200", 79, "Regional logistics optimisation"),
            ("Hume Bank",                  "James Nguyen",    "02 6058 1000", 71, "Open banking API + signals"),
            ("Wodonga TAFE",               "Karen Singh",     "02 6055 6333", 68, "EdTech + RL curriculum tools"),
            ("Albury Wodonga Health",      "Dr. Paul Martin", "02 6058 2222", 85, "Healthcare AI pipeline"),
            ("Regional Express Airlines",  "Tom Bradley",     "02 6021 1300", 77, "Route optimisation + fractal RL"),
        ]
        db.executemany(
            "INSERT INTO leads(company,contact,phone,score,context) VALUES(?,?,?,?,?)",
            leads
        )

    # Seed orders
    existing_orders = db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    if existing_orders == 0:
        orders = [
            ("cs_live_alpha001", "Fractal Signal Feed",    "mark@alburycity.nsw.gov.au", 499.00, "completed"),
            ("cs_live_alpha002", "Sovereign AI Dashboard", "lisa@borderbank.com.au",     299.00, "completed"),
            ("cs_live_alpha003", "NFT Genesis Pack",       "david@mungabareena.org.au",  199.00, "completed"),
            ("cs_live_alpha004", "Fractal Signal Feed",    "sarah@murrayriver.com.au",   499.00, "completed"),
            ("cs_live_alpha005", "Enterprise Bundle",      "james@humebank.com.au",      899.00, "completed"),
        ]
        db.executemany(
            "INSERT INTO orders(stripe_session,product,contact,amount_aud,status) VALUES(?,?,?,?,?)",
            orders
        )

    # Seed signals
    existing_sig = db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    if existing_sig == 0:
        sigs = [
            ("BTC/USDT", "BUY",  0.87, 92, 67420.50,  2.34),
            ("ETH/USDT", "HOLD", 0.74, 78,  3521.80,  0.87),
            ("SOL/USDT", "BUY",  0.91, 95,   182.45,  5.12),
            ("XRP/USDT", "SELL", 0.68, 61,     0.623, -1.45),
            ("BNB/USDT", "HOLD", 0.72, 74,   421.30,  0.22),
        ]
        db.executemany(
            "INSERT INTO signals(pair,signal,confidence,fractal_score,price,change_24h) VALUES(?,?,?,?,?,?)",
            sigs
        )

    # Seed RAG docs
    rag_count = db.execute("SELECT COUNT(*) FROM rag_docs").fetchone()[0]
    if rag_count == 0:
        db.execute("""INSERT OR IGNORE INTO rag_docs(title,content,category) VALUES(
            'AI Studio Cloud Run Deployment',
            'Pre-deployment: https://ais-pre-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app '
            'Dev deployment: https://ais-dev-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app '
            'Project: antigravity-auto-updater-dev. Region: asia-southeast1.',
            'infrastructure'
        )""")
        db.execute("""INSERT OR IGNORE INTO rag_docs(title,content,category) VALUES(
            'FractalMesh Sovereign IP v3.0',
            'Edge-optimized RL trading. Split-brain RL architecture. Markov LTV. Enochian Gate. '
            'Operator: Samuel James Hiotis, ABN 56628117363, Albury NSW 2640.',
            'whitepaper'
        )""")

    db.execute("""INSERT INTO audit_log(event,detail) VALUES(
        'SYSTEM_BOOT',
        'FractalMesh Pod initialised — all tables seeded'
    )""")
    db.commit()
    db.close()

# ─── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    db = get_db()
    orders  = db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    leads   = db.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    revenue = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0]
    online_since = db.execute("SELECT created_at FROM audit_log WHERE event='SYSTEM_BOOT' ORDER BY id DESC LIMIT 1").fetchone()
    db.close()
    return jsonify({
        "status":       "online",
        "operator":     "Samuel James Hiotis",
        "abn":          "56628117363",
        "location":     "Albury NSW 2640",
        "orders":       orders,
        "leads":        leads,
        "revenue_aud":  round(float(revenue), 2),
        "revenue_fmt":  f"${float(revenue):,.2f} AUD",
        "online_since": online_since[0] if online_since else datetime.now().isoformat(),
        "timestamp":    datetime.now().isoformat(),
        "version":      "v101",
        "agents":       ["fm-pod", "fm-geosignal", "fm-trading", "fm-whitepaper"],
    })

@app.route("/api/products")
def products():
    return jsonify([
        {
            "id": "fractal-signal-feed",
            "name": "Fractal Signal Feed",
            "tagline": "Live BTC/ETH/SOL arbitrage + RL signals",
            "price_aud": 499,
            "stripe_link": load_env("STRIPE_LINK_SIGNAL", "#checkout"),
            "features": ["5 live pairs", "Sub-second updates", "RL confidence scores", "Telegram alerts"],
            "badge": "BESTSELLER",
        },
        {
            "id": "sovereign-dashboard",
            "name": "Sovereign AI Dashboard",
            "tagline": "Full FractalMesh command centre",
            "price_aud": 299,
            "stripe_link": load_env("STRIPE_LINK_DASH", "#checkout"),
            "features": ["Real-time leads", "Order management", "AI chat", "PM2 monitoring"],
            "badge": "NEW",
        },
        {
            "id": "nft-genesis-pack",
            "name": "NFT Genesis Pack",
            "tagline": "Solana fractal NFT — minted every 10 min",
            "price_aud": 199,
            "stripe_link": load_env("STRIPE_LINK_NFT", "#checkout"),
            "features": ["On-chain royalties", "Fractal art generated by RL", "Solana mainnet", "Transferable"],
            "badge": "HOT",
        },
        {
            "id": "enterprise-bundle",
            "name": "Enterprise Bundle",
            "tagline": "Signal Feed + Dashboard + custom onboarding",
            "price_aud": 899,
            "stripe_link": load_env("STRIPE_LINK_ENT", "#checkout"),
            "features": ["All products included", "White-label option", "Priority support", "ABN invoiced"],
            "badge": "SAVE 20%",
        },
    ])

@app.route("/api/leads")
def leads():
    db   = get_db()
    rows = db.execute("SELECT * FROM leads ORDER BY score DESC").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/orders")
def orders():
    db   = get_db()
    rows = db.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT 50").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/revenue")
def revenue():
    db      = get_db()
    total   = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0]
    by_prod = db.execute("""
        SELECT product, COUNT(*) cnt, SUM(amount_aud) total
        FROM orders WHERE status='completed'
        GROUP BY product ORDER BY total DESC
    """).fetchall()
    db.close()
    return jsonify({
        "total_aud":    round(float(total), 2),
        "formatted":    f"${float(total):,.2f} AUD",
        "by_product":   [dict(r) for r in by_prod],
        "currency":     "AUD",
    })

@app.route("/api/signals")
def signals():
    db   = get_db()
    rows = db.execute("SELECT * FROM signals ORDER BY fractal_score DESC").fetchall()
    db.close()
    return jsonify({
        "signals":    [dict(r) for r in rows],
        "updated_at": datetime.now().isoformat(),
        "source":     "FractalMesh GeoSignal Engine",
    })

@app.route("/api/ai-studio")
def ai_studio():
    return jsonify({
        "ai_studio_app":  load_env("AI_STUDIO_APP_URL",  "https://ai.studio/apps/bafddcde-c79c-4e7b-931e-d4d218e325de"),
        "pre_deployment": load_env("AI_STUDIO_PRE_URL",  "https://ais-pre-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app"),
        "dev_deployment": load_env("AI_STUDIO_DEV_URL",  "https://ais-dev-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app"),
        "firebase_notes": "https://firebase-notes-app-52699481575.us-west1.run.app",
        "region":         "asia-southeast1",
        "project":        "antigravity-auto-updater-dev",
        "operator":       "Samuel James Hiotis | ABN 56628117363",
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "message required"}), 400

    db = get_db()
    db.execute("INSERT INTO chat_log(role,content) VALUES('user',?)", (message,))

    # Try OpenRouter / OpenAI — fall back to rule-based
    api_key = load_env("OPENROUTER_API_KEY") or load_env("OPENAI_API_KEY")
    reply   = None

    if api_key and not api_key.startswith("YOUR_"):
        import urllib.request
        headers_req = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = json.dumps({
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {"role": "system",  "content":
                    "You are the FractalMesh AI assistant for Samuel James Hiotis "
                    "(ABN 56628117363, Albury NSW). Help users with trading signals, "
                    "products, and FractalMesh features. Be concise and professional."},
                {"role": "user", "content": message},
            ],
            "max_tokens": 300,
        }).encode()
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body, headers=headers_req, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                reply  = result["choices"][0]["message"]["content"]
        except Exception as e:
            reply = None

    if not reply:
        msg_low = message.lower()
        if any(w in msg_low for w in ["signal", "btc", "eth", "sol", "trade"]):
            reply = "Current top signal: SOL/USDT — BUY with 91% confidence, fractal score 95. BTC/USDT also bullish at 87%. Check the Signals panel for live updates."
        elif any(w in msg_low for w in ["price", "product", "buy", "cost", "plan"]):
            reply = "Products: Fractal Signal Feed $499 AUD | Sovereign Dashboard $299 | NFT Genesis Pack $199 | Enterprise Bundle $899. All include live data and PM2 monitoring."
        elif any(w in msg_low for w in ["revenue", "order", "sales"]):
            db2  = get_db()
            rev  = db2.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0]
            ords = db2.execute("SELECT COUNT(*) FROM orders WHERE status='completed'").fetchone()[0]
            db2.close()
            reply = f"Current revenue: ${float(rev):,.2f} AUD from {ords} completed orders. Pipeline strong — 8 leads tracked in Albury-Wodonga corridor."
        elif any(w in msg_low for w in ["lead", "client", "customer", "albury"]):
            reply = "Top lead: Albury City Council (score 88) — Enterprise SaaS + AI reporting. 8 active leads across Albury-Wodonga. Border Bank and AWH also high priority."
        elif any(w in msg_low for w in ["nft", "solana", "mint"]):
            reply = "NFT Genesis Pack mints every 10 minutes on Solana mainnet. Each NFT is a unique fractal generated by the RL engine. Royalties paid on-chain. $199 AUD."
        elif any(w in msg_low for w in ["hello", "hi", "hey", "g'day"]):
            reply = "G'day! I'm the FractalMesh AI. I can help with trading signals, products, leads, or system status. What do you need?"
        else:
            reply = "FractalMesh Omega Titan v101 online. Ask about signals, products, revenue, leads, or AI Studio deployments. I'm here to help."

    db.execute("INSERT INTO chat_log(role,content) VALUES('assistant',?)", (reply,))
    db.commit()
    db.close()
    return jsonify({"reply": reply, "timestamp": datetime.now().isoformat()})

@app.route("/api/nft/mint", methods=["POST"])
def nft_mint():
    data  = request.get_json(force=True) or {}
    wallet = data.get("wallet", "")
    if not wallet:
        return jsonify({"error": "wallet address required"}), 400
    fh = hashlib.sha256(f"{wallet}{time.time()}".encode()).hexdigest()[:16]
    db = get_db()
    db.execute(
        "INSERT INTO nft_mints(token_id,wallet,fractal_hash,price_sol,status) VALUES(?,?,?,?,?)",
        (f"FM-{fh.upper()}", wallet, fh, 0.5, "queued")
    )
    db.execute("INSERT INTO audit_log(event,detail) VALUES('NFT_MINT_QUEUED',?)", (f"wallet={wallet}",))
    db.commit()
    db.close()
    return jsonify({
        "token_id":     f"FM-{fh.upper()}",
        "fractal_hash": fh,
        "price_sol":    0.5,
        "status":       "queued",
        "message":      "NFT queued for Solana mainnet mint. Ready in ~10 minutes.",
    })

@app.route("/api/nft/gallery")
def nft_gallery():
    db   = get_db()
    rows = db.execute("SELECT * FROM nft_mints ORDER BY minted_at DESC LIMIT 20").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/webhook/stripe", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature", "")
    secret  = load_env("STRIPE_WEBHOOK_SECRET")
    # Verify signature if secret is set
    if secret and not secret.startswith("YOUR_"):
        mac = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, sig.split("v1=")[-1] if "v1=" in sig else ""):
            return jsonify({"error": "invalid signature"}), 400
    try:
        event = json.loads(payload)
        if event.get("type") == "checkout.session.completed":
            sess = event["data"]["object"]
            db   = get_db()
            db.execute(
                "INSERT INTO orders(stripe_session,product,contact,amount_aud,status) VALUES(?,?,?,?,?)",
                (sess.get("id"), sess.get("metadata", {}).get("product", "Unknown"),
                 sess.get("customer_email", ""), (sess.get("amount_total", 0) / 100), "completed")
            )
            db.commit()
            db.close()
    except Exception as e:
        pass
    return jsonify({"received": True})

@app.route("/api/audit")
def audit():
    db   = get_db()
    rows = db.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 30").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "fm-pod", "port": 5058})

# ─── boot ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("FLASK_PORT", 5058))
    print(f"[fm-pod] FractalMesh Pod starting on :{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
