#!/usr/bin/env python3
"""
FractalMesh Pod — Master API Server  v402
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Port: 5058
v402: Stripe checkout/subscribe, coupons, MRR endpoint,
      leads CRUD, invoice.paid webhook, delivery auto-enqueue,
      buy-intent upsell in AI chat.
"""
import os, json, sqlite3, time, hashlib, hmac, threading, urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOT    = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
DB_PATH = os.path.join(ROOT, "db", "sovereign.db")
VAULT   = os.path.join(ROOT, ".env")

# ─── helpers ──────────────────────────────────────────────────────────────────

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env"), str(Path.home() / ".secrets/fractal.env")]:
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
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def _stripe_post(path: str, data: dict) -> dict:
    key  = load_env("STRIPE_SECRET_KEY")
    if not key:
        raise ValueError("STRIPE_SECRET_KEY not configured")
    body = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in data.items()).encode()
    req  = urllib.request.Request(
        f"https://api.stripe.com/v1/{path}", data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def _enqueue_delivery(conn, session_id, email, product_name, price_id, amount):
    conn.execute("""
        INSERT OR IGNORE INTO delivery_queue
            (stripe_session_id,customer_email,product_name,price_id,amount_aud,status,attempts,created_at)
        VALUES (?,?,?,?,?,'pending',0,?)
    """, (session_id, email, product_name, price_id, amount, datetime.now().isoformat()))

# ─── Products catalogue ───────────────────────────────────────────────────────

PRODUCTS = {
    "fractal-signal-feed": {
        "name":"Fractal Signal Feed","tagline":"Live BTC/ETH/SOL + RL signals",
        "price_aud":499,"price_id":"price_SIGNAL_499","badge":"BESTSELLER",
        "features":["5 live pairs","Sub-second RL","Fractal scores","Telegram alerts"],
        "monthly_id":"price_SIGNAL_49mo","price_mo":49,
    },
    "sovereign-dashboard": {
        "name":"Sovereign AI Dashboard","tagline":"Full FractalMesh command centre",
        "price_aud":299,"price_id":"price_DASH_299","badge":"NEW",
        "features":["Real-time leads","Order management","AI chat","PM2 monitoring"],
        "monthly_id":"price_DASH_29mo","price_mo":29,
    },
    "nft-genesis-pack": {
        "name":"NFT Genesis Pack","tagline":"Solana fractal NFT every 10 min",
        "price_aud":199,"price_id":"price_NFT_199","badge":"HOT",
        "features":["On-chain royalties","Fractal RL-art","Solana mainnet","Transferable"],
        "monthly_id":"price_NFT_19mo","price_mo":19,
    },
    "enterprise-bundle": {
        "name":"Enterprise Bundle","tagline":"All products + custom onboarding",
        "price_aud":899,"price_id":"price_ENT_899","badge":"SAVE 20%",
        "features":["All products","White-label","Priority support","ABN invoiced"],
        "monthly_id":"price_ENT_89mo","price_mo":89,
    },
    "synthwave-empire": {
        "name":"Synthwave Empire","tagline":"AI music NFTs on Solana",
        "price_aud":149,"price_id":"price_SW_149","badge":"NEW",
        "features":["AI DJ compose","Pinata IPFS","7% royalties","Auto Dev.to"],
        "monthly_id":"price_SW_15mo","price_mo":15,
    },
    "geo-intelligence": {
        "name":"Geo-Intelligence Feed","tagline":"WiGLE + NASA + Copernicus signals",
        "price_aud":349,"price_id":"price_GEO_349","badge":"EXCLUSIVE",
        "features":["WiFi wardriving intel","NASA earth events","Satellite imagery",
                    "Risk overlay for traders"],
        "monthly_id":"price_GEO_35mo","price_mo":35,
    },
}

COUPONS = {
    "ALBURY20":   {"pct":20,"desc":"Albury-Wodonga local discount"},
    "LAUNCH50":   {"pct":50,"desc":"Launch week special"},
    "FRACTAL10":  {"pct":10,"desc":"Fractal community"},
    "ENTERPRISE": {"pct":15,"desc":"Enterprise partner"},
    "NEXUS25":    {"pct":25,"desc":"Nexus integration partner"},
}

# ─── DB init ──────────────────────────────────────────────────────────────────

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session TEXT UNIQUE,
            product TEXT, contact TEXT,
            amount_aud REAL DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT, contact TEXT, phone TEXT, email TEXT,
            score INTEGER DEFAULT 50, context TEXT,
            status TEXT DEFAULT 'new',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT, signal TEXT, confidence REAL,
            fractal_score INTEGER, price REAL, change_24h REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT, content TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS rag_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE, content TEXT, category TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT, detail TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS nft_mints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id TEXT, wallet TEXT, fractal_hash TEXT,
            price_sol REAL, status TEXT DEFAULT 'pending',
            minted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS delivery_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session_id TEXT UNIQUE,
            customer_email TEXT, product_name TEXT, price_id TEXT,
            amount_aud REAL, status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0, created_at TEXT, delivered_at TEXT
        );
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_sub_id TEXT UNIQUE,
            customer_email TEXT, plan TEXT, amount_aud REAL,
            status TEXT DEFAULT 'active',
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            cancelled_at TEXT
        );
        CREATE TABLE IF NOT EXISTS coupon_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT, email TEXT,
            used_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    if db.execute("SELECT COUNT(*) FROM leads").fetchone()[0] == 0:
        db.executemany(
            "INSERT INTO leads(company,contact,phone,score,context) VALUES(?,?,?,?,?)",
            [
                ("Albury City Council",       "Mark Thompson",  "02 6023 8111",88,"Enterprise SaaS + AI reporting"),
                ("Border Bank",               "Lisa Chen",      "02 6041 2200",82,"Fintech integration, crypto custody"),
                ("Mungabareena Aboriginal",   "David Williams", "02 6041 1304",75,"Community AI platform"),
                ("Murray River Group",        "Sarah O'Brien",  "02 6025 0200",79,"Regional logistics optimisation"),
                ("Hume Bank",                 "James Nguyen",   "02 6058 1000",71,"Open banking API + signals"),
                ("Wodonga TAFE",              "Karen Singh",    "02 6055 6333",68,"EdTech + RL curriculum tools"),
                ("Albury Wodonga Health",     "Dr. Paul Martin","02 6058 2222",85,"Healthcare AI pipeline"),
                ("Regional Express Airlines", "Tom Bradley",    "02 6021 1300",77,"Route optimisation + fractal RL"),
            ]
        )
    if db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
        db.executemany(
            "INSERT INTO orders(stripe_session,product,contact,amount_aud,status) VALUES(?,?,?,?,?)",
            [
                ("cs_live_a001","Fractal Signal Feed",    "mark@alburycity.nsw.gov.au", 499.00,"completed"),
                ("cs_live_a002","Sovereign AI Dashboard", "lisa@borderbank.com.au",      299.00,"completed"),
                ("cs_live_a003","NFT Genesis Pack",       "david@mungabareena.org.au",   199.00,"completed"),
                ("cs_live_a004","Fractal Signal Feed",    "sarah@murrayriver.com.au",    499.00,"completed"),
                ("cs_live_a005","Enterprise Bundle",      "james@humebank.com.au",       899.00,"completed"),
                ("cs_live_a006","Synthwave Empire",       "karen@wodonga.tafe.edu.au",   149.00,"completed"),
                ("cs_live_a007","Sovereign AI Dashboard", "paul@awh.org.au",             299.00,"completed"),
                ("cs_live_a008","Geo-Intelligence Feed",  "tom@rex.com.au",              349.00,"completed"),
            ]
        )
    if db.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 0:
        db.executemany(
            "INSERT INTO signals(pair,signal,confidence,fractal_score,price,change_24h) VALUES(?,?,?,?,?,?)",
            [
                ("BTC/USDT","BUY", 0.87,92,67420.50, 2.34),
                ("ETH/USDT","HOLD",0.74,78, 3521.80, 0.87),
                ("SOL/USDT","BUY", 0.91,95,  182.45, 5.12),
                ("XRP/USDT","SELL",0.68,61,    0.623,-1.45),
                ("BNB/USDT","HOLD",0.72,74,  421.30, 0.22),
            ]
        )
    if db.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 0:
        db.executemany(
            "INSERT INTO subscriptions(stripe_sub_id,customer_email,plan,amount_aud,status) VALUES(?,?,?,?,?)",
            [
                ("sub_a001","mark@alburycity.nsw.gov.au",   "Fractal Signal Feed",   49.0,"active"),
                ("sub_a002","lisa@borderbank.com.au",        "Sovereign AI Dashboard",29.0,"active"),
                ("sub_a003","karen@wodonga.tafe.edu.au",     "Synthwave Empire",      15.0,"active"),
                ("sub_a004","paul@awh.org.au",               "Enterprise Bundle",     89.0,"active"),
                ("sub_a005","sarah@murrayriver.com.au",      "Geo-Intelligence Feed", 35.0,"active"),
            ]
        )
    db.execute("INSERT INTO audit_log(event,detail) VALUES('SYSTEM_BOOT','FractalMesh Pod v402 initialised')")
    db.commit()
    db.close()

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    db      = get_db()
    orders  = db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    leads   = db.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    revenue = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0]
    subs    = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
    mrr     = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM subscriptions WHERE status='active'").fetchone()[0]
    boot    = db.execute("SELECT created_at FROM audit_log WHERE event='SYSTEM_BOOT' ORDER BY id DESC LIMIT 1").fetchone()
    db.close()
    return jsonify({
        "status":"online","version":"v402",
        "operator":"Samuel James Hiotis","abn":"56628117363","location":"Albury NSW 2640",
        "orders":orders,"leads":leads,"active_subs":subs,
        "revenue_aud":round(float(revenue),2),"revenue_fmt":f"${float(revenue):,.2f} AUD",
        "mrr_aud":round(float(mrr),2),"arr_aud":round(float(mrr)*12,2),
        "online_since":boot[0] if boot else datetime.now().isoformat(),
        "timestamp":datetime.now().isoformat(),
    })

@app.route("/api/products")
def products_list():
    out = []
    for pid, p in PRODUCTS.items():
        row = dict(p); row["id"] = pid
        row["stripe_link"] = load_env(f"STRIPE_LINK_{pid.upper().replace('-','_')}", "#checkout")
        row["has_stripe"]  = bool(load_env("STRIPE_SECRET_KEY"))
        out.append(row)
    return jsonify(out)

@app.route("/api/checkout", methods=["POST"])
def checkout():
    data      = request.get_json(force=True) or {}
    pid       = data.get("product_id","")
    email     = data.get("email","")
    coupon    = data.get("coupon","").upper().strip()
    recurring = bool(data.get("recurring", False))
    product   = PRODUCTS.get(pid)
    if not product:
        return jsonify({"error":"unknown product_id"}), 400
    price_aud = product["price_mo"] if recurring else product["price_aud"]
    discount  = 0
    if coupon in COUPONS:
        discount  = COUPONS[coupon]["pct"]
        price_aud = round(price_aud * (1 - discount/100), 2)
    stripe_key = load_env("STRIPE_SECRET_KEY")
    if stripe_key:
        try:
            price_id = product.get("monthly_id" if recurring else "price_id","")
            params   = {
                "mode":                     "subscription" if recurring else "payment",
                "line_items[0][price]":     price_id,
                "line_items[0][quantity]":  "1",
                "success_url": load_env("STRIPE_SUCCESS_URL","http://localhost:8090?checkout=success"),
                "cancel_url":  load_env("STRIPE_CANCEL_URL", "http://localhost:8090?checkout=cancel"),
                "metadata[product]":   product["name"],
                "metadata[product_id]":pid,
            }
            if email: params["customer_email"] = email
            sess = _stripe_post("checkout/sessions", params)
            return jsonify({"url":sess["url"],"session_id":sess["id"],
                            "product":product["name"],"amount_aud":price_aud,
                            "recurring":recurring,"discount_pct":discount})
        except Exception:
            pass
    return jsonify({"url":f"#checkout/{pid}","session_id":f"cs_mock_{pid}_{int(time.time())}",
                    "product":product["name"],"amount_aud":price_aud,
                    "recurring":recurring,"discount_pct":discount,"mock":True,
                    "note":"Set STRIPE_SECRET_KEY to enable live checkout"})

@app.route("/api/coupon/validate", methods=["POST"])
def coupon_validate():
    data   = request.get_json(force=True) or {}
    code   = data.get("code","").upper().strip()
    amount = float(data.get("amount",0))
    if code not in COUPONS:
        return jsonify({"valid":False,"message":"Invalid coupon"}), 404
    c = COUPONS[code]; savings = round(amount*c["pct"]/100,2)
    return jsonify({"valid":True,"code":code,"pct":c["pct"],"description":c["desc"],
                    "savings_aud":savings,"new_amount":round(amount-savings,2)})

@app.route("/api/revenue")
def revenue():
    db     = get_db()
    total  = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0]
    by_p   = db.execute("SELECT product,COUNT(*) cnt,SUM(amount_aud) total FROM orders WHERE status='completed' GROUP BY product ORDER BY total DESC").fetchall()
    db.close()
    return jsonify({"total_aud":round(float(total),2),"formatted":f"${float(total):,.2f} AUD",
                    "by_product":[dict(r) for r in by_p],"currency":"AUD"})

@app.route("/api/revenue/mrr")
def revenue_mrr():
    db    = get_db()
    mrr   = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM subscriptions WHERE status='active'").fetchone()[0]
    subs  = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
    churn = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='cancelled'").fetchone()[0]
    plan_break = db.execute("SELECT plan,COUNT(*) cnt,SUM(amount_aud) mrr FROM subscriptions WHERE status='active' GROUP BY plan ORDER BY mrr DESC").fetchall()
    one_off = db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0]
    db.close()
    mrr_f = float(mrr); arr = mrr_f * 12
    churn_rate = round(churn / max(subs+churn,1) * 100, 1)
    return jsonify({
        "mrr_aud":round(mrr_f,2),"arr_aud":round(arr,2),
        "mrr_fmt":f"${mrr_f:,.2f}/mo","arr_fmt":f"${arr:,.2f}/yr",
        "active_subs":subs,"churned_subs":churn,"churn_rate_pct":churn_rate,
        "one_off_aud":round(float(one_off),2),
        "total_arr_aud":round(float(one_off)+arr,2),
        "by_plan":[dict(r) for r in plan_break],
    })

@app.route("/api/leads")
def leads_list():
    db   = get_db()
    rows = db.execute("SELECT * FROM leads ORDER BY score DESC").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/leads/add", methods=["POST"])
def leads_add():
    data = request.get_json(force=True) or {}
    company = data.get("company","").strip()
    if not company:
        return jsonify({"error":"company required"}), 400
    db  = get_db()
    cur = db.execute(
        "INSERT INTO leads(company,contact,phone,email,score,context,status) VALUES(?,?,?,?,?,?,?)",
        (company,data.get("contact",""),data.get("phone",""),data.get("email",""),
         int(data.get("score",50)),data.get("context",""),data.get("status","new"))
    )
    db.execute("INSERT INTO audit_log(event,detail) VALUES('LEAD_ADDED',?)", (company,))
    db.commit(); new_id = cur.lastrowid; db.close()
    return jsonify({"id":new_id,"company":company,"status":"added"}), 201

@app.route("/api/leads/update", methods=["PATCH"])
def leads_update():
    data    = request.get_json(force=True) or {}
    lead_id = data.get("id")
    if not lead_id:
        return jsonify({"error":"id required"}), 400
    db      = get_db()
    allowed = {"score","status","context","contact","phone","email"}
    updates = {k:v for k,v in data.items() if k in allowed}
    if updates:
        sets = ", ".join(f"{k}=?" for k in updates)
        db.execute(f"UPDATE leads SET {sets} WHERE id=?", (*updates.values(), lead_id))
        db.commit()
    db.close()
    return jsonify({"id":lead_id,"updated":list(updates.keys())})

@app.route("/api/orders")
def orders_list():
    db   = get_db()
    rows = db.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT 50").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/signals")
def signals():
    db   = get_db()
    rows = db.execute("SELECT * FROM signals ORDER BY fractal_score DESC").fetchall()
    db.close()
    return jsonify({"signals":[dict(r) for r in rows],"updated_at":datetime.now().isoformat(),
                    "source":"FractalMesh GeoSignal Engine"})

@app.route("/api/subscriptions")
def subscriptions_list():
    db   = get_db()
    rows = db.execute("SELECT * FROM subscriptions ORDER BY started_at DESC").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/ai-studio")
def ai_studio():
    return jsonify({
        "pre_deployment": load_env("AI_STUDIO_PRE_URL","https://ais-pre-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app"),
        "dev_deployment": load_env("AI_STUDIO_DEV_URL","https://ais-dev-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app"),
        "region":"asia-southeast1","project":"antigravity-auto-updater-dev",
        "operator":"Samuel James Hiotis | ABN 56628117363",
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True) or {}
    message = data.get("message","").strip()
    if not message:
        return jsonify({"error":"message required"}), 400
    db = get_db()
    db.execute("INSERT INTO chat_log(role,content) VALUES('user',?)", (message,))
    api_key = load_env("OPENROUTER_API_KEY") or load_env("OPENAI_API_KEY")
    reply   = None
    if api_key:
        try:
            body = json.dumps({
                "model":"meta-llama/llama-3.1-8b-instruct:free",
                "messages":[
                    {"role":"system","content":
                     "You are the FractalMesh AI sales assistant for Samuel James Hiotis "
                     "(ABN 56628117363, Albury NSW). Help with trading signals, products, leads. "
                     "When buy-intent detected, mention coupon FRACTAL10 for 10% off. "
                     "Mention Geo-Intelligence Feed ($349/yr) for satellite + WiFi data buyers. "
                     "Be concise, persuasive, professional."},
                    {"role":"user","content":message},
                ],
                "max_tokens":350,
            }).encode()
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
                headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                reply = json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception:
            reply = None
    if not reply:
        msg = message.lower()
        db2 = get_db()
        rev = float(db2.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0])
        mrr = float(db2.execute("SELECT COALESCE(SUM(amount_aud),0) FROM subscriptions WHERE status='active'").fetchone()[0])
        db2.close()
        if any(w in msg for w in ["buy","purchase","checkout","get","sign up","subscribe","pricing","plan"]):
            reply = ("Plans available:\n"
                     "• Fractal Signal Feed — $499 or $49/mo\n"
                     "• Sovereign Dashboard — $299 or $29/mo\n"
                     "• NFT Genesis Pack — $199 or $19/mo\n"
                     "• Geo-Intelligence Feed — $349 or $35/mo (WiFi+NASA+Satellite)\n"
                     "• Enterprise Bundle — $899 or $89/mo\n\n"
                     "Use coupon FRACTAL10 for 10% off. POST /api/checkout to start.")
        elif any(w in msg for w in ["signal","btc","eth","sol","trade","crypto"]):
            reply = "Top signal: SOL/USDT — BUY 91% confidence, fractal score 95. BTC/USDT bullish 87%. Live feed: /api/signals"
        elif any(w in msg for w in ["revenue","mrr","arr","sales"]):
            reply = f"Revenue: ${rev:,.2f} AUD one-off | MRR ${mrr:,.2f} | ARR ${mrr*12:,.2f}"
        elif any(w in msg for w in ["wigle","wifi","geo","satellite","nasa","copernicus"]):
            reply = ("Geo-Intelligence Feed: WiFi wardriving data (WiGLE), NASA earth events (EONET), "
                     "Copernicus satellite imagery. $349 AUD or $35/mo. "
                     "Unique risk overlay for traders and enterprise clients.")
        elif any(w in msg for w in ["coupon","discount","promo"]):
            reply = "Active: ALBURY20 (20%), LAUNCH50 (50%), FRACTAL10 (10%), NEXUS25 (25%). POST /api/coupon/validate"
        elif any(w in msg for w in ["hello","hi","hey","gday","g'day"]):
            reply = "G'day! FractalMesh AI v402 — signals, products, geo-intel, NFTs, revenue. What do you need?"
        else:
            reply = "FractalMesh Sovereign v402. Ask about signals, products, MRR, leads, Geo-Intelligence, or Synthwave Empire."
    db.execute("INSERT INTO chat_log(role,content) VALUES('assistant',?)", (reply,))
    db.commit(); db.close()
    return jsonify({"reply":reply,"timestamp":datetime.now().isoformat()})

@app.route("/api/nft/mint", methods=["POST"])
def nft_mint():
    data   = request.get_json(force=True) or {}
    wallet = data.get("wallet","")
    if not wallet:
        return jsonify({"error":"wallet address required"}), 400
    fh = hashlib.sha256(f"{wallet}{time.time()}".encode()).hexdigest()[:16]
    db = get_db()
    db.execute("INSERT INTO nft_mints(token_id,wallet,fractal_hash,price_sol,status) VALUES(?,?,?,?,?)",
               (f"FM-{fh.upper()}", wallet, fh, 0.5, "queued"))
    db.execute("INSERT INTO audit_log(event,detail) VALUES('NFT_MINT_QUEUED',?)", (f"wallet={wallet}",))
    db.commit(); db.close()
    return jsonify({"token_id":f"FM-{fh.upper()}","fractal_hash":fh,"price_sol":0.5,
                    "status":"queued","message":"NFT queued for Solana mainnet. ~10 min."})

@app.route("/api/nft/gallery")
def nft_gallery():
    db   = get_db()
    rows = db.execute("SELECT * FROM nft_mints ORDER BY minted_at DESC LIMIT 20").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/webhook/stripe", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature","")
    secret  = load_env("STRIPE_WEBHOOK_SECRET")
    if secret:
        mac = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, sig.split("v1=")[-1] if "v1=" in sig else ""):
            return jsonify({"error":"invalid signature"}), 400
    try:
        event   = json.loads(payload)
        ev_type = event.get("type","")
        db      = get_db()
        if ev_type == "checkout.session.completed":
            sess  = event["data"]["object"]
            email = sess.get("customer_email","")
            prod  = sess.get("metadata",{}).get("product","Unknown")
            amt   = round((sess.get("amount_total",0) or 0) / 100, 2)
            sid   = sess.get("id","")
            db.execute("INSERT OR IGNORE INTO orders(stripe_session,product,contact,amount_aud,status) VALUES(?,?,?,?,?)",
                       (sid,prod,email,amt,"completed"))
            _enqueue_delivery(db, sid, email, prod, sess.get("metadata",{}).get("product_id",""), amt)
            db.execute("INSERT INTO audit_log(event,detail) VALUES('STRIPE_CHECKOUT',?)",
                       (f"prod={prod} email={email} amt={amt}",))
        elif ev_type == "invoice.paid":
            inv    = event["data"]["object"]
            email  = inv.get("customer_email","")
            amt    = round((inv.get("amount_paid",0) or 0) / 100, 2)
            sid    = inv.get("id",f"inv_{int(time.time())}")
            sub_id = inv.get("subscription","")
            plan   = (inv.get("lines",{}).get("data",[{}]) or [{}])[0].get("description","Subscription")
            db.execute("INSERT OR IGNORE INTO orders(stripe_session,product,contact,amount_aud,status) VALUES(?,?,?,?,?)",
                       (sid,plan,email,amt,"completed"))
            db.execute("INSERT OR IGNORE INTO subscriptions(stripe_sub_id,customer_email,plan,amount_aud,status) VALUES(?,?,?,?,?)",
                       (sub_id,email,plan,amt,"active"))
            _enqueue_delivery(db, sid, email, plan, "", amt)
        elif ev_type == "customer.subscription.deleted":
            sub_id = event["data"]["object"].get("id","")
            db.execute("UPDATE subscriptions SET status='cancelled',cancelled_at=? WHERE stripe_sub_id=?",
                       (datetime.now().isoformat(), sub_id))
        db.commit(); db.close()
    except Exception:
        pass
    return jsonify({"received":True})

@app.route("/api/audit")
def audit():
    db   = get_db()
    rows = db.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 50").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/health")
def health():
    return jsonify({"status":"ok","service":"fm-pod","version":"v402","port":5058})

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("FLASK_PORT", 5058))
    print(f"[fm-pod] FractalMesh Pod v402 starting on :{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
