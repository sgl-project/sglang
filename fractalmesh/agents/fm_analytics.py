#!/usr/bin/env python3
"""
FractalMesh Analytics Engine — Port 5060
MRR/ARR/LTV/Cohort/Funnel revenue analytics
Samuel James Hiotis | ABN 56628117363
"""
import os, sqlite3, json
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

app  = Flask(__name__)
CORS(app)

ROOT    = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
DB_PATH = os.path.join(ROOT, "db", "sovereign.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def _safe(val):
    try:
        return round(float(val), 2)
    except Exception:
        return 0.0

@app.route("/health")
def health():
    return jsonify({"status":"ok","service":"fm-analytics","port":5060,
                    "timestamp":datetime.now().isoformat()})

@app.route("/api/analytics/mrr")
def mrr():
    db  = get_db()
    mrr_val  = _safe(db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM subscriptions WHERE status='active'").fetchone()[0])
    arr_val  = round(mrr_val * 12, 2)
    subs     = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
    churned  = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='cancelled'").fetchone()[0]
    by_plan  = db.execute("SELECT plan,COUNT(*) cnt,SUM(amount_aud) mrr FROM subscriptions WHERE status='active' GROUP BY plan ORDER BY mrr DESC").fetchall()
    one_off  = _safe(db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0])
    db.close()
    churn_r  = round(churned / max(subs+churned,1)*100, 1)
    ltv_avg  = round(mrr_val / max(subs,1) / max(churn_r/100,0.01), 2)
    return jsonify({
        "mrr_aud":mrr_val,"arr_aud":arr_val,
        "mrr_fmt":f"${mrr_val:,.2f}/mo","arr_fmt":f"${arr_val:,.2f}/yr",
        "active_subs":subs,"churned_subs":churned,"churn_rate_pct":churn_r,
        "avg_ltv_aud":ltv_avg,"one_off_aud":one_off,
        "total_revenue_aud":round(one_off+arr_val,2),
        "by_plan":[dict(r) for r in by_plan],
        "generated_at":datetime.now().isoformat(),
    })

@app.route("/api/analytics/funnel")
def funnel():
    """Leads → Contacted → Demo → Trial → Paid → Champion."""
    db = get_db()
    leads_total  = db.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    leads_hot    = db.execute("SELECT COUNT(*) FROM leads WHERE score >= 75").fetchone()[0]
    leads_active = db.execute("SELECT COUNT(*) FROM leads WHERE status != 'new'").fetchone()[0]
    orders_total = db.execute("SELECT COUNT(*) FROM orders WHERE status='completed'").fetchone()[0]
    subs_active  = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
    db.close()
    conv_lead_to_order = round(orders_total / max(leads_total,1)*100, 1)
    conv_order_to_sub  = round(subs_active / max(orders_total,1)*100, 1)
    return jsonify({
        "funnel":[
            {"stage":"Leads Total",    "count":leads_total,  "pct":100},
            {"stage":"Hot Leads (≥75)","count":leads_hot,    "pct":round(leads_hot/max(leads_total,1)*100,1)},
            {"stage":"Active Leads",   "count":leads_active, "pct":round(leads_active/max(leads_total,1)*100,1)},
            {"stage":"Paying Orders",  "count":orders_total, "pct":conv_lead_to_order},
            {"stage":"Active Subs",    "count":subs_active,  "pct":round(subs_active/max(leads_total,1)*100,1)},
        ],
        "conversion_lead_to_order_pct":conv_lead_to_order,
        "conversion_order_to_sub_pct": conv_order_to_sub,
        "generated_at":datetime.now().isoformat(),
    })

@app.route("/api/analytics/cohorts")
def cohorts():
    """Monthly order cohorts for retention analysis."""
    db   = get_db()
    rows = db.execute("""
        SELECT strftime('%Y-%m', created_at) month,
               COUNT(*) orders, SUM(amount_aud) revenue
        FROM orders WHERE status='completed'
        GROUP BY month ORDER BY month DESC LIMIT 12
    """).fetchall()
    db.close()
    return jsonify({
        "cohorts": [dict(r) for r in rows],
        "periods": len(rows),
        "generated_at": datetime.now().isoformat(),
    })

@app.route("/api/analytics/ltv")
def ltv():
    """LTV per plan and overall."""
    db = get_db()
    mrr = _safe(db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM subscriptions WHERE status='active'").fetchone()[0])
    subs = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
    churn = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='cancelled'").fetchone()[0]
    by_plan = db.execute("SELECT plan,COUNT(*) cnt,AVG(amount_aud) avg_mo FROM subscriptions WHERE status='active' GROUP BY plan").fetchall()
    avg_order = _safe(db.execute("SELECT AVG(amount_aud) FROM orders WHERE status='completed'").fetchone()[0])
    db.close()
    churn_r  = churn / max(subs+churn, 1)
    avg_mo   = mrr / max(subs, 1)
    avg_ltv  = round(avg_mo / max(churn_r, 0.01), 2) if churn_r > 0 else round(avg_mo * 24, 2)
    plan_ltv = []
    for r in by_plan:
        plan_churn = max(churn_r, 0.05)
        plan_ltv.append({
            "plan": r["plan"],
            "subscribers": r["cnt"],
            "avg_mo_aud": round(float(r["avg_mo"]),2),
            "ltv_aud": round(float(r["avg_mo"]) / plan_churn, 2),
        })
    return jsonify({
        "avg_ltv_aud": avg_ltv,
        "avg_order_aud": avg_order,
        "churn_rate_pct": round(churn_r*100,1),
        "by_plan": plan_ltv,
        "generated_at": datetime.now().isoformat(),
    })

@app.route("/api/analytics/summary")
def summary():
    """All-in-one dashboard summary."""
    db  = get_db()
    mrr = _safe(db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM subscriptions WHERE status='active'").fetchone()[0])
    rev = _safe(db.execute("SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'").fetchone()[0])
    subs   = db.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
    leads  = db.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    orders = db.execute("SELECT COUNT(*) FROM orders WHERE status='completed'").fetchone()[0]
    nfts   = db.execute("SELECT COUNT(*) FROM nft_mints").fetchone()[0]
    top_product = db.execute("SELECT product,SUM(amount_aud) total FROM orders WHERE status='completed' GROUP BY product ORDER BY total DESC LIMIT 1").fetchone()
    db.close()
    return jsonify({
        "revenue": {"one_off_aud":rev,"mrr_aud":mrr,"arr_aud":round(mrr*12,2),
                    "total_aud":round(rev+mrr*12,2)},
        "customers":{"total_orders":orders,"active_subs":subs,"leads_pipeline":leads},
        "nfts":     {"minted":nfts},
        "top_product":dict(top_product) if top_product else {},
        "health":   {"status":"optimal","agents":14,"memory_budget_mb":250},
        "generated_at":datetime.now().isoformat(),
    })

@app.route("/api/analytics/pnl")
def pnl():
    """Trading P&L summary."""
    db = get_db()
    try:
        total_pnl = db.execute("SELECT COALESCE(SUM(pnl_usd),0) FROM trade_pnl").fetchone()[0]
        by_pair   = db.execute("SELECT pair,COUNT(*) trades,SUM(pnl_usd) pnl FROM trade_pnl GROUP BY pair ORDER BY pnl DESC").fetchall()
        last_snap = db.execute("SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        db.close()
        return jsonify({
            "total_pnl_usd":  round(float(total_pnl),2),
            "by_pair":        [dict(r) for r in by_pair],
            "last_snapshot":  dict(last_snap) if last_snap else None,
            "generated_at":   datetime.now().isoformat(),
        })
    except Exception:
        db.close()
        return jsonify({"total_pnl_usd":0,"by_pair":[],"note":"trade_pnl table not yet created — run fm-trading first"})

if __name__ == "__main__":
    port = int(os.environ.get("ANALYTICS_PORT", 5060))
    print(f"[fm-analytics] Analytics engine starting on :{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
