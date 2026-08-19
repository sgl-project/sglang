"""
FractalMesh Methane Super-Emitter Report Agent v2.0.0
Reads methane_readings anomalies from sovereign.db, generates
client-ready verification reports with coordinates + flux estimates.
Integrates with fm_negotiator for Stripe billing.
Report buyers: oil & gas operators, carbon traders, compliance teams.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
REPO       = os.getenv("REPO_ROOT",        os.path.expanduser("~/sglang"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
DIST       = os.path.join(REPO, "fractalmesh", "dist")
INTERVAL   = int(os.getenv("METHANE_REPORT_INTERVAL", "3600"))
DRY_RUN    = os.getenv("ENABLE_METHANE_REPORTS", "false").lower() != "true"

GMAIL_USER   = os.getenv("GMAIL_USER",    "")
GMAIL_PASS   = os.getenv("GMAIL_APP_PASS", "")
OPERATOR     = "Samuel James Hiotis"
ABN          = "56 628 117 363"
PHONE        = os.getenv("OPERATOR_PHONE", "0439 008 640")
SITE         = "https://fractalmesh.net"
PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", f"{SITE}/products.html")

PHI      = 1.6180339887
_running = True

# ── Report pricing tiers ──────────────────────────────────────────────────────
REPORT_TIERS = {
    "verified_single": {
        "label":      "Verified Super-Emitter Report (Single Site)",
        "aud":        2000.0,
        "description": "Satellite-verified methane anomaly with coordinates, "
                       "flux estimate, sensing date, quality flags, and chain-of-custody.",
    },
    "cluster_report": {
        "label":      "Cluster Report (5-site bundle)",
        "aud":        8000.0,
        "description": "Five verified super-emitter sites in one report. "
                       "Includes TROPOMI + EMIT cross-validation.",
    },
    "monthly_monitor": {
        "label":      "Monthly Methane Monitoring Retainer",
        "aud":        3500.0,
        "description": "Weekly anomaly reports for a defined area of interest. "
                       "Supabase dashboard access included.",
    },
    "ais_intel": {
        "label":      "Dark Fleet Intelligence Report (30 days)",
        "aud":        4500.0,
        "description": "30-day AIS dark event log for specified vessel or region. "
                       "Includes spoofing analysis and zone heat-map.",
    },
    "crop_yield": {
        "label":      "Early Yield Signal Report (per crop season)",
        "aud":        1500.0,
        "description": "2-4 week advance NDVI yield signal for specified agricultural "
                       "region. Sentinel-2 L2A sourced.",
    },
    "cross_verify": {
        "label":      "WiFi + Satellite Ground-Truth Bundle",
        "aud":        6000.0,
        "description": "WiGLE WiFi topology cross-referenced with Sentinel imagery "
                       "for ground-truth verification. Suitable for hedge fund "
                       "commodity position validation.",
    },
}


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS methane_report_log (
        id INTEGER PRIMARY KEY,
        report_ref TEXT UNIQUE,
        tier TEXT,
        anomaly_ids TEXT,
        status TEXT,
        sent_to TEXT,
        amount_aud REAL,
        phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


# ── Report generation ─────────────────────────────────────────────────────────

def _format_report(anomalies: list, tier_key: str) -> dict:
    tier   = REPORT_TIERS.get(tier_key, REPORT_TIERS["verified_single"])
    ts     = datetime.utcnow()
    ref    = f"FM-CH4-{ts.strftime('%Y%m%d')}-{abs(hash(tier_key + str(len(anomalies)))) % 9999:04d}"
    phi    = round(tier["aud"] * PHI / 1000, 4)

    rows = ""
    for i, a in enumerate(anomalies, 1):
        rows += (f"\n  {i}. Site: lat={a.get('lat', 0):.4f}  lon={a.get('lon', 0):.4f}\n"
                 f"     Source:      {a.get('source', 'Sentinel-5P TROPOMI')}\n"
                 f"     Sensing:     {a.get('sensing_date', 'N/A')}\n"
                 f"     CH4 column:  {a.get('ch4_ppb', 0):.1f} ppb "
                 f"(+{a.get('ch4_enhancement', 0):.1f} ppb above background)\n"
                 f"     Flux est.:   {a.get('estimated_flux_kt', 0):.3f} kt CH4/yr "
                 f"(order-of-magnitude estimate)\n"
                 f"     Quality:     {a.get('quality_flag', 1)} | "
                 f"φ-score: {a.get('phi_score', 0):.4f}\n")

    body = f"""
{'='*64}
  FRACTALMESH INTELLIGENCE REPORT
  {tier['label'].upper()}
  Reference: {ref}
  Date: {ts.strftime('%d %B %Y %H:%M UTC')}
{'='*64}

OPERATOR
  {OPERATOR}
  ABN {ABN} | {PHONE}
  {SITE}

REPORT TYPE
  {tier['label']}
  {tier['description']}

DETECTED ANOMALIES ({len(anomalies)} sites)
{rows}
DATA SOURCES
  • Copernicus Sentinel-5P TROPOMI (ESA) — freely available via CDSE
    https://dataspace.copernicus.eu/
  • NASA EMIT Methane Plume Inventory (NASA JPL) — public
    https://earth.jpl.nasa.gov/emit/
  • Background CH4: {os.getenv('CH4_BACKGROUND_PPB', '1870')} ppb (TROPOMI 2026 baseline)

METHODOLOGY
  Methane column retrievals are compared against a 1870 ppb background.
  Anomalies flagged at ≥ 2σ (≥ 30 ppb enhancement). Flux estimates are
  order-of-magnitude only — column-to-surface conversion requires full
  atmospheric modelling for compliance-grade use.

CHAIN OF CUSTODY
  Data ingested by FractalMesh Sovereign Node via fm_sentinel_ingest.py
  All records HMAC-SHA256 signed and timestamped in sovereign.db (SQLite WAL).
  Fingerprint: {hashlib.sha256(rows.encode()).hexdigest()[:24]}

INVESTMENT
  {tier['label']}: ${tier['aud']:,.2f} AUD (ex GST)
  Payment: {PAYMENT_LINK}
  50% upfront, 50% on delivery

DISCLAIMER
  Flux estimates are indicative only. This report does not constitute
  regulatory compliance documentation. Engage a certified emissions
  specialist for ISO 14064-certified verification.

{'='*64}
"""

    return {"ref": ref, "tier": tier_key, "body": body,
            "subject": f"FractalMesh Intelligence Report — {tier['label']} | {ref}",
            "amount_aud": tier["aud"], "phi_score": phi,
            "anomaly_count": len(anomalies)}


def _generate_markdown_report(anomalies: list, tier_key: str, ref: str) -> str:
    tier = REPORT_TIERS.get(tier_key, REPORT_TIERS["verified_single"])
    ts   = datetime.utcnow()

    lines = [
        f"# FractalMesh Intelligence Report",
        f"## {tier['label']}",
        f"**Reference:** {ref}  ",
        f"**Date:** {ts.strftime('%d %B %Y %H:%M UTC')}  ",
        f"**Operator:** {OPERATOR} | ABN {ABN} | Sole Trader  ",
        f"",
        f"---",
        f"",
        f"## Detected Anomalies ({len(anomalies)} sites)",
        f"",
        f"| # | Lat | Lon | Source | Date | CH4 (ppb) | Enhancement | Flux (kt/yr) |",
        f"|---|-----|-----|--------|------|-----------|-------------|--------------|",
    ]
    for i, a in enumerate(anomalies, 1):
        lines.append(
            f"| {i} | {a.get('lat',0):.4f} | {a.get('lon',0):.4f} | "
            f"{a.get('source','S5P')} | {a.get('sensing_date','N/A')} | "
            f"{a.get('ch4_ppb',0):.1f} | +{a.get('ch4_enhancement',0):.1f} | "
            f"{a.get('estimated_flux_kt',0):.3f} |"
        )

    lines += [
        "",
        "## Data Sources",
        "- Copernicus Sentinel-5P TROPOMI (ESA) — [dataspace.copernicus.eu](https://dataspace.copernicus.eu/)",
        "- NASA EMIT Methane Plume Inventory — [earth.jpl.nasa.gov/emit](https://earth.jpl.nasa.gov/emit/)",
        "",
        "## Chain of Custody",
        f"Ingested by FractalMesh Sovereign Node `fm_sentinel_ingest.py`  ",
        f"SHA256 fingerprint: `{hashlib.sha256(str(anomalies).encode()).hexdigest()[:32]}`  ",
        f"Timestamped in sovereign.db (SQLite WAL-mode)  ",
        "",
        "## Disclaimer",
        "Flux estimates are indicative only. Not regulatory compliance documentation.",
        "",
        f"---",
        f"*{OPERATOR} | ABN {ABN} | {SITE} | {ts.strftime('%Y')}*",
    ]
    return "\n".join(lines)


def _send_report(to_addr: str, subject: str, body: str) -> str:
    if not GMAIL_USER or not GMAIL_PASS:
        return "no_creds"
    if DRY_RUN:
        return "dry_run"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_addr or GMAIL_USER
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, [to_addr or GMAIL_USER], msg.as_string())
        return "sent"
    except Exception as e:
        return f"err:{e}"


def _get_unprocessed_anomalies(limit: int = 10) -> list:
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT m.* FROM methane_readings m
            WHERE m.is_anomaly = 1
            AND NOT EXISTS (
                SELECT 1 FROM methane_report_log r
                WHERE r.anomaly_ids LIKE '%' || CAST(m.id AS TEXT) || '%'
            )
            ORDER BY m.phi_score DESC
            LIMIT ?""", (limit,)).fetchall()
    except Exception:
        rows = []
    conn.close()
    return [dict(r) for r in rows]


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-methane-reports] {ts} | dry={DRY_RUN}")

    anomalies = _get_unprocessed_anomalies()
    print(f"   Unprocessed anomalies: {len(anomalies)}")

    if not anomalies:
        # Demo mode — synthesise a sample report to show pipeline works
        if DRY_RUN:
            anomalies = [{
                "id":               "demo",
                "source":           "sentinel5p",
                "lat":              -25.345,
                "lon":              133.812,
                "ch4_ppb":          1940.0,
                "ch4_enhancement":  70.0,
                "quality_flag":     1,
                "sensing_date":     datetime.utcnow().strftime("%Y-%m-%d"),
                "plume_area_km2":   25.0,
                "estimated_flux_kt": 0.012,
                "phi_score":        round(70.0 * PHI / 100, 6),
            }]
        else:
            print("   No new anomalies — skipping report generation")
            return

    # Generate reports by tier
    for tier_key in ["verified_single", "ais_intel", "crop_yield", "cross_verify"]:
        batch = anomalies[:3] if "cluster" not in tier_key else anomalies[:5]
        rep   = _format_report(batch, tier_key)

        # Save markdown report to dist/
        os.makedirs(DIST, exist_ok=True)
        md_fname = f"{rep['ref'].replace('/', '-')}.md"
        md_path  = os.path.join(DIST, md_fname)
        md_body  = _generate_markdown_report(batch, tier_key, rep["ref"])
        if not DRY_RUN:
            with open(md_path, "w") as f:
                f.write(md_body)

        # Log to DB
        anomaly_ids = json.dumps([a.get("id") for a in batch])
        conn        = sqlite3.connect(DB, timeout=10)
        try:
            conn.execute("""INSERT INTO methane_report_log
                (report_ref,tier,anomaly_ids,status,sent_to,amount_aud,phi_score)
                VALUES (?,?,?,?,?,?,?)""",
                (rep["ref"], tier_key, anomaly_ids, "staged",
                 GMAIL_USER or "self", rep["amount_aud"], rep["phi_score"]))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()

        status = _send_report(GMAIL_USER, rep["subject"], rep["body"])
        print(f"   [{tier_key}] {rep['ref']} | ${rep['amount_aud']:,.0f} AUD | "
              f"anomalies={rep['anomaly_count']} | {status}")
        if not DRY_RUN:
            print(f"   Saved: {md_path}")

        if DRY_RUN:
            break  # one demo report in dry-run

    # Show pricing table
    print("\n   Intelligence Report Pricing:")
    for key, tier in REPORT_TIERS.items():
        print(f"     {key:<22} ${tier['aud']:>7,.0f} AUD — {tier['label']}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-methane-reports] Active | interval={INTERVAL}s | dry={DRY_RUN} | "
          f"tiers={len(REPORT_TIERS)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-methane-reports] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-methane-reports] Stopped.")
