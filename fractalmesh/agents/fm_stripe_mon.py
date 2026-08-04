#!/usr/bin/env python3
"""
fm_stripe_mon.py — Stripe webhook receiver + Gmail blueprint delivery
Listens on port 8091 for Stripe checkout.session.completed events,
then emails the AI Node Blueprint PDF to the customer.

Samuel James Hiotis | ABN 56628117363
Vault keys: GMAIL_USER, GMAIL_APP_PASS, STRIPE_WEBHOOK_SECRET
"""
import os, time, logging, smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from flask import Flask, request, jsonify

# ── Load vault ───────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STRIPE-MON] %(message)s",
)
log = logging.getLogger("stripe_mon")

app = Flask("fm_stripe_mon")

GMAIL_USER    = os.getenv("GMAIL_USER", "")
GMAIL_PASS    = os.getenv("GMAIL_APP_PASS", "")
BLUEPRINT_PATH = Path(os.getenv(
    "FRACTALMESH_HOME",
    os.path.expanduser("~/fmsaas")
)) / "dist" / "ai_node_blueprint.md"
OPERATOR_NAME = os.getenv("NAME", "Samuel James Hiotis")
OPERATOR_PHONE = os.getenv("PHONE", "")


def send_blueprint(customer_email: str, customer_name: str = "Valued Customer") -> bool:
    """Send AI Node Blueprint to customer via Gmail SMTP."""
    if not GMAIL_USER or not GMAIL_PASS:
        log.warning("GMAIL credentials not set — skipping delivery")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = "⚡ AI Node Blueprint: Deployment Initiated"
    msg["From"]    = GMAIL_USER
    msg["To"]      = customer_email

    body = (
        f"Dear {customer_name},\n\n"
        "Your AI Node Blueprint has been attached to this email.\n\n"
        "This document covers the complete sovereign architecture for\n"
        "autonomous AI agents on local hardware.\n\n"
        f"Support: {OPERATOR_PHONE}\n"
        f"— {OPERATOR_NAME}"
    )
    msg.attach(MIMEText(body, "plain"))

    if BLUEPRINT_PATH.exists():
        with open(BLUEPRINT_PATH, "rb") as f:
            attachment = MIMEApplication(f.read(), Name="ai_node_blueprint.md")
            attachment["Content-Disposition"] = (
                'attachment; filename="ai_node_blueprint.md"'
            )
            msg.attach(attachment)
    else:
        log.warning("Blueprint file not found at %s", BLUEPRINT_PATH)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.send_message(msg)
        log.info("Blueprint delivered to %s", customer_email)
        return True
    except Exception as exc:
        log.error("Gmail delivery failed: %s", exc)
        return False


@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    """Stripe webhook endpoint — handles checkout.session.completed."""
    payload = request.get_data()

    # Verify signature if secret is configured
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    if webhook_secret:
        sig_header = request.headers.get("Stripe-Signature", "")
        try:
            import hmac, hashlib
            ts_part  = [p for p in sig_header.split(",") if p.startswith("t=")]
            sig_part = [p for p in sig_header.split(",") if p.startswith("v1=")]
            if ts_part and sig_part:
                ts        = ts_part[0][2:]
                expected  = hmac.new(
                    webhook_secret.encode(),
                    f"{ts}.".encode() + payload,
                    hashlib.sha256
                ).hexdigest()
                received  = sig_part[0][3:]
                if not hmac.compare_digest(expected, received):
                    log.warning("Invalid Stripe signature")
                    return jsonify(error="invalid_signature"), 400
        except Exception as exc:
            log.warning("Signature check failed: %s", exc)

    try:
        import json
        event = json.loads(payload)
    except Exception:
        return jsonify(error="invalid_json"), 400

    if event.get("type") == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        customer_email = session.get("customer_details", {}).get("email", "")
        customer_name  = session.get("customer_details", {}).get("name", "Valued Customer")
        if customer_email:
            log.info("Checkout completed for %s — sending blueprint", customer_email)
            send_blueprint(customer_email, customer_name)

    return jsonify(success=True), 200


@app.route("/health")
def health():
    from datetime import datetime, timezone
    return jsonify({
        "ok":      True,
        "service": "fm-stripe-mon",
        "ts":      datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    port = int(os.getenv("STRIPE_MON_PORT", "8091"))
    log.info("fm-stripe-mon online on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
