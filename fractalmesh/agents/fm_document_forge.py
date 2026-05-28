"""
FractalMesh OMEGA Titan — Document Forge Agent
Auto-generates professional documents: invoices, proposals, reports, NDAs, SOWs.
Port: 7857  |  Samuel James Hiotis | ABN 56 628 117 363
"""
import base64
import gzip
import hashlib
import hmac
import html
import io
import json
import os
import random
import re
import sqlite3
import string
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault / env loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT             = int(os.environ.get("DOCUMENT_FORGE_PORT", "7857"))
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM    = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET     = os.environ.get("ADMIN_SECRET", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"
_START_TIME = time.time()


def _get_conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id              INTEGER PRIMARY KEY,
            doc_type        TEXT,
            title           TEXT,
            ref_number      TEXT UNIQUE,
            status          TEXT DEFAULT 'draft',
            recipient_email TEXT,
            recipient_name  TEXT,
            sender_name     TEXT,
            content_html    TEXT,
            content_text    TEXT,
            variables       TEXT,
            created_at      REAL,
            sent_at         REAL
        );
        CREATE TABLE IF NOT EXISTS templates (
            id               INTEGER PRIMARY KEY,
            name             TEXT UNIQUE,
            doc_type         TEXT,
            template_html    TEXT,
            variables_schema TEXT,
            created_at       REAL
        );
        CREATE TABLE IF NOT EXISTS sends (
            id          INTEGER PRIMARY KEY,
            document_id INTEGER,
            to_email    TEXT,
            method      TEXT,
            status      TEXT,
            response    TEXT,
            sent_at     REAL
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Reference number
# ---------------------------------------------------------------------------
_PREFIX_MAP = {
    "invoice":  "INV",
    "proposal": "PROP",
    "nda":      "NDA",
    "sow":      "SOW",
    "report":   "RPT",
}


def _make_ref(doc_type: str) -> str:
    prefix = _PREFIX_MAP.get(doc_type, "DOC")
    date_str = time.strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}-{date_str}-{suffix}"


# ---------------------------------------------------------------------------
# Built-in HTML templates
# ---------------------------------------------------------------------------

_INVOICE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Invoice {invoice_number}</title>
<style>
  body{font-family:Arial,sans-serif;color:#222;margin:40px;}
  .header{display:flex;justify-content:space-between;border-bottom:3px solid #1a56db;padding-bottom:16px;margin-bottom:24px;}
  .company h1{font-size:28px;color:#1a56db;margin:0;}
  .company p{margin:2px 0;font-size:13px;}
  .invoice-meta{text-align:right;}
  .invoice-meta h2{font-size:22px;color:#1a56db;margin:0 0 8px;}
  .invoice-meta p{margin:2px 0;font-size:13px;}
  .bill-to{background:#f4f7ff;padding:16px;border-radius:6px;margin-bottom:24px;}
  .bill-to h3{margin:0 0 8px;color:#1a56db;}
  table{width:100%;border-collapse:collapse;margin-bottom:20px;}
  th{background:#1a56db;color:#fff;padding:10px 12px;text-align:left;}
  td{padding:9px 12px;border-bottom:1px solid #e0e7ff;}
  tr:nth-child(even) td{background:#f9faff;}
  .totals{float:right;width:300px;}
  .totals table td{font-size:14px;}
  .totals .grand-total td{font-weight:bold;font-size:16px;color:#1a56db;border-top:2px solid #1a56db;}
  .payment{clear:both;margin-top:32px;background:#f4f7ff;padding:16px;border-radius:6px;}
  .payment h3{margin:0 0 8px;color:#1a56db;}
  .footer{margin-top:40px;text-align:center;font-size:12px;color:#888;border-top:1px solid #e0e7ff;padding-top:12px;}
  .clearfix::after{content:"";display:table;clear:both;}
</style>
</head>
<body>
<div class="header">
  <div class="company">
    <h1>{company_name}</h1>
    <p>ABN: {abn}</p>
    <p>{address}</p>
  </div>
  <div class="invoice-meta">
    <h2>TAX INVOICE</h2>
    <p><strong>Invoice #:</strong> {invoice_number}</p>
    <p><strong>Issue Date:</strong> {issue_date}</p>
    <p><strong>Due Date:</strong> {due_date}</p>
  </div>
</div>
<div class="bill-to">
  <h3>Bill To</h3>
  <p><strong>{client_name}</strong></p>
  <p>{client_email}</p>
</div>
<table>
  <thead><tr><th>Description</th><th>Qty</th><th>Unit Price</th><th>Line Total</th></tr></thead>
  <tbody>{items_rows}</tbody>
</table>
<div class="clearfix">
  <div class="totals">
    <table>
      <tr><td>Subtotal:</td><td style="text-align:right">${subtotal}</td></tr>
      <tr><td>GST (10%):</td><td style="text-align:right">${gst}</td></tr>
      <tr class="grand-total"><td>Total Due:</td><td style="text-align:right">${total}</td></tr>
    </table>
  </div>
</div>
<div class="payment">
  <h3>Payment Instructions</h3>
  <p><strong>Bank:</strong> {bank_name}</p>
  <p><strong>BSB:</strong> {bsb} &nbsp; <strong>Account:</strong> {account_number}</p>
  <p><strong>Account Name:</strong> {account_name}</p>
  <p>Please use Invoice #{invoice_number} as the payment reference.</p>
</div>
<div class="footer">
  <p>{company_name} | ABN {abn} | Thank you for your business.</p>
</div>
</body></html>"""

_INVOICE_SCHEMA = json.dumps({
    "company_name": "string", "abn": "string", "address": "string",
    "invoice_number": "string", "issue_date": "string", "due_date": "string",
    "client_name": "string", "client_email": "string",
    "items": "JSON array of {description, qty, unit_price}",
    "bank_name": "string", "bsb": "string",
    "account_number": "string", "account_name": "string"
})

_PROPOSAL_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Proposal — {project_name}</title>
<style>
  body{font-family:Georgia,serif;color:#1a1a2e;margin:0;padding:0;}
  .cover{background:linear-gradient(135deg,#1a56db,#0f3460);color:#fff;padding:60px 48px;}
  .cover h1{font-size:36px;margin:0 0 8px;}
  .cover .subtitle{font-size:18px;opacity:0.85;}
  .cover .meta{margin-top:32px;font-size:14px;opacity:0.8;}
  .body{padding:40px 48px;max-width:820px;margin:0 auto;}
  h2{color:#1a56db;border-bottom:2px solid #e0e7ff;padding-bottom:6px;margin-top:36px;}
  ul{padding-left:20px;line-height:1.8;}
  table{width:100%;border-collapse:collapse;margin:16px 0;}
  th{background:#1a56db;color:#fff;padding:10px 14px;text-align:left;}
  td{padding:9px 14px;border-bottom:1px solid #e0e7ff;}
  .cta{background:#f4f7ff;border-left:4px solid #1a56db;padding:20px 24px;margin:32px 0;border-radius:0 6px 6px 0;}
  .signature{display:flex;gap:60px;margin-top:48px;}
  .sig-block{flex:1;}
  .sig-line{border-bottom:1px solid #aaa;margin:32px 0 6px;}
  .valid{color:#888;font-size:13px;margin-top:8px;}
  .footer{margin-top:40px;text-align:center;font-size:12px;color:#aaa;border-top:1px solid #e0e7ff;padding:16px 0;}
</style>
</head>
<body>
<div class="cover">
  <h1>{project_name}</h1>
  <div class="subtitle">Service Proposal</div>
  <div class="meta">
    <p>Prepared by: <strong>{company_name}</strong></p>
    <p>Prepared for: <strong>{client_name}</strong></p>
    <p>Date: {proposal_date}</p>
    <p>Valid Until: {valid_until}</p>
  </div>
</div>
<div class="body">
  <h2>Executive Summary</h2>
  <p>We are pleased to present this proposal to <strong>{client_name}</strong> for the
  engagement of <strong>{company_name}</strong> in delivering {project_name}.
  This document outlines the scope of work, deliverables, timeline, and investment
  required to achieve your objectives.</p>

  <h2>Scope of Work</h2>
  <ul>{scope_items_html}</ul>

  <h2>Deliverables</h2>
  <ul>{deliverables_html}</ul>

  <h2>Timeline</h2>
  <p>Estimated project duration: <strong>{timeline_weeks} weeks</strong> from engagement commencement.</p>

  <h2>Investment</h2>
  <table>
    <thead><tr><th>Item</th><th>Amount (AUD)</th></tr></thead>
    <tbody>
      <tr><td>{project_name} — Total Engagement</td><td>${investment_total}</td></tr>
    </tbody>
  </table>
  <p><strong>Payment Terms:</strong> {payment_terms}</p>

  <h2>Terms &amp; Conditions</h2>
  <ul>
    <li>This proposal is valid until {valid_until}.</li>
    <li>Work commences upon receipt of signed agreement and deposit.</li>
    <li>Intellectual property created during the engagement is transferred to the client upon full payment.</li>
    <li>Governing law: New South Wales, Australia.</li>
  </ul>

  <div class="cta">
    <strong>Ready to proceed?</strong><br>
    Sign below and return this document to accept this proposal. We look forward to working with you.
  </div>

  <div class="signature">
    <div class="sig-block">
      <div class="sig-line"></div>
      <p><strong>{company_name}</strong><br>Authorised Representative</p>
    </div>
    <div class="sig-block">
      <div class="sig-line"></div>
      <p><strong>{client_name}</strong><br>Authorised Representative</p>
    </div>
  </div>
  <p class="valid">Proposal Date: {proposal_date}</p>
</div>
<div class="footer">{company_name} — Confidential Proposal</div>
</body></html>"""

_PROPOSAL_SCHEMA = json.dumps({
    "company_name": "string", "client_name": "string", "project_name": "string",
    "proposal_date": "string", "valid_until": "string",
    "scope_items": "JSON array of strings",
    "deliverables": "JSON array of strings",
    "timeline_weeks": "number", "investment_total": "number",
    "payment_terms": "string"
})

_NDA_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Non-Disclosure Agreement</title>
<style>
  body{font-family:"Times New Roman",Times,serif;color:#1a1a1a;margin:60px 80px;line-height:1.7;}
  h1{text-align:center;font-size:22px;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;}
  .subtitle{text-align:center;font-size:14px;color:#555;margin-bottom:32px;}
  h2{font-size:15px;margin-top:28px;text-transform:uppercase;letter-spacing:1px;}
  p{text-align:justify;margin:8px 0;}
  .parties{background:#f9f9f9;border:1px solid #ddd;padding:16px 24px;border-radius:4px;margin:20px 0;}
  .sig-table{width:100%;margin-top:40px;border-collapse:collapse;}
  .sig-table td{width:50%;padding:8px 24px 8px 0;vertical-align:bottom;}
  .sig-line{border-bottom:1px solid #333;margin:40px 0 4px;}
</style>
</head>
<body>
<h1>Mutual Non-Disclosure Agreement</h1>
<div class="subtitle">Effective Date: {effective_date}</div>

<div class="parties">
  <p><strong>Party A:</strong> {party_a}</p>
  <p><strong>Party B:</strong> {party_b}</p>
</div>

<h2>Recitals</h2>
<p>The parties wish to explore a potential business relationship relating to <em>{purpose}</em>
(the "Purpose") and, in connection therewith, each party may disclose to the other certain
confidential and proprietary information. The parties desire to protect such information in
accordance with the terms of this Agreement.</p>

<h2>1. Definition of Confidential Information</h2>
<p>"Confidential Information" means any non-public information disclosed by either party to
the other, either directly or indirectly, in writing, orally, or by inspection of tangible
objects, including without limitation technical data, trade secrets, know-how, research,
product plans, products, services, customers, customer lists, markets, software, developments,
inventions, processes, formulas, technology, designs, drawings, engineering, hardware
configuration information, marketing, finances, or other business information.</p>

<h2>2. Obligations of Receiving Party</h2>
<p>Each receiving party agrees to: (a) hold the disclosing party's Confidential Information
in strict confidence; (b) not disclose it to third parties without prior written consent;
(c) use it solely for the Purpose; and (d) protect it using the same degree of care used to
protect its own confidential information, but in no case less than reasonable care.</p>

<h2>3. Exclusions</h2>
<p>These obligations do not apply to information that: (a) is or becomes publicly known through
no breach of this Agreement; (b) was rightfully in the receiving party's possession prior to
disclosure; (c) is rightfully obtained from a third party without restriction; or
(d) is required to be disclosed by law or court order, provided prompt written notice is given
to the disclosing party.</p>

<h2>4. Term</h2>
<p>This Agreement shall remain in effect for <strong>{term_years} years</strong> from the
Effective Date, unless earlier terminated by mutual written agreement. Obligations with respect
to Confidential Information disclosed during the term shall survive for a further two (2) years
following termination.</p>

<h2>5. Return of Information</h2>
<p>Upon request or termination, each party shall promptly return or destroy all Confidential
Information and any copies thereof in its possession or control and certify in writing that
it has done so.</p>

<h2>6. No Licence</h2>
<p>Nothing in this Agreement grants either party any rights in or to the other party's
Confidential Information except as expressly set forth herein.</p>

<h2>7. Governing Law</h2>
<p>This Agreement shall be governed by and construed in accordance with the laws of
New South Wales, Australia. The parties submit to the exclusive jurisdiction of the courts
of New South Wales.</p>

<h2>8. Entire Agreement</h2>
<p>This Agreement constitutes the entire agreement between the parties with respect to the
subject matter hereof and supersedes all prior agreements, understandings, negotiations,
and discussions, whether oral or written.</p>

<table class="sig-table">
  <tr>
    <td>
      <div class="sig-line"></div>
      <strong>{party_a}</strong><br>
      Signature: ___________________<br>
      Name: ___________________<br>
      Date: ___________________
    </td>
    <td>
      <div class="sig-line"></div>
      <strong>{party_b}</strong><br>
      Signature: ___________________<br>
      Name: ___________________<br>
      Date: ___________________
    </td>
  </tr>
</table>
</body></html>"""

_NDA_SCHEMA = json.dumps({
    "party_a": "string", "party_b": "string",
    "effective_date": "string", "purpose": "string",
    "term_years": "number (default 2)"
})

_SOW_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Statement of Work — {project_name}</title>
<style>
  body{font-family:Arial,sans-serif;color:#1a1a2e;margin:50px 60px;line-height:1.7;}
  .header{border-bottom:3px solid #1a56db;padding-bottom:16px;margin-bottom:28px;}
  .header h1{color:#1a56db;font-size:26px;margin:0 0 4px;}
  .meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 32px;font-size:13px;margin-top:12px;}
  .meta-grid span{color:#555;}
  h2{color:#1a56db;margin-top:32px;border-left:4px solid #1a56db;padding-left:10px;}
  ul{padding-left:20px;line-height:1.9;}
  table{width:100%;border-collapse:collapse;margin:16px 0;}
  th{background:#1a56db;color:#fff;padding:10px 14px;text-align:left;}
  td{padding:9px 14px;border-bottom:1px solid #e0e7ff;}
  tr:nth-child(even) td{background:#f9faff;}
  .clause{background:#fff8e7;border-left:4px solid #f59e0b;padding:12px 18px;margin:20px 0;border-radius:0 6px 6px 0;}
  .sig-row{display:flex;gap:60px;margin-top:48px;}
  .sig-block{flex:1;}
  .sig-line{border-bottom:1px solid #aaa;margin:36px 0 6px;}
  .footer{margin-top:40px;text-align:center;font-size:12px;color:#aaa;border-top:1px solid #e0e7ff;padding:14px 0;}
</style>
</head>
<body>
<div class="header">
  <h1>Statement of Work</h1>
  <div class="meta-grid">
    <div><span>Project:</span> <strong>{project_name}</strong></div>
    <div><span>Client:</span> <strong>{client_name}</strong></div>
    <div><span>Service Provider:</span> <strong>{company_name}</strong></div>
    <div><span>Start Date:</span> {start_date}</div>
    <div><span>End Date:</span> {end_date}</div>
  </div>
</div>

<h2>1. Project Background &amp; Objectives</h2>
<p>{company_name} has been engaged by {client_name} to deliver {project_name}.
This Statement of Work defines the specific deliverables, timeline, and commercial
arrangements governing this engagement.</p>

<h2>2. Scope of Work</h2>
<ul>{scope_items_html}</ul>

<h2>3. Out of Scope</h2>
<p>The following items are explicitly excluded from this engagement unless agreed in writing:</p>
<ul>
  <li>Ongoing maintenance beyond the agreed project end date unless a separate retainer is agreed.</li>
  <li>Third-party licensing costs unless itemised in the payment schedule below.</li>
  <li>Infrastructure hardware procurement.</li>
</ul>

<h2>4. Deliverables &amp; Acceptance Criteria</h2>
<ul>{deliverables_html}</ul>
<p>Each deliverable will be considered accepted if the client provides written approval within
five (5) business days of delivery. Silence beyond this period constitutes acceptance.</p>

<h2>5. Milestones &amp; Payment Schedule</h2>
<table>
  <thead><tr><th>Milestone</th><th>Target Date</th><th>Amount (AUD inc. GST)</th></tr></thead>
  <tbody>{milestones_rows}</tbody>
</table>

<h2>6. Change Management</h2>
<div class="clause">
  Any changes to the scope of work must be agreed in writing via a Change Request (CR) signed
  by both parties. Changes may affect timeline and cost. {company_name} will provide a written
  impact assessment within three (3) business days of receiving a change request.
</div>

<h2>7. Governing Law</h2>
<p>This Statement of Work is governed by the laws of New South Wales, Australia and forms part
of the Master Services Agreement between {company_name} and {client_name}.</p>

<div class="sig-row">
  <div class="sig-block">
    <div class="sig-line"></div>
    <p><strong>{company_name}</strong><br>
    Signature: ___________________<br>Date: ___________________</p>
  </div>
  <div class="sig-block">
    <div class="sig-line"></div>
    <p><strong>{client_name}</strong><br>
    Signature: ___________________<br>Date: ___________________</p>
  </div>
</div>
<div class="footer">{company_name} | Confidential — Statement of Work | {project_name}</div>
</body></html>"""

_SOW_SCHEMA = json.dumps({
    "company_name": "string", "client_name": "string", "project_name": "string",
    "start_date": "string", "end_date": "string",
    "scope_items": "JSON array of strings (optional, defaults to deliverables)",
    "deliverables": "JSON array of strings",
    "milestones": "JSON array of {name, date, amount}"
})

_REPORT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{title}</title>
<style>
  body{font-family:Arial,sans-serif;color:#1a1a2e;margin:0;padding:0;}
  .cover{background:#1a56db;color:#fff;padding:70px 60px;min-height:220px;}
  .cover h1{font-size:34px;margin:0 0 12px;}
  .cover .meta{font-size:15px;opacity:0.85;}
  .toc{background:#f4f7ff;padding:28px 60px;}
  .toc h2{color:#1a56db;font-size:16px;margin-bottom:12px;}
  .toc ol{padding-left:20px;line-height:2;}
  .body{padding:40px 60px;max-width:860px;margin:0 auto;}
  h2.section-heading{color:#1a56db;border-bottom:2px solid #e0e7ff;padding-bottom:6px;margin-top:36px;}
  h3{color:#0f3460;margin-top:24px;}
  p{line-height:1.75;margin:10px 0;}
  .exec-summary{background:#f4f7ff;border-left:4px solid #1a56db;padding:18px 24px;margin:20px 0;border-radius:0 6px 6px 0;}
  .recommendations{background:#fff8e7;border-left:4px solid #f59e0b;padding:18px 24px;margin:28px 0;border-radius:0 6px 6px 0;}
  .appendix{margin-top:40px;border-top:2px solid #e0e7ff;padding-top:20px;}
  .footer{margin-top:40px;text-align:center;font-size:12px;color:#aaa;border-top:1px solid #e0e7ff;padding:14px 0;}
</style>
</head>
<body>
<div class="cover">
  <h1>{title}</h1>
  <div class="meta">
    <p>Author: {author}</p>
    <p>Date: {date}</p>
  </div>
</div>

<div class="toc">
  <h2>Table of Contents</h2>
  <ol>{toc_items}</ol>
</div>

<div class="body">
  <h2 class="section-heading">Executive Summary</h2>
  <div class="exec-summary">
    <p>This report, <em>{title}</em>, was prepared by {author} on {date}.
    It presents key findings, analysis, and recommendations based on available data and research.</p>
  </div>

  {sections_html}

  <div class="recommendations">
    <h2>Recommendations</h2>
    <p>Based on the findings presented in this report, the following actions are recommended.
    Each section above contains detailed analysis; stakeholders should review the relevant
    sections and implement recommendations in priority order.</p>
  </div>

  <div class="appendix">
    <h2>Appendix</h2>
    <p>Additional data, source documents, and supporting materials are available upon request.
    All figures are current as of the report date: {date}.</p>
  </div>
</div>
<div class="footer">{title} | Prepared by {author} | {date} | Confidential</div>
</body></html>"""

_REPORT_SCHEMA = json.dumps({
    "title": "string", "author": "string", "date": "string",
    "sections": "JSON array of {heading, body}"
})

_BUILTIN_TEMPLATES = [
    {
        "name": "invoice",
        "doc_type": "invoice",
        "template_html": _INVOICE_HTML,
        "variables_schema": _INVOICE_SCHEMA,
    },
    {
        "name": "proposal",
        "doc_type": "proposal",
        "template_html": _PROPOSAL_HTML,
        "variables_schema": _PROPOSAL_SCHEMA,
    },
    {
        "name": "nda",
        "doc_type": "nda",
        "template_html": _NDA_HTML,
        "variables_schema": _NDA_SCHEMA,
    },
    {
        "name": "sow",
        "doc_type": "sow",
        "template_html": _SOW_HTML,
        "variables_schema": _SOW_SCHEMA,
    },
    {
        "name": "report",
        "doc_type": "report",
        "template_html": _REPORT_HTML,
        "variables_schema": _REPORT_SCHEMA,
    },
]


def _seed_templates():
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
    if count == 0:
        now = time.time()
        for t in _BUILTIN_TEMPLATES:
            conn.execute(
                "INSERT OR IGNORE INTO templates (name, doc_type, template_html, variables_schema, created_at) "
                "VALUES (?,?,?,?,?)",
                (t["name"], t["doc_type"], t["template_html"], t["variables_schema"], now)
            )
        conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Rendering engine
# ---------------------------------------------------------------------------

def _render_invoice(tmpl: str, variables: dict) -> str:
    items_raw = variables.get("items", "[]")
    if isinstance(items_raw, str):
        try:
            items = json.loads(items_raw)
        except Exception:
            items = []
    else:
        items = items_raw

    rows = ""
    subtotal = 0.0
    for item in items:
        desc = html.escape(str(item.get("description", "")))
        qty = float(item.get("qty", 1))
        unit = float(item.get("unit_price", 0))
        line = qty * unit
        subtotal += line
        rows += (
            f"<tr><td>{desc}</td><td>{qty:.0f}</td>"
            f"<td>${unit:,.2f}</td><td>${line:,.2f}</td></tr>\n"
        )

    gst = subtotal * 0.10
    total = subtotal + gst

    rendered = tmpl
    rendered = rendered.replace("{items_rows}", rows)
    rendered = rendered.replace("{subtotal}", f"{subtotal:,.2f}")
    rendered = rendered.replace("{gst}", f"{gst:,.2f}")
    rendered = rendered.replace("{total}", f"{total:,.2f}")
    for k, v in variables.items():
        if k != "items":
            rendered = rendered.replace("{" + k + "}", html.escape(str(v)))
    return rendered


def _render_proposal(tmpl: str, variables: dict) -> str:
    scope_raw = variables.get("scope_items", "[]")
    if isinstance(scope_raw, str):
        try:
            scope = json.loads(scope_raw)
        except Exception:
            scope = []
    else:
        scope = scope_raw

    deliv_raw = variables.get("deliverables", "[]")
    if isinstance(deliv_raw, str):
        try:
            deliverables = json.loads(deliv_raw)
        except Exception:
            deliverables = []
    else:
        deliverables = deliv_raw

    scope_html = "".join(f"<li>{html.escape(str(s))}</li>" for s in scope)
    deliv_html = "".join(f"<li>{html.escape(str(d))}</li>" for d in deliverables)

    rendered = tmpl
    rendered = rendered.replace("{scope_items_html}", scope_html)
    rendered = rendered.replace("{deliverables_html}", deliv_html)
    for k, v in variables.items():
        if k not in ("scope_items", "deliverables"):
            rendered = rendered.replace("{" + k + "}", html.escape(str(v)))
    return rendered


def _render_sow(tmpl: str, variables: dict) -> str:
    scope_raw = variables.get("scope_items", "[]")
    if isinstance(scope_raw, str):
        try:
            scope = json.loads(scope_raw)
        except Exception:
            scope = []
    else:
        scope = scope_raw

    deliv_raw = variables.get("deliverables", "[]")
    if isinstance(deliv_raw, str):
        try:
            deliverables = json.loads(deliv_raw)
        except Exception:
            deliverables = []
    else:
        deliverables = deliv_raw

    milestones_raw = variables.get("milestones", "[]")
    if isinstance(milestones_raw, str):
        try:
            milestones = json.loads(milestones_raw)
        except Exception:
            milestones = []
    else:
        milestones = milestones_raw

    scope_html = "".join(f"<li>{html.escape(str(s))}</li>" for s in scope)
    deliv_html = "".join(f"<li>{html.escape(str(d))}</li>" for d in deliverables)
    ms_rows = ""
    for m in milestones:
        name = html.escape(str(m.get("name", "")))
        date = html.escape(str(m.get("date", "")))
        amount = float(m.get("amount", 0))
        ms_rows += f"<tr><td>{name}</td><td>{date}</td><td>${amount:,.2f}</td></tr>\n"

    rendered = tmpl
    rendered = rendered.replace("{scope_items_html}", scope_html)
    rendered = rendered.replace("{deliverables_html}", deliv_html)
    rendered = rendered.replace("{milestones_rows}", ms_rows)
    for k, v in variables.items():
        if k not in ("scope_items", "deliverables", "milestones"):
            rendered = rendered.replace("{" + k + "}", html.escape(str(v)))
    return rendered


def _render_report(tmpl: str, variables: dict) -> str:
    sections_raw = variables.get("sections", "[]")
    if isinstance(sections_raw, str):
        try:
            sections = json.loads(sections_raw)
        except Exception:
            sections = []
    else:
        sections = sections_raw

    toc_items = "".join(f"<li>{html.escape(str(s.get('heading', '')))}</li>" for s in sections)
    sections_html = ""
    for s in sections:
        heading = html.escape(str(s.get("heading", "")))
        body = html.escape(str(s.get("body", "")))
        sections_html += f"<h2 class=\"section-heading\">{heading}</h2>\n<p>{body}</p>\n"

    rendered = tmpl
    rendered = rendered.replace("{toc_items}", toc_items)
    rendered = rendered.replace("{sections_html}", sections_html)
    for k, v in variables.items():
        if k != "sections":
            rendered = rendered.replace("{" + k + "}", html.escape(str(v)))
    return rendered


def _render_nda(tmpl: str, variables: dict) -> str:
    rendered = tmpl
    for k, v in variables.items():
        rendered = rendered.replace("{" + k + "}", html.escape(str(v)))
    return rendered


def _render_template(doc_type: str, tmpl_html: str, variables: dict) -> str:
    if doc_type == "invoice":
        return _render_invoice(tmpl_html, variables)
    elif doc_type == "proposal":
        return _render_proposal(tmpl_html, variables)
    elif doc_type == "sow":
        return _render_sow(tmpl_html, variables)
    elif doc_type == "report":
        return _render_report(tmpl_html, variables)
    else:
        # NDA and generic: simple replacement
        return _render_nda(tmpl_html, variables)


def _html_to_text(content_html: str) -> str:
    """Strip HTML tags and decode entities to produce plain text."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", content_html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|h[1-6]|li|tr|section|article)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _make_pdf_like(content_html: str, title: str) -> str:
    """Create a base64-encoded gzip-compressed structured representation."""
    text = _html_to_text(content_html)
    envelope = {
        "format": "FractalMesh-Doc-v1",
        "title": title,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "content_text": text,
        "content_html_hash": hashlib.sha256(content_html.encode()).hexdigest(),
        "pages_approx": max(1, len(text) // 3000),
    }
    raw = json.dumps(envelope, indent=2).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(raw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# SendGrid helper
# ---------------------------------------------------------------------------

def _send_via_sendgrid(to_email: str, subject: str, html_body: str) -> dict:
    import urllib.request
    if not SENDGRID_API_KEY:
        return {"ok": False, "error": "SENDGRID_API_KEY not configured"}
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}]
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"ok": True, "status_code": resp.status}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Background weekly summary thread
# ---------------------------------------------------------------------------

def _generate_weekly_summary():
    try:
        conn = _get_conn()
        since = time.time() - 7 * 86400
        rows = conn.execute(
            "SELECT doc_type, status, COUNT(*) as cnt FROM documents "
            "WHERE created_at >= ? GROUP BY doc_type, status",
            (since,)
        ).fetchall()
        conn.close()

        summary_by_type: dict = {}
        summary_by_status: dict = {}
        for row in rows:
            dt = row["doc_type"] or "unknown"
            st = row["status"] or "unknown"
            cnt = row["cnt"]
            summary_by_type[dt] = summary_by_type.get(dt, 0) + cnt
            summary_by_status[st] = summary_by_status.get(st, 0) + cnt

        type_rows = "".join(
            f"<tr><td>{html.escape(dt)}</td><td>{cnt}</td></tr>"
            for dt, cnt in summary_by_type.items()
        )
        status_rows = "".join(
            f"<tr><td>{html.escape(st)}</td><td>{cnt}</td></tr>"
            for st, cnt in summary_by_status.items()
        )
        total = sum(summary_by_type.values())

        report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Weekly Document Summary</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;}}
h1{{color:#1a56db;}} table{{border-collapse:collapse;width:400px;}}
th{{background:#1a56db;color:#fff;padding:8px 14px;text-align:left;}}
td{{padding:8px 14px;border-bottom:1px solid #e0e7ff;}}</style></head>
<body>
<h1>Weekly Document Summary</h1>
<p>Period: Last 7 days | Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}</p>
<p><strong>Total Documents:</strong> {total}</p>
<h2>By Document Type</h2>
<table><thead><tr><th>Type</th><th>Count</th></tr></thead><tbody>{type_rows}</tbody></table>
<h2>By Status</h2>
<table><thead><tr><th>Status</th><th>Count</th></tr></thead><tbody>{status_rows}</tbody></table>
</body></html>"""

        content_text = _html_to_text(report_html)
        ref = _make_ref("report")
        now = time.time()
        conn2 = _get_conn()
        conn2.execute(
            "INSERT INTO documents (doc_type, title, ref_number, status, sender_name, "
            "content_html, content_text, variables, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            ("report", "Weekly Document Summary", ref, "draft", "System",
             report_html, content_text,
             json.dumps({"period": "7d", "total": total}), now)
        )
        conn2.commit()
        conn2.close()
    except Exception as exc:
        print(f"[DocumentForge] Weekly summary error: {exc}")


def _background_loop():
    while True:
        time.sleep(3600)
        _generate_weekly_summary()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_response(handler, data, status=200):
    body = json.dumps(data, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, msg, status=400):
    _json_response(handler, {"error": msg}, status)


def _require_admin(handler) -> bool:
    auth = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or not hmac.compare_digest(auth, ADMIN_SECRET):
        _error(handler, "Unauthorized — X-Admin-Secret required", 401)
        return False
    return True


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_id(path: str, prefix: str) -> int | None:
    tail = path[len(prefix):]
    id_part = tail.split("/")[0]
    try:
        return int(id_part)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class DocumentForgeHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    # ------------------------------------------------------------------
    # GET dispatcher
    # ------------------------------------------------------------------

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        qs = self.path[len(p)+1:] if "?" in self.path else ""

        if p == "/health":
            self._handle_health()
        elif p == "/documents":
            self._handle_list_documents(qs)
        elif re.match(r"^/documents/\d+/html$", p):
            doc_id = int(re.search(r"/documents/(\d+)/html", p).group(1))
            self._handle_doc_html(doc_id)
        elif re.match(r"^/documents/\d+/text$", p):
            doc_id = int(re.search(r"/documents/(\d+)/text", p).group(1))
            self._handle_doc_text(doc_id)
        elif re.match(r"^/documents/\d+$", p):
            doc_id = int(re.search(r"/documents/(\d+)", p).group(1))
            self._handle_get_document(doc_id)
        elif p == "/templates":
            self._handle_list_templates()
        elif re.match(r"^/templates/[^/]+$", p):
            name = p[len("/templates/"):]
            self._handle_get_template(name)
        else:
            _error(self, "Not found", 404)

    # ------------------------------------------------------------------
    # POST dispatcher
    # ------------------------------------------------------------------

    def do_POST(self):
        p = self.path.split("?")[0].rstrip("/")

        if p == "/generate":
            self._handle_generate()
        elif re.match(r"^/documents/\d+/send$", p):
            doc_id = int(re.search(r"/documents/(\d+)/send", p).group(1))
            self._handle_send(doc_id)
        elif p == "/templates":
            self._handle_create_template()
        else:
            _error(self, "Not found", 404)

    # ------------------------------------------------------------------
    # PUT dispatcher
    # ------------------------------------------------------------------

    def do_PUT(self):
        p = self.path.split("?")[0].rstrip("/")
        if re.match(r"^/documents/\d+$", p):
            doc_id = int(re.search(r"/documents/(\d+)", p).group(1))
            self._handle_update_document(doc_id)
        else:
            _error(self, "Not found", 404)

    # ------------------------------------------------------------------
    # DELETE dispatcher
    # ------------------------------------------------------------------

    def do_DELETE(self):
        p = self.path.split("?")[0].rstrip("/")
        if re.match(r"^/documents/\d+$", p):
            doc_id = int(re.search(r"/documents/(\d+)", p).group(1))
            self._handle_delete_document(doc_id)
        else:
            _error(self, "Not found", 404)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_health(self):
        conn = _get_conn()
        rows = conn.execute(
            "SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type"
        ).fetchall()
        conn.close()
        counts = {row["doc_type"]: row["cnt"] for row in rows}
        _json_response(self, {
            "service": "DocumentForge",
            "status": "healthy",
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "port": PORT,
            "doc_counts_by_type": counts,
        })

    def _handle_list_documents(self, qs: str):
        params: dict = {}
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
        doc_type = params.get("doc_type")
        status = params.get("status")
        limit = min(int(params.get("limit", "50")), 500)

        conditions = []
        args: list = []
        if doc_type:
            conditions.append("doc_type = ?")
            args.append(doc_type)
        if status:
            conditions.append("status = ?")
            args.append(status)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        args.append(limit)

        conn = _get_conn()
        rows = conn.execute(
            f"SELECT id, doc_type, title, ref_number, status, recipient_email, "
            f"recipient_name, sender_name, created_at, sent_at "
            f"FROM documents {where} ORDER BY created_at DESC LIMIT ?",
            args
        ).fetchall()
        conn.close()
        _json_response(self, {"documents": [dict(r) for r in rows]})

    def _handle_get_document(self, doc_id: int):
        conn = _get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if not row:
            _error(self, f"Document {doc_id} not found", 404)
            return
        data = dict(row)
        # include base64 PDF-like blob
        data["pdf_base64"] = _make_pdf_like(data.get("content_html", ""), data.get("title", ""))
        _json_response(self, data)

    def _handle_doc_html(self, doc_id: int):
        conn = _get_conn()
        row = conn.execute("SELECT content_html FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if not row:
            _error(self, f"Document {doc_id} not found", 404)
            return
        body = (row["content_html"] or "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_doc_text(self, doc_id: int):
        conn = _get_conn()
        row = conn.execute("SELECT content_text FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if not row:
            _error(self, f"Document {doc_id} not found", 404)
            return
        body = (row["content_text"] or "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_list_templates(self):
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, name, doc_type, variables_schema, created_at FROM templates ORDER BY id"
        ).fetchall()
        conn.close()
        _json_response(self, {"templates": [dict(r) for r in rows]})

    def _handle_get_template(self, name: str):
        conn = _get_conn()
        row = conn.execute("SELECT * FROM templates WHERE name = ?", (name,)).fetchone()
        conn.close()
        if not row:
            _error(self, f"Template '{name}' not found", 404)
            return
        _json_response(self, dict(row))

    def _handle_generate(self):
        body = _read_body(self)
        doc_type = body.get("doc_type", "").strip().lower()
        if not doc_type:
            _error(self, "doc_type is required")
            return
        recipient_name  = body.get("recipient_name", "")
        recipient_email = body.get("recipient_email", "")
        sender_name     = body.get("sender_name", "FractalMesh")
        variables       = body.get("variables", {})
        if not isinstance(variables, dict):
            _error(self, "variables must be a JSON object")
            return

        conn = _get_conn()
        tmpl_row = conn.execute(
            "SELECT * FROM templates WHERE doc_type = ? OR name = ? LIMIT 1",
            (doc_type, doc_type)
        ).fetchone()
        conn.close()

        if not tmpl_row:
            _error(self, f"No template found for doc_type '{doc_type}'", 404)
            return

        tmpl_html = tmpl_row["template_html"]
        try:
            content_html = _render_template(doc_type, tmpl_html, variables)
        except Exception as exc:
            _error(self, f"Render error: {exc}", 500)
            return

        content_text = _html_to_text(content_html)
        ref_number = _make_ref(doc_type)
        title_var = variables.get("title") or variables.get("project_name") or \
                    variables.get("invoice_number") or f"{doc_type.upper()} {ref_number}"
        title = str(title_var)
        now = time.time()

        conn2 = _get_conn()
        cur = conn2.execute(
            "INSERT INTO documents (doc_type, title, ref_number, status, "
            "recipient_email, recipient_name, sender_name, content_html, content_text, "
            "variables, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (doc_type, title, ref_number, "draft",
             recipient_email, recipient_name, sender_name,
             content_html, content_text, json.dumps(variables), now)
        )
        doc_id = cur.lastrowid
        conn2.commit()
        conn2.close()

        preview = content_text[:500] if content_text else ""
        _json_response(self, {
            "id": doc_id,
            "ref_number": ref_number,
            "doc_type": doc_type,
            "title": title,
            "status": "draft",
            "preview": preview,
        }, 201)

    def _handle_send(self, doc_id: int):
        if not _require_admin(self):
            return
        body = _read_body(self)
        to_email = body.get("to_email", "").strip()
        if not to_email:
            _error(self, "to_email is required")
            return

        conn = _get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        if not row:
            _error(self, f"Document {doc_id} not found", 404)
            return

        subject = f"[FractalMesh] {row['title']} — {row['ref_number']}"
        result = _send_via_sendgrid(to_email, subject, row["content_html"] or "")
        now = time.time()

        conn2 = _get_conn()
        conn2.execute(
            "INSERT INTO sends (document_id, to_email, method, status, response, sent_at) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, to_email, "sendgrid",
             "sent" if result.get("ok") else "failed",
             json.dumps(result), now)
        )
        if result.get("ok"):
            conn2.execute(
                "UPDATE documents SET status='sent', sent_at=? WHERE id=?",
                (now, doc_id)
            )
        conn2.commit()
        conn2.close()

        _json_response(self, {"document_id": doc_id, "to_email": to_email, "result": result})

    def _handle_create_template(self):
        if not _require_admin(self):
            return
        body = _read_body(self)
        name = body.get("name", "").strip()
        doc_type = body.get("doc_type", "").strip().lower()
        tmpl_html = body.get("template_html", "")
        variables_schema = body.get("variables_schema", "{}")

        if not name or not doc_type or not tmpl_html:
            _error(self, "name, doc_type, and template_html are required")
            return

        now = time.time()
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO templates (name, doc_type, template_html, variables_schema, created_at) "
                "VALUES (?,?,?,?,?)",
                (name, doc_type, tmpl_html,
                 variables_schema if isinstance(variables_schema, str) else json.dumps(variables_schema),
                 now)
            )
            conn.commit()
            row = conn.execute("SELECT * FROM templates WHERE name=?", (name,)).fetchone()
            conn.close()
            _json_response(self, dict(row), 201)
        except sqlite3.IntegrityError:
            conn.close()
            _error(self, f"Template '{name}' already exists", 409)

    def _handle_update_document(self, doc_id: int):
        if not _require_admin(self):
            return
        body = _read_body(self)
        conn = _get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            conn.close()
            _error(self, f"Document {doc_id} not found", 404)
            return

        updates = []
        args = []
        if "status" in body:
            updates.append("status=?")
            args.append(str(body["status"]))
        if "variables" in body:
            updates.append("variables=?")
            v = body["variables"]
            args.append(json.dumps(v) if isinstance(v, dict) else str(v))
        if not updates:
            conn.close()
            _error(self, "Nothing to update — provide status and/or variables")
            return

        args.append(doc_id)
        conn.execute(f"UPDATE documents SET {', '.join(updates)} WHERE id=?", args)
        conn.commit()
        updated = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        conn.close()
        _json_response(self, dict(updated))

    def _handle_delete_document(self, doc_id: int):
        if not _require_admin(self):
            return
        conn = _get_conn()
        row = conn.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            conn.close()
            _error(self, f"Document {doc_id} not found", 404)
            return
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.execute("DELETE FROM sends WHERE document_id=?", (doc_id,))
        conn.commit()
        conn.close()
        _json_response(self, {"deleted": doc_id})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _init_db()
    _seed_templates()

    # Generate an initial weekly summary at startup
    t_summary = threading.Thread(target=_generate_weekly_summary, daemon=True)
    t_summary.start()

    # Background loop for hourly runs
    t_bg = threading.Thread(target=_background_loop, daemon=True)
    t_bg.start()

    server = HTTPServer(("0.0.0.0", PORT), DocumentForgeHandler)
    print(f"[DocumentForge] Listening on port {PORT} — DB: {_DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[DocumentForge] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
