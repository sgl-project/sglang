# FractalMesh Omega Titan v2.0.0

**Sovereign AI Mesh Infrastructure for Sole Traders, Operators & DePIN Nodes**

> Built, operated and maintained by **Samuel James Hiotis**  
> ABN: 56 628 117 363 | Albury NSW 2640, Australia  
> Contact: [fractalmesh.net/contact.html](https://fractalmesh.net/contact.html)  
> Products: [fractalmesh.net/products.html](https://fractalmesh.net/products.html)

---

## What Is FractalMesh?

FractalMesh is a self-hosted, sovereign AI agent mesh designed for sole traders and small operators who need enterprise-grade automation without enterprise overhead. Every agent runs on your own hardware — Termux (Android), Raspberry Pi, or any Linux server — under your own credentials, with zero vendor lock-in.

The system handles lead discovery, automated outreach, proposal generation, compliance documentation, revenue tracking, crypto node management, watermarking, licensing, geo-validation, and much more — all via a φ-harmonic (golden ratio) scoring engine that prioritises the highest-value tasks at every cycle.

---

## Quick Start

### 1. Clone & Configure Vault

```bash
git clone https://github.com/samhiotisiddn-jpg/sglang ~/sglang
mkdir -p ~/.secrets
cp ~/.secrets/fractal.env.example ~/.secrets/fractal.env  # or run deploy script
chmod 600 ~/.secrets/fractal.env
nano ~/.secrets/fractal.env   # fill in your rotated credentials
```

### 2. Deploy (Linux / Termux)

```bash
bash ~/sglang/fractalmesh/scripts/deploy_omega_v40.sh
```

This script:
- Detects Termux vs Debian/Linux
- Installs Python deps (including Rust for ormsgpack on ARM)
- Generates `sovereign.db` with all required tables
- Writes `ecosystem.config.js` with 30+ agents
- Starts the core 6 agents first: pulse-bus, dashboard, health-api, sovereign-ops, healer, salvage-crew

### 3. Start Full Swarm

```bash
cd ~/fmsaas
pm2 start ecosystem.config.js --env production
pm2 save
```

### 4. Access Omni-Dashboard

```
http://127.0.0.1:8090
```

Hot-upgrade, agent control, log viewer, vault key presence, DB stats — all in-browser.

### 5. Individual API Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Omni-Dashboard | http://127.0.0.1:8090 | Full control panel |
| Health API | http://127.0.0.1:5057/health | Uptime check |
| Pulse Bus | http://127.0.0.1:5060/health | Event bus status |
| RAG API | http://127.0.0.1:8001/docs | Semantic search |
| Billing API | http://127.0.0.1:8003/docs | Revenue metering |

---

## Vault Configuration (`~/.secrets/fractal.env`)

**Never commit this file. Never paste values into chat.**

```bash
# ── Core identity ─────────────────────────────────────────
OPERATOR_NAME="Samuel James Hiotis"
OPERATOR_EMAIL="your@email.com"
OPERATOR_PHONE="0439 008 640"
ABN="56628117363"

# ── Internal security ─────────────────────────────────────
BUS_SECRET="generate-with: python3 -c 'import secrets; print(secrets.token_hex(32))'"
DASHBOARD_TOKEN="generate-with: python3 -c 'import secrets; print(secrets.token_hex(24))'"

# ── Email (Gmail App Password — not your main password) ───
GMAIL_USER="your.gmail@gmail.com"
GMAIL_APP_PASS="xxxx xxxx xxxx xxxx"

# ── Payments ──────────────────────────────────────────────
STRIPE_SECRET_KEY="sk_live_..."
STRIPE_PUBLISHABLE_KEY="pk_live_..."
STRIPE_WEBHOOK_SECRET="whsec_..."
STRIPE_PAYMENT_LINK="https://buy.stripe.com/..."
PAYPAL_CLIENT_ID=""
PAYPAL_CLIENT_SECRET=""
LEMONSQUEEZY_API_KEY=""

# ── Trading (rotate all keys periodically) ────────────────
BLOFIN_API_KEY=""
BLOFIN_API_SECRET=""
KUCOIN_API_KEY=""
KUCOIN_API_SECRET=""
KUCOIN_API_PASSPHRASE=""
PIONEX_API_KEY=""
PIONEX_API_SECRET=""

# ── AI / LLM ──────────────────────────────────────────────
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
XAI_API_KEY=""
OLLAMA_HOST="http://localhost:11434"

# ── Infra ─────────────────────────────────────────────────
CF_TUNNEL_TOKEN=""
SUPABASE_URL=""
SUPABASE_ANON_KEY=""
PINATA_JWT=""

# ── OSINT ─────────────────────────────────────────────────
GOOGLE_CSE_API_KEY=""
GOOGLE_CSE_ID=""
WIGLE_API_NAME=""
WIGLE_API_TOKEN=""

# ── AdMob ─────────────────────────────────────────────────
ADMOB_PUBLISHER_ID=""
ADMOB_API_KEY=""

# ── Feature flags (set "true" to enable live mode) ────────
ENABLE_AFFILIATE_LIVE="false"
ENABLE_AUTO_ADVERT="false"
ENABLE_CONTRACT_FORGE="false"
ENABLE_DORK_ENGINE="false"
ENABLE_OSINT_SPIDER="false"
ENABLE_NEGOTIATOR="false"
ENABLE_HEALER_RESTARTS="false"
ENABLE_WIGLE_QUERY="false"
ENABLE_STRIPE_GATEWAY="false"
ENABLE_SMART_CONTRACTS="false"
ENABLE_ADMOB="false"
ENABLE_WATERMARK="false"
ENABLE_GEO_VALIDATOR="false"
ENABLE_CARBON_CREDITS="false"
ENABLE_CAMPAIGNS="false"
```

---

## Agent Catalogue

### Core Infrastructure

| Agent | File | Port | Purpose |
|-------|------|------|---------|
| fm-pulse-bus | `agents/fm_pulse_bus.py` | 5060 | HMAC-signed event bus |
| fm-health-api | `api/health_app.py` | 5057 | Uptime + health endpoint |
| omni-dashboard | `api/omni_dashboard.py` | 8090 | Full control UI |
| fm-enterprise-bus | `agents/fm_enterprise_bus.py` | — | Circuit-breaker message bus |
| fm-sovereign-ops | `agents/fm_sovereign_ops.py` | — | HMAC heartbeat + compliance |
| fm-oversight | `agents/fm_oversight.py` | — | PM2 + DB audit |

### Self-Healing & Persistence

| Agent | File | Purpose |
|-------|------|---------|
| fm-healer | `agents/fm_healer.py` | Restarts crashed PM2 nodes |
| fm-immortality | `agents/fm_immortality.py` | SQLite WAL snapshots + SHA256 verify |
| fm-salvage-crew | `agents/fm_salvage_crew.py` | Rebuilds missing DB tables |

### Revenue & Payments

| Agent | File | Purpose |
|-------|------|---------|
| fm-stripe-gateway | `agents/fm_stripe_gateway.py` | Stripe balance + charge polling |
| fm-stripe-mon | `agents/fm_stripe_mon.py` | Stripe webhook monitor |
| fm-live-tokenomics | `agents/fm_live_tokenomics.py` | φ-weighted royalty pool management |
| fm-micro-charge | `agents/economics/fm_micro_charge.py` | Micro-billing automation |
| billing-api | `api/billing_api.py` | REST billing/metering API |

### Intelligence & Lead Generation

| Agent | File | Purpose |
|-------|------|---------|
| fm-osint-spider | `agents/fm_osint_spider.py` | 18-dork Google CSE lead discovery |
| fm-dork-engine | `agents/fm_dork_engine.py` | Targeted Google dorking |
| fm-wigle-oracle | `agents/fm_wigle_oracle.py` | WiFi topology around Albury NSW |
| rag-api | `api/rag_api.py` | Semantic knowledge search |

### Outreach & Negotiation

| Agent | File | Purpose |
|-------|------|---------|
| fm-negotiator | `agents/fm_negotiator.py` | 4A proposal generation + delivery |
| fm-auto-advert | `agents/fm_auto_advert.py` | Automated Gmail ad campaigns |
| fm-affiliate | `agents/fm_affiliate.py` | Affiliate link health + tracking |
| fm-campaign-manager | `agents/marketing/fm_campaign_manager.py` | Multi-channel campaign lifecycle |

### Compliance & Legal Automation

| Agent | File | Purpose |
|-------|------|---------|
| fm-licensing | `agents/fm_licensing.py` | Auto copyright/license file stamping |
| fm-watermark | `agents/fm_watermark.py` | Document + image watermarking |
| fm-smart-contracts | `agents/fm_smart_contracts.py` | EVM contract generation + tracking |
| fm-lba-bridge | `agents/fm_lba_bridge.py` | ATO/LBA compliance heartbeat |
| fm-contract-forge | `agents/fm_contract_forge.py` | MSA contract document generation |

### Geo, Environment & Telemetry

| Agent | File | Purpose |
|-------|------|---------|
| fm-geo-validator | `agents/fm_geo_validator.py` | Coordinate validation + geotagging |
| fm-carbon-credits | `agents/fm_carbon_credits.py` | Carbon credit analysis + reporting |
| fm-device-bridge | `agents/fm_device_bridge.py` | ADB/SSH Android + RPi health |
| fm-samsung-warden | `agents/fm_samsung_warden.py` | Android battery + uptime monitor |

### AI & Learning

| Agent | File | Purpose |
|-------|------|---------|
| fm-azr-rl | `agents/fm_azr_rl.py` | Q-learning over mesh nodes |
| fm-rl-quad | `agents/fm_rl_quad.py` | 4-quadrant RL (revenue/compliance/infra/rep) |
| fm-enochian-hash | `agents/fm_enochian_hash.py` | φ-seeded multi-round SHA3-256 |

### Marketing & Content

| Agent | File | Purpose |
|-------|------|---------|
| fm-admob-bridge | `agents/fm_admob_bridge.py` | Google AdMob revenue tracking |
| fm-devto | `agents/marketing/fm_devto.py` | dev.to article publishing |
| fm-synthwave | `agents/fm_synthwave.py` | φ-harmonic content queue |

---

## Affiliate Program

FractalMesh operates affiliate partnerships with these services. All links loaded from vault — never hardcoded.

| Partner | Category | Commission | Vault Key |
|---------|----------|------------|-----------|
| **Proton VPN/Mail** | Privacy | 30-100% first year | `AFF_PROTON_URL` |
| **Mullvad VPN** | Privacy | Flat referral | `AFF_MULLVAD_URL` |
| **DigitalOcean** | Cloud | $25/referral | `AFF_DO_URL` |
| **Vultr** | Cloud | Up to $100 | `AFF_VULTR_URL` |
| **KuCoin** | Crypto Exchange | 20% fee share | `AFF_KUCOIN_URL` |
| **Pionex** | Crypto Trading Bots | 20% lifetime | `AFF_PIONEX_URL` |
| **Helium** | DePIN | Node deploy ref | `AFF_HELIUM_URL` |
| **XYO Network** | DePIN | Sentinel referral | `AFF_XYO_URL` |
| **BloFin** | Futures Trading | 20% rebate | `BLOFIN_REFERRAL_URL` |

To activate: set `ENABLE_AFFILIATE_LIVE="true"` in vault and add your referral URLs.

### Shout-Outs & Inspirations

This project is built on the shoulders of amazing open-source work:

- **[PM2](https://pm2.keymetrics.io/)** — battle-tested Node.js process manager that keeps Python agents alive
- **[LangChain](https://github.com/langchain-ai/langchain)** — the glue that makes multi-agent LLM pipelines manageable
- **[FastAPI](https://fastapi.tiangolo.com/)** — async Python APIs that don't get in the way
- **[SQLite WAL](https://www.sqlite.org/wal.html)** — surprisingly excellent for single-node sovereign state
- **[Helium Network](https://www.helium.com/)** — proof that DePIN can be real and profitable
- **[XYO Network](https://xyo.network/)** — geo-proof of origin on-chain
- **[Cloudflare Tunnels](https://www.cloudflare.com/products/tunnel/)** — free, zero-config public endpoints for home/Termux nodes
- **[Termux](https://termux.dev/)** — because your Android phone is a server
- **[Ollama](https://ollama.ai/)** — self-hosted LLMs without cloud bills
- **[Stripe](https://stripe.com/au)** — the gold standard for payment APIs
- **[WiGLE](https://wigle.net/)** — crowd-sourced WiFi frequency intelligence

---

## 4A Signature Service Tiers

| Tier | Label | Investment | Timeline |
|------|-------|-----------|----------|
| Audit | Infrastructure Audit | $750 AUD ex GST | 5 business days |
| Compliance | Compliance Automation | $1,500 AUD ex GST | 10 business days |
| Deploy | Sovereign Node Deploy | $2,500 AUD ex GST | 14 business days |
| Bundle | Full Sovereign Bundle | $4,500 AUD ex GST | 21 business days |

Payment terms: 50% upfront, 50% on delivery via Stripe.  
Book a call: [fractalmesh.net/contact.html](https://fractalmesh.net/contact.html)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              FRACTALMESH OMEGA TITAN v2.0.0              │
│          Sole Trader Sovereign Mesh Infrastructure       │
├──────────────┬──────────────┬──────────────┬────────────┤
│  RAG API     │ Billing API  │ Omni-Dash    │ Pulse Bus  │
│  :8001       │ :8003        │ :8090        │ :5060      │
├──────────────┴──────────────┴──────────────┴────────────┤
│  Enterprise Bus (circuit breakers, HMAC, monetization)  │
├─────────────────────────────────────────────────────────┤
│  OSINT Spider │ Negotiator │ Campaign Mgr │ Affiliates  │
├─────────────────────────────────────────────────────────┤
│  Licensing   │ Watermark  │ Smart Contracts │ Geo Valid │
├─────────────────────────────────────────────────────────┤
│  Carbon Credits │ AdMob │ WiGLE Topology │ RL Engines  │
├─────────────────────────────────────────────────────────┤
│  sovereign.db (SQLite WAL) │ ~/.secrets/fractal.env     │
└─────────────────────────────────────────────────────────┘
```

### φ-Harmonic Scoring

All prioritisation uses the golden ratio φ = 1.6180339887:

- **φ-score** = `value × φ^layer_index` — higher layers compound exponentially
- **φ-decay** = `sum(base / φ^i)` — convergent diminishing returns for risk
- **Geometric stacking** = `sum(base × φ^i for i in 0..n)` — royalty compounding
- **Enochian hash** = 7-round φ-seeded SHA3-256 for content fingerprinting

---

## Automation Features

### Automatic Licensing & Copyright
`fm_licensing.py` stamps every generated document with:
- SPDX license headers
- Copyright notice with year, operator name, ABN
- Registers to `licensing_log` in sovereign.db

### Automatic Watermarking
`fm_watermark.py` applies invisible and visible watermarks to:
- PDF contracts and proposals (via reportlab)
- Markdown documents (metadata injection)
- Images (PIL-based steganographic + visible)

### Smart Contract Automation
`fm_smart_contracts.py` generates and tracks:
- ERC-20 token contracts (DePIN revenue tokens)
- Simple payment split contracts
- Royalty distribution contracts
- All stored in `smart_contracts` table

### Geo Validation & Geotagging
`fm_geo_validator.py` validates and tags all outreach:
- Verifies prospect coordinates against known business regions
- Tags leads with lat/lon metadata
- Albury NSW bounding box primary target
- Expands to NSW/VIC corridor

### Carbon Credit Analysis
`fm_carbon_credits.py` monitors:
- Energy consumption of node fleet (RPi + Android)
- Estimated CO2 offset opportunities
- Generates reports to `carbon_reports` table
- Integrates with Gold Standard API when available

### AdMob Integration
`fm_admob_bridge.py` tracks:
- Ad unit revenue across FractalMesh companion apps
- RPM, impressions, fill rate
- Correlates with product performance
- Logs to `admob_revenue` table

### Campaign Manager
`fm_campaign_manager.py` manages:
- Multi-channel outreach sequences (email → follow-up → proposal)
- A/B test tracking
- Prospect state machine (prospect → warm → proposal → closed)
- ROI per campaign

---

## Directory Structure

```
fractalmesh/
├── README.md                    ← You are here
├── ecosystem.config.js          ← PM2 swarm config (29+ agents)
├── agents/
│   ├── fm_pulse_bus.py          ← HMAC event bus
│   ├── fm_negotiator.py         ← 4A proposal engine
│   ├── fm_osint_spider.py       ← 18-dork lead discovery
│   ├── fm_enterprise_bus.py     ← Circuit-breaker bus
│   ├── fm_licensing.py          ← Auto copyright/license
│   ├── fm_watermark.py          ← Document watermarking
│   ├── fm_smart_contracts.py    ← EVM contract management
│   ├── fm_geo_validator.py      ← Geo validation + tagging
│   ├── fm_carbon_credits.py     ← Carbon credit analysis
│   ├── fm_admob_bridge.py       ← AdMob revenue tracking
│   ├── marketing/
│   │   ├── fm_campaign_manager.py
│   │   └── fm_devto.py
│   └── system/
│       ├── git_oracle.py
│       └── omni_graph.py
├── api/
│   ├── omni_dashboard.py        ← Flask control panel :8090
│   ├── health_app.py            ← Health endpoint :5057
│   ├── rag_api.py               ← FastAPI RAG :8001
│   └── billing_api.py           ← FastAPI billing :8003
├── modules/
│   └── fractal_royalty_engine.py
├── scripts/
│   └── deploy_omega_v40.sh      ← One-shot deploy
└── dist/                        ← Generated contracts/proposals
```

---

## Security Notes

1. **Vault** (`~/.secrets/fractal.env`) must be `chmod 600` — deploy script enforces this
2. **TFN** — never stored or logged; only presence boolean checked in `fm_lba_bridge.py`
3. **Trading keys** — rotate every 90 days minimum; set calendar reminder
4. **Seed phrases** — never in vault; hardware wallet or encrypted storage only
5. **DASHBOARD_TOKEN** — set this before exposing port 8090 externally
6. **BUS_SECRET** — 32-byte hex; regenerate if any agent logs are compromised
7. All agents default to `DRY_RUN=true` until you set the `ENABLE_*` flag in vault

---

## Contact & Services

**Samuel James Hiotis**  
ABN: 56 628 117 363  
Location: Albury NSW 2640, Australia  
Web: [fractalmesh.net](https://fractalmesh.net)  
Contact: [fractalmesh.net/contact.html](https://fractalmesh.net/contact.html)  
Products: [fractalmesh.net/products.html](https://fractalmesh.net/products.html)  

For consulting enquiries, automation proposals, or DePIN node deployment — use the contact form or the fm-negotiator will reach out automatically once your details appear in the lead pipeline.

---

*FractalMesh is operated by Samuel James Hiotis as a sole trader under ABN 56 628 117 363. No company has been registered. All services are provided under Australian Consumer Law.*
