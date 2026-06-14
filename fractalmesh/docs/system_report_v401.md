# FRACTALMESH SOVEREIGN v401.6 — SYSTEM DOCUMENTATION

**Deployment Date:** 2026-03-11
**Principal:** Samuel James Hiotis (Sole Trader, ABN 56628117363)
**Infrastructure:** Termux Android / Proot-Distro Ubuntu / ARM64
**Location:** Albury NSW 2640
**Status:** Production — 21 Agents Operational

---

## Executive Summary

FRACTALMESH SOVEREIGN v401.6 is a **mobile-native autonomous income system** operating entirely from an Android Termux environment on ARM64 hardware. The architecture prioritizes:

1. **Edge Sovereignty** — All credentials, databases, and agents local to device
2. **Zero External Infrastructure** — Gmail alerts replace Telegram (no phone number required)
3. **Modular Monetization** — 7 distinct revenue streams via API integration
4. **QBM (Quantized Boxing Method)** — Memory-constrained PM2 process isolation

---

## Agent Mesh Topology

| Layer | Agents | Function |
|-------|--------|----------|
| **Financial Core** | fm-trading, fm-arbitrage, fm-stripe, fm-gumroad | Liquidity conversion |
| **Content Engine** | fm-devto-ads, fm-rag-feed, fm-shopfront-pro | Audience acquisition |
| **Asset Generation** | fm-nft-minter, fm-royalty, fm-affiliate | IP monetization |
| **Intelligence** | fm-orchestrator, fm-gbwigle, fm-lead | Opportunity identification |
| **Infrastructure** | fm-pod, fm-tunnel, geosignal, fm-dashboard | System integrity |
| **Delivery** | fm-delivery, rf-bridge | Fulfillment + telemetry sync |

---

## Full Agent Inventory (v401.6)

| ID | Agent | Function | Port | Memory | Status |
|----|-------|----------|------|--------|--------|
| 0 | fm-dashboard | System monitoring | 8088 | ~5MB | Online |
| 1 | geosignal | Geospatial telemetry | 5057 | ~16MB | Online |
| 2 | fm-trading | Trading engine (KuCoin/Pionex) | — | ~6MB | Online |
| 3 | fm-arbitrage | Arbitrage scanner | — | ~17MB | Online |
| 4 | fm-stripe | Payment processing | 5059 | ~5MB | Online |
| 5 | fm-gumroad | Digital product storefront | — | ~5MB | Online |
| 6 | fm-pod | Master API / VTS bridge | 5058 | ~5MB | Online |
| 7 | fm-bounty | Bug bounty automation | — | ~7MB | Online |
| 8 | fm-devto | Dev.to content (legacy) | — | ~6MB | Online |
| 9 | fm-affiliate | Affiliate tracking | — | ~6MB | Online |
| 10 | fm-advert | Ad rotation engine | — | ~7MB | Online |
| 11 | fm-tg-report | Alert reports | — | ~5MB | Online |
| 12 | fm-tunnel | SSH/serveo tunnel | — | ~2MB | Online |
| 13 | fm-royalty | NFT royalty tracker | — | ~6MB | Online |
| 14 | fm-stock-rotation | Product promotion | — | ~6MB | Online |
| 15 | fm-rag-feed | RAG knowledge ingestion | 8087 | ~7MB | Online |
| 16 | fm-shopfront-pro | Cyberpunk storefront | 8088 | ~5MB | Online |
| 18 | fm-gbwigle | GBWiGLE RF capture | — | ~7MB | Online |
| 19 | fm-lead | Lead generation (Albury) | 5060 | ~6MB | Online |
| 21 | fm-devto-ads | AI content publisher | — | ~29MB | Online |
| 22 | fm-nft-minter | Fractal NFT / IPFS | — | ~26MB | Online |
| 23 | fm-orchestrator | OpenRouter AI gateway | 5055 | ~35MB | Online |

**Total Memory Footprint:** ~215MB
**Total Agents:** 21 (100% operational)
**Restart Count:** 0 across all agents

---

## Monetization Streams

| Stream | Mechanism | Status |
|--------|-----------|--------|
| Crypto Trading | KuCoin/Pionex HMAC API | Active |
| Stripe Payments | Live checkout (`sk_live_*`) | Active |
| Digital Products | Gumroad storefront | Active |
| Dev.to Content | Automated articles + affiliate footers | Active |
| NFT Minting | Pinata IPFS → OpenSea Polygon | Active |
| Affiliate Commissions | KuCoin/Pionex/Crypto.com referrals | Tracking |
| AI Services | OpenRouter gateway | Active |

---

## API Integration Matrix

| Service | Credential Key | Agent | Status |
|---------|---------------|-------|--------|
| Dev.to | `DEVTO_API_KEY` | fm-devto-ads | Active |
| Pinata IPFS | `PINATA_JWT` | fm-nft-minter | Active |
| OpenRouter | `OPENROUTER_API_KEY` | fm-orchestrator | Active |
| Stripe | `STRIPE_SECRET_KEY` | fm-stripe | Active |
| Gmail SMTP | `GMAIL_APP_PASS` | all alerts | Active |
| KuCoin | `KUCOIN_API_KEY` | fm-trading | Active |
| Pionex | `PIONEX_API_KEY` | fm-trading | Active |

*Credentials stored in `~/.secrets/fractal.env` (chmod 600). Never committed to version control.*

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  TERMUX / ANDROID (ARM64)                │
│              Proot-Distro Ubuntu — Local Only           │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  PM2 v5+     │  │  SQLite WAL  │  │  Flask APIs  │  │
│  │  Process Mgr │  │  sovereign   │  │  5055–8090   │  │
│  │  21 agents   │  │  .db         │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│           AGENT MESH LAYERS (21 nodes, ~215MB)          │
│  Trading → Arbitrage → Royalty → NFT → AI Orchestrator  │
│  Shopfront → Content → Leads → RF Telemetry → Delivery  │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Stripe Live │  │  OpenRouter  │  │  Pinata IPFS │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Dev.to API  │  │  KuCoin REST │  │  Gmail SMTP  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Security Posture

- No hardcoded credentials — all keys loaded from `~/.secrets/fractal.env`
- No Telegram dependency — Gmail SMTP alerts remove phone number exposure
- SQLite WAL mode — database integrity on mobile NAND storage
- PM2 process isolation with per-agent memory ceilings (QBM)
- `.env` files chmod 600, excluded from version control via `.gitignore`

---

## Capacity Assessment

21 agents on mobile ARM64 is **optimal capacity** for this hardware tier:

- Sustained RAM usage ~215MB — within Android low-memory threshold
- CPU thermal ceiling: 3W TDP sustained on 8 cores @ 2.8GHz
- Any additional agents risk Android Signal 9 (OOM kill)

**Recommendation:** Monitor revenue 48h before further expansion. Current monetization stack is feature-complete.

---

## Port Map

| Port | Service |
|------|---------|
| 5055 | fm-orchestrator (OpenRouter gateway) |
| 5057 | geosignal (RF/location API) |
| 5058 | fm-pod (Master API) |
| 5059 | fm-stripe (Stripe processing) |
| 5060 | fm-lead (Lead generation) |
| 8087 | fm-rag-feed (RAG knowledge) |
| 8088 | fm-dashboard / fm-shopfront-pro |
| 8090 | fm-dashboard (main web UI) |

---

*Samuel James Hiotis | Sole Trader | ABN 56628117363 | Albury NSW 2640*
*Document Version: v401.6 | March 2026*
