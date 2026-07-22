# AI NODE BLUEPRINT
**By Samuel James Hiotis | ABN 56 628 117 363**

*Sovereign Architecture for Autonomous AI Agents on Local Hardware*

---

## Overview

A complete sovereign architecture for 18+ autonomous AI agents operating on Samsung Android hardware and local edge nodes.

**Core Principles:**
- Zero Cloud dependency for critical operations
- Zero ongoing subscriptions for core mesh
- Natively immortal: full state resurrection from a single boot command
- Hardware-bound secrets via vault pattern (chmod 600)
- WAL-locked SQLite for audit-ready operations

---

## Architecture

### Edge Node Stack
- **Runtime:** Python 3 + PM2 process manager
- **Database:** SQLite 3 (WAL mode) — sovereign.db
- **Secrets:** `~/.secrets/fractal.env` (vault, never committed)
- **Tunnel:** Cloudflare tunnel via `tunnels/start_tunnel.sh`
- **API gateway:** Flask (port 8080)
- **Health endpoint:** Flask (port 5057)

### Agent Categories

| Category | Agents | Purpose |
|----------|--------|---------|
| Revenue | fm_stripe_mon, fm_stripe_gateway, fm_micro_charge | Payment processing + delivery |
| Intelligence | fm_dork_engine, fm_wigle_oracle, fm_geo_oracle | Recon + RF intelligence |
| Compliance | fm_sovereign_ops, fm_lba_bridge, fm_oversight | Identity + audit |
| Automation | fm_workspace_sync, fm_gitops_runner, fm_email_listener | Workflow automation |
| Infrastructure | fm_pulse_bus, fm_dashboard, fm_healer | Mesh backbone |

### Security Model
- All credentials in vault (`~/.secrets/fractal.env`, chmod 600)
- HMAC-SHA256 payload signing via `security_core.py`
- TFN and sensitive PII never logged or transmitted
- Defensive-only posture — no offensive capabilities

---

## Hardware Immortality Protocol

The mesh can be fully resurrected from any hardware using:

```bash
git clone <repo>
bash fractalmesh/scripts/deploy_v10057_00.sh
```

Prerequisites:
1. Populate `~/.secrets/fractal.env` with real values
2. Ensure `cloudflared`, `python3`, `pm2`, `sqlite3` are installed

---

## Deployment Tiers

| Tier | Hardware | Nodes | Use Case |
|------|----------|-------|----------|
| Micro | Raspberry Pi 4 | 3–5 | RF scanning + relay |
| Standard | Samsung Android + Pi | 10–15 | Full mesh |
| Full | Dedicated Linux box | 26+ | Apex swarm |

---

## Support
Samuel James Hiotis | ABN 56 628 117 363 | Albury, NSW
Phone: see vault (PHONE key)
Email: see vault (GMAIL_USER key)

*White-hat, defensive operations only.*
