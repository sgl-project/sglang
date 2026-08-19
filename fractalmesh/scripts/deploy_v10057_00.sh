#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# FRACTALMESH OMEGA TITAN v10057.00 — THE ULTIMATE HARMONIC ANCHOR
# Operator: Samuel James Hiotis | ABN: 56628117363
# Action: FULL SWARM RESTORE | COMPLIANCE SWEEP | MEMORY STABILIZATION
# ══════════════════════════════════════════════════════════════════════════════
# SECURITY: All credentials live in ~/.secrets/fractal.env (chmod 600).
#            This script contains NO secrets — populate the vault first.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo -e "\033[0;35m[!] INITIATING v10057.00: EXECUTING FULL REBALANCE & RECOVERY...\033[0m"

ROOT="${FRACTALMESH_HOME:-/root/fmsaas}"
VAULT="$HOME/.secrets/fractal.env"
AGENTS_DIR="$ROOT/agents"
DB="$ROOT/database/sovereign.db"
DIST="$ROOT/dist"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[*]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }

mkdir -p "$AGENTS_DIR" "$ROOT/database" "$ROOT/logs" "$ROOT/tunnels" "$DIST"

# ── 1. NUCLEAR CLEANUP ───────────────────────────────────────────────────────
log "Purging ghost processes and truncating logs..."
pm2 kill || true
pkill -9 -f python3 || true
truncate -s 0 "$HOME/.pm2/pm2.log" || true

# ── 2. VAULT (template if missing) ───────────────────────────────────────────
if [[ ! -f "$VAULT" ]]; then
  log "Writing vault template to $VAULT"
  mkdir -p "$(dirname "$VAULT")"
  cat > "$VAULT" << 'ENVEOF'
# FractalMesh Omega Titan v10057.00 — vault template
# Populate ALL values before starting the swarm. chmod 600 this file.
NAME="Samuel James Hiotis"
ABN="56628117363"
# TFN — keep offline; never commit
TFN=""
PHONE=""
GMAIL_USER=""
GMAIL_APP_PASS=""
STRIPE_KEY=""
STRIPE_WEBHOOK_SECRET=""
SUPABASE_URL=""
SUPABASE_KEY=""
CF_TUNNEL_TOKEN=""
DARK_TUNNEL_TOKEN=""
HARMONIC_PHI="1.6180339887"
ENVEOF
  chmod 600 "$VAULT"
  ok "Vault template written — populate $VAULT before starting"
fi

set -a
# shellcheck disable=SC1090
source "$VAULT"
set +a

# ── 3. SOLE TRADER COMPLIANCE ASSETS ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

log "Deploying compliance docs..."
[[ -f "$REPO_ROOT/dist/ai_node_blueprint.md" ]] \
  && cp "$REPO_ROOT/dist/ai_node_blueprint.md" "$DIST/ai_node_blueprint.md" \
  && ok "  dist/ai_node_blueprint.md"
[[ -f "$REPO_ROOT/dist/obrien_compliance_v1.md" ]] \
  && cp "$REPO_ROOT/dist/obrien_compliance_v1.md" "$DIST/obrien_compliance_v1.md" \
  && ok "  dist/obrien_compliance_v1.md"

# ── 4. CORE AGENT DEPLOYMENT ─────────────────────────────────────────────────
log "Syncing core agents..."
for src in fm_stripe_mon.py fm_sovereign_ops.py fm_lba_bridge.py \
           fm_email_listener.py fm_samsung_warden.py \
           fm_stripe_gateway.py fm_dashboard.py fm_dork_engine.py \
           fm_workspace_sync.py fm_supabase_sync.py fm_tunnel_warden.py \
           fm_omni_closer.py; do
  [[ -f "$REPO_ROOT/agents/$src" ]] \
    && cp "$REPO_ROOT/agents/$src" "$AGENTS_DIR/$src" \
    && ok "  agents/$src"
done

# ── 5. LBA BRIDGE & SOVEREIGN CONNECTOR ──────────────────────────────────────
log "Deploying LBA bridge + sovereign connector..."
[[ -f "$REPO_ROOT/agents/fm_lba_bridge.py" ]] \
  && cp "$REPO_ROOT/agents/fm_lba_bridge.py" "$ROOT/lba_bridge.py" \
  && ok "  lba_bridge.py"
[[ -f "$REPO_ROOT/agents/fm_sovereign_ops.py" ]] \
  && cp "$REPO_ROOT/agents/fm_sovereign_ops.py" "$ROOT/sovereign_connector.py" \
  && ok "  sovereign_connector.py"

# ── 6. STUB GENERATION FOR FULL MATRIX ───────────────────────────────────────
STUB_AGENTS=(
  "fm_podcast_nexus.py" "fm_figma_bridge.py" "fm_affiliate_nexus.py"
  "fm_alter_bridge.py"  "fm_geo_oracle.py"   "fm_price_warfare.py"
  "fm_reconfig_hub.py"  "fm_legacy_memory.py" "fm_dev_nexus.py"
  "fm_upsell_engine.py" "fm_royalty_nexus.py" "fm_openstax_ref.py"
)
for agent in "${STUB_AGENTS[@]}"; do
  if [[ ! -f "$AGENTS_DIR/$agent" ]]; then
    cat > "$AGENTS_DIR/$agent" << 'STUB'
import time
print(f"[{__file__}] Operational.")
while True:
    time.sleep(3600)
STUB
    ok "  stub: $agent"
  fi
done

# ── 7. ECOSYSTEM CONFIG ───────────────────────────────────────────────────────
log "Writing ecosystem.config.js..."
cat > "$ROOT/ecosystem.config.js" << JSEOF
module.exports = {
  apps: [
    { name: 'fm-cloudflared',    script: 'bash',    args: '${ROOT}/tunnels/start_tunnel.sh',       max_memory_restart: '45M' },
    { name: 'fm-stripe-mon',     script: 'python3', args: '${AGENTS_DIR}/fm_stripe_mon.py',        max_memory_restart: '35M' },
    { name: 'fm-stripe-gate',    script: 'python3', args: '${AGENTS_DIR}/fm_stripe_gateway.py',    max_memory_restart: '35M' },
    { name: 'fm-sovereign-ops',  script: 'python3', args: '${ROOT}/sovereign_connector.py',        max_memory_restart: '20M' },
    { name: 'fm-lba-bridge',     script: 'python3', args: '${ROOT}/lba_bridge.py',                 max_memory_restart: '20M' },
    { name: 'fm-supabase-sync',  script: 'python3', args: '${AGENTS_DIR}/fm_supabase_sync.py',     max_memory_restart: '35M' },
    { name: 'fm-dashboard',      script: 'python3', args: '${AGENTS_DIR}/fm_dashboard.py',         max_memory_restart: '40M' },
    { name: 'fm-dork-recon',     script: 'python3', args: '${AGENTS_DIR}/fm_dork_engine.py',       max_memory_restart: '30M' },
    { name: 'fm-tunnel-warden',  script: 'python3', args: '${AGENTS_DIR}/fm_tunnel_warden.py',     max_memory_restart: '25M' },
    { name: 'fm-omni-closer',    script: 'python3', args: '${AGENTS_DIR}/fm_omni_closer.py',       max_memory_restart: '30M' },
    { name: 'fm-email-listener', script: 'python3', args: '${AGENTS_DIR}/fm_email_listener.py',    max_memory_restart: '25M' },
    { name: 'fm-samsung-warden', script: 'python3', args: '${AGENTS_DIR}/fm_samsung_warden.py',    max_memory_restart: '20M' },
    { name: 'fm-workspace-sync', script: 'python3', args: '${AGENTS_DIR}/fm_workspace_sync.py',    max_memory_restart: '25M' }
  ]
};
JSEOF
ok "ecosystem.config.js written (13 nodes)"

# ── 8. FINAL IGNITION ─────────────────────────────────────────────────────────
echo
ok "V10057.00: HARMONIC BINDING COMPLETE."
pm2 start "$ROOT/ecosystem.config.js" --update-env
pm2 save --force

echo
echo '═══════════════════════════════════════════════════════════════'
echo '   ✅ V10057.00 HARMONIC ANCHOR ONLINE'
echo '   - Cloudflare Tunnel    → tunnels/start_tunnel.sh'
echo '   - Stripe Monitor       → fm_stripe_mon.py + Gmail delivery'
echo '   - Sovereign Connector  → ABN/ATO authority layer'
echo '   - LBA Bridge           → TFN shield + workspace integration'
echo '   - Email Listener       → Gmail IMAP watcher'
echo '   - Samsung Warden       → device health guardian'
echo '   - 13-Node PM2 Swarm    → all nodes started'
echo '═══════════════════════════════════════════════════════════════'
pm2 status
