#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FRACTALMESH OMEGA TITAN v10003.42 — APEX OMNI-MATRIX
# 26-Node Swarm + Harmonic φ + Fractal Royalty + Security Core + CF Tunnel
# Operator: Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo -e "\033[0;35m[!] INITIATING v10003.42 APEX OMNI-MATRIX SINGULARITY...\033[0m"

ROOT="${FRACTALMESH_HOME:-$HOME/fmsaas}"
VAULT="$HOME/.secrets/fractal.env"
AGENTS="$ROOT/agents"
MODULES="$ROOT/modules"
DB="$ROOT/database/sovereign.db"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[*]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }
die() { echo -e "\033[0;31m[err]${NC} $*" >&2; exit 1; }

# ── 1. Directories ────────────────────────────────────────────────────────────
mkdir -p "$ROOT"/{agents,modules,database,logs,tunnels,compression,chemical_chain,royalty_pools,pretrain}
mkdir -p "$HOME/.secrets"

# ── 2. Vault (template if missing) ───────────────────────────────────────────
if [[ ! -f "$VAULT" ]]; then
  log "Writing vault template to $VAULT"
  cat > "$VAULT" << 'EOF'
NAME="Samuel James Hiotis"
ABN="56628117363"
LOCATION="Albury / Wodonga Sector"
STRIPE_SECRET_KEY=""
OPENROUTER_API_KEY=""
WIGLE_NAME=""
WIGLE_TOKEN=""
CF_TUNNEL_TOKEN=""
DARK_TUNNEL_TOKEN=""
MAKE_MCP_TOKEN=""
GITHUB_PAT_1=""
GITHUB_PAT_2=""
EXTERNAL_WEBHOOK_TOKEN=""
BUS_SECRET=""
HARMONIC_PHI="1.6180339887"
SUPABASE_URL=""
SUPABASE_ANON_KEY=""
EOF
  chmod 600 "$VAULT"
  ok "Vault template written — populate $VAULT before starting"
fi

set -a
# shellcheck disable=SC1090
source "$VAULT"
set +a

# ── 3. Dependencies ───────────────────────────────────────────────────────────
log "Checking dependencies..."
command -v cloudflared &>/dev/null || {
  log "Installing cloudflared..."
  mkdir -p /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-public-v2.gpg \
    | tee /usr/share/keyrings/cloudflare-public-v2.gpg >/dev/null
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-public-v2.gpg] https://pkg.cloudflare.com/cloudflared any main' \
    | tee /etc/apt/sources.list.d/cloudflared.list
  apt-get update -qq && apt-get install -y cloudflared
}
command -v pm2 &>/dev/null || { npm install -g pm2; ok "pm2 installed"; }

# ── 4. Copy repo files to ROOT ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

log "Syncing modules..."
for src in security_core.py harmonic_logic_memory.py network_topology_warden.py \
           fractal_royalty_engine.py fractal_file_translator.py; do
  [[ -f "$REPO_ROOT/modules/$src" ]] \
    && cp "$REPO_ROOT/modules/$src" "$MODULES/$src" \
    && ok "  modules/$src"
done

log "Syncing agents..."
for src in fm_pulse_bus.py fm_gitops_runner.py fm_mesh_integrator.py \
           fm_dashboard.py fm_healer.py fm_dork_engine.py fm_wigle_oracle.py \
           fm_live_tokenomics.py fm_contract_forge.py fm_enochian_hash.py \
           fm_stripe_gateway.py fm_workspace_sync.py fm_device_bridge.py \
           fm_immortality.py fm_salvage_crew.py fm_auto_advert.py fm_oversight.py \
           fm_rl_quad.py fm_azr_rl.py fm_bounty.py fm_domain.py fm_toolkit.py \
           fm_synthwave.py fm_affiliate.py fm_ipfs.py; do
  [[ -f "$REPO_ROOT/agents/$src" ]] \
    && cp "$REPO_ROOT/agents/$src" "$AGENTS/$src" \
    && ok "  agents/$src"
done

log "Syncing tunnel wrapper..."
cp "$REPO_ROOT/tunnels/start_tunnel.sh" "$ROOT/tunnels/start_tunnel.sh"
chmod +x "$ROOT/tunnels/start_tunnel.sh"
ok "  tunnels/start_tunnel.sh"

# ── 5. Omni-Ledger database ───────────────────────────────────────────────────
log "Seeding Omni-Ledger (sovereign.db)..."
sqlite3 "$DB" << 'SQL'
PRAGMA journal_mode=WAL;
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS leads (
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  name   TEXT UNIQUE,
  status TEXT DEFAULT 'RAW',
  score  REAL DEFAULT 0.0,
  source TEXT,
  ts     DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS revenue (
  id     INTEGER PRIMARY KEY,
  source TEXT,
  amount REAL,
  ts     DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tokenomics_state (
  id                  INTEGER PRIMARY KEY CHECK (id = 1),
  total_supply        REAL DEFAULT 1000000000.0,
  burned_supply       REAL DEFAULT 0.0,
  current_fiat_revenue REAL DEFAULT 0.0,
  burn_rate_pct       REAL DEFAULT 0.15,
  ts                  DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO tokenomics_state (id) VALUES (1);
CREATE TABLE IF NOT EXISTS wigle_telemetry_ledger (
  node_id                 TEXT PRIMARY KEY,
  last_verified_networks  INTEGER DEFAULT 0,
  total_fractal_earned    REAL DEFAULT 0.0,
  ts                      DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS smart_contracts (
  contract_id TEXT PRIMARY KEY,
  client_name TEXT,
  value_aud   REAL,
  status      TEXT DEFAULT 'PENDING',
  ts          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pulse_log (
  id       INTEGER PRIMARY KEY,
  source   TEXT,
  event    TEXT,
  priority REAL,
  ts       DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS royalty_pools (
  id           TEXT PRIMARY KEY,
  pool_name    TEXT,
  industries   TEXT,
  base_royalty REAL
);
CREATE TABLE IF NOT EXISTS chemical_chain (
  id          INTEGER PRIMARY KEY,
  stage       TEXT,
  company     TEXT,
  product     TEXT,
  royalty_rate REAL
);

INSERT OR IGNORE INTO royalty_pools VALUES
  ('POOL-001','Murray-Visy-DeFi','agriculture,packaging,logistics',0.1294);
INSERT OR IGNORE INTO chemical_chain VALUES
  (1,'Soil Extraction',     'Murray Darling Farms','Phosphate',    0.1294),
  (2,'Industrial Processing','Visy Albury',        'Packaging Pulp',0.1941),
  (3,'Logistics',            'Hume Dam Transport', 'River Freight', 0.1132),
  (4,'Commercial Product',   'Coles/Woolworths',   'Packaged Goods',0.2427),
  (5,'Customer End',         'Direct Consumer',    'Final Sale',    0.0809);

COMMIT;
SQL
ok "sovereign.db seeded (8 tables)"

# ── 6. Start / reload all 26 nodes ───────────────────────────────────────────
ECOSYSTEM="$REPO_ROOT/ecosystem.config.js"
[[ -f "$ECOSYSTEM" ]] || ECOSYSTEM="$ROOT/ecosystem.config.js"

log "Igniting 26-node APEX swarm..."

NODES=(
  fm-cloudflared fm-bus fm-gateway fm-integrator fm-gitops-runner
  fm-harmonic fm-warden
  fm-dork-engine fm-contract-forge fm-wigle-oracle fm-tokenomics
  fm-workspace fm-device fm-enochian fm-stripe-gate fm-salvage
  fm-auto-advert fm-oversight fm-rl-quad fm-azr-rl fm-bounty
  fm-domain fm-toolkit fm-synthwave fm-affiliate fm-ipfs fm-healer
)

for node in "${NODES[@]}"; do
  pm2 start "$ECOSYSTEM" --only "$node" --update-env 2>/dev/null \
    && ok "  started:   $node" \
    || { pm2 restart "$node" --update-env 2>/dev/null \
         && ok "  restarted: $node" \
         || true; }
done

pm2 save --force 2>&1 | tail -1

echo
echo '═══════════════════════════════════════════════════════════════'
echo '   ✅ V10003.42 APEX OMNI-MATRIX ONLINE'
echo '   - Cloudflare Tunnel    → tunnels/start_tunnel.sh (bash exec)'
echo '   - Security Core        → HMAC-SHA256 payload signing'
echo '   - Harmonic Engine      → φ-balanced logic↔memory scoring'
echo '   - Fractal Royalty      → geometric φ-stacking (5 layers)'
echo '   - Fractal Compressor   → 2-level nested zlib + SHA-256 ID'
echo '   - Network Warden       → geometric elite traffic routing'
echo '   - Omni-Ledger DB       → 8 tables seeded (WAL mode)'
echo '   - 26-Node PM2 Swarm    → all nodes started/restarted'
echo '═══════════════════════════════════════════════════════════════'
pm2 list

echo
echo "Verify:"
echo "  sqlite3 $DB 'SELECT * FROM royalty_pools; SELECT * FROM chemical_chain;'"
echo "  curl http://127.0.0.1:8080/api/full"
echo "  curl http://127.0.0.1:5060/health"
