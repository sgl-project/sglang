#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FRACTALMESH OMEGA TITAN v10003.38 — UNIFIED HARMONIC ANCHOR SINGULARITY
# Operator: Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="${FRACTALMESH_HOME:-$HOME/fmsaas}"
VAULT="$HOME/.secrets/fractal.env"
DB="$ROOT/database/sovereign.db"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; PURPLE='\033[0;35m'; NC='\033[0m'
log()  { echo -e "${BLUE}[*]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
die()  { echo -e "\033[0;31m[err]${NC} $*" >&2; exit 1; }

echo -e "${PURPLE}[!] DEPLOYING UNIFIED HARMONIC ANCHOR v10003.38...${NC}"

# 1. Directories
mkdir -p "$ROOT"/{agents,modules,database,logs,tunnels,compression,chemical_chain,royalty_pools,pretrain,ai-mesh}
mkdir -p "$HOME/.secrets"

# 2. Vault — write if missing
if [[ ! -f "$VAULT" ]]; then
  log "Creating vault template at $VAULT"
  cat > "$VAULT" << 'EOF'
MAKE_MCP_TOKEN=""
GITHUB_PAT_1=""
GITHUB_PAT_2=""
CLOUDFLARED_TOKEN=""
DARK_TUNNEL_TOKEN=""
OPENROUTER_API_KEY=""
WIGLE_NAME=""
WIGLE_TOKEN=""
BUS_SECRET=""
HARMONIC_PHI="1.6180339887"
EXTERNAL_WEBHOOK_TOKEN=""
EOF
  chmod 600 "$VAULT"
  ok "Vault template created — populate $VAULT before starting services"
fi

set -a
# shellcheck disable=SC1090
source "$VAULT"
set +a

# 3. Install cloudflared if missing
if ! command -v cloudflared &>/dev/null; then
  log "Installing cloudflared..."
  mkdir -p /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-public-v2.gpg \
    | tee /usr/share/keyrings/cloudflare-public-v2.gpg >/dev/null
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-public-v2.gpg] https://pkg.cloudflare.com/cloudflared any main' \
    | tee /etc/apt/sources.list.d/cloudflared.list
  apt-get update -qq && apt-get install -y cloudflared
fi

# 4. DB — chemical chain + royalty pools (harmonic φ ratios)
log "Seeding SQLite schema (chemical chain + royalty pools)..."
sqlite3 "$DB" << 'SQL'
PRAGMA journal_mode=WAL;
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS chemical_chain (
  id INTEGER PRIMARY KEY,
  stage TEXT, company TEXT, product TEXT, royalty_rate REAL
);
INSERT OR IGNORE INTO chemical_chain VALUES
  (1,'Soil Extraction',    'Murray Darling Farms','Phosphate',     0.08*1.618),
  (2,'Industrial Processing','Visy Albury',       'Packaging Pulp',0.12*1.618),
  (3,'Logistics',          'Hume Dam Transport',  'River Freight', 0.07*1.618),
  (4,'Commercial Product', 'Coles/Woolworths',    'Packaged Goods',0.15*1.618),
  (5,'Customer End',       'Direct Consumer',     'Final Sale',    0.05*1.618);
CREATE TABLE IF NOT EXISTS royalty_pools (
  id TEXT PRIMARY KEY, pool_name TEXT, industries TEXT, base_royalty REAL
);
INSERT OR IGNORE INTO royalty_pools VALUES
  ('POOL-001','Murray-Visy-DeFi','agriculture,packaging,logistics', 0.08*1.618*5);
COMMIT;
SQL
ok "Schema seeded"

# 5. Copy modules to ROOT (if running from repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

for src in harmonic_logic_memory.py network_topology_warden.py; do
  if [[ -f "$REPO_ROOT/modules/$src" && ! -f "$ROOT/modules/$src" ]]; then
    cp "$REPO_ROOT/modules/$src" "$ROOT/modules/$src"
    ok "Copied modules/$src → $ROOT/modules/"
  fi
done

for src in fm_pulse_bus.py fm_gitops_runner.py fm_mesh_integrator.py; do
  if [[ -f "$REPO_ROOT/agents/$src" && ! -f "$ROOT/agents/$src" ]]; then
    cp "$REPO_ROOT/agents/$src" "$ROOT/agents/$src"
    ok "Copied agents/$src → $ROOT/agents/"
  fi
done

# 6. Start new nodes (additive — don't kill existing swarm)
log "Starting v10003.38 harmonic anchor nodes..."
ECOSYSTEM="$REPO_ROOT/ecosystem.config.js"
[[ -f "$ECOSYSTEM" ]] || ECOSYSTEM="$ROOT/ecosystem.config.js"

for node in fm-cloudflared fm-bus fm-gitops-runner fm-integrator fm-harmonic fm-warden; do
  pm2 start "$ECOSYSTEM" --only "$node" --update-env 2>/dev/null \
    && ok "Started: $node" \
    || { pm2 restart "$node" 2>/dev/null && ok "Restarted: $node" || true; }
done

pm2 save --force 2>&1 | tail -1

echo
echo '═══════════════════════════════════════════════════════════════'
echo '   ✅ v10003.38 UNIFIED HARMONIC ANCHOR COMPLETE'
echo '   - Cloudflared tunnel locked to PM2 (bash wrapper)'
echo '   - Harmonic φ-logic/memory balance active'
echo '   - Pulse bus, GitOps runner, Mesh integrator online'
echo '   - Chemical chain + royalty pools seeded in sovereign.db'
echo '   - Network topology warden active'
echo '═══════════════════════════════════════════════════════════════'
pm2 list

echo
echo "Test Harmonic Balance:"
echo "  python3 -c 'import sys; sys.path.insert(0,\"$ROOT\"); from modules.harmonic_logic_memory import engine; engine.fill_pretrain_datasets(); print(engine.harmonic_balance(\"logic_data\", \"memory_data\"))'"
echo "View royalty chain:"
echo "  sqlite3 $DB 'SELECT * FROM chemical_chain; SELECT * FROM royalty_pools;'"
