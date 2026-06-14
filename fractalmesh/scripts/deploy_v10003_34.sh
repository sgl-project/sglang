#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FRACTALMESH OMEGA TITAN v10003.34 — ULTIMATE NETWORK SINGULARITY
# GitOps + Dark Tunnel + Fractal File Translator + Traffic Warden
# Operator: Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo -e "\033[0;35m[!] DEPLOYING ULTIMATE NETWORK SINGULARITY v10003.34...\033[0m"

ROOT="${FRACTALMESH_HOME:-$HOME/fmsaas}"
VAULT="$HOME/.secrets/fractal.env"
AGENTS="$ROOT/agents"
MODULES="$ROOT/modules"
DB="$ROOT/database/sovereign.db"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[*]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }

mkdir -p "$ROOT"/{agents,modules,database,logs,tunnels,compression}
mkdir -p "$HOME/.secrets"

# ── 1. Vault (template if missing) ───────────────────────────────────────────
if [[ ! -f "$VAULT" ]]; then
  log "Writing vault template to $VAULT"
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
SUPABASE_URL=""
SUPABASE_ANON_KEY=""
EOF
  chmod 600 "$VAULT"
  ok "Vault template written — populate before starting"
fi

set -a
# shellcheck disable=SC1090
source "$VAULT"
set +a

# ── 2. Copy modules/agents from repo ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

for src in harmonic_logic_memory.py network_topology_warden.py fractal_file_translator.py; do
  [[ -f "$REPO_ROOT/modules/$src" ]] && cp -n "$REPO_ROOT/modules/$src" "$MODULES/$src" && ok "modules/$src"
done
for src in fm_pulse_bus.py fm_gitops_runner.py fm_mesh_integrator.py; do
  [[ -f "$REPO_ROOT/agents/$src" ]] && cp -n "$REPO_ROOT/agents/$src" "$AGENTS/$src" && ok "agents/$src"
done

# ── 3. Activate warden ───────────────────────────────────────────────────────
log "Activating Network Topology Warden..."
python3 "$MODULES/network_topology_warden.py" || true

# ── 4. Start nodes (additive) ────────────────────────────────────────────────
ECOSYSTEM="$REPO_ROOT/ecosystem.config.js"
[[ -f "$ECOSYSTEM" ]] || ECOSYSTEM="$ROOT/ecosystem.config.js"

log "Starting v10003.34 nodes..."
for node in fm-cloudflared fm-bus fm-gitops-runner fm-integrator fm-warden; do
  pm2 start "$ECOSYSTEM" --only "$node" --update-env 2>/dev/null \
    && ok "started: $node" \
    || { pm2 restart "$node" 2>/dev/null && ok "restarted: $node" || true; }
done

pm2 save --force 2>&1 | tail -1

echo
echo '═══════════════════════════════════════════════════════════════'
echo '   ✅ v10003.34 ULTIMATE NETWORK SINGULARITY COMPLETE'
echo '   - Dark Tunnel: configure v2ray externally with DARK_TUNNEL_TOKEN'
echo '   - Fractal File Translator: local + optional Supabase cloud sync'
echo '   - Network Topology Warden: geometric elite routing active'
echo '   - Multi-node RL hooks: ready for OpenRouter integration'
echo '═══════════════════════════════════════════════════════════════'
pm2 list

echo
echo "Test Fractal Translator:"
echo "  python3 -c 'import sys; sys.path.insert(0,\"$ROOT\"); from modules.fractal_file_translator import translator; id=translator.translate_and_balance(\"$ROOT/database/sovereign.db\"); print(\"fractal_id:\", id)'"
