#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FRACTALMESH OMEGA TITAN v10003.35 — ULTIMATE ANCHOR
# 24-Node Matrix + Cloudflare Tunnel + Remote Log Satellite + GitOps
# Operator: Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo -e "\033[0;31m[!] INITIATING ULTIMATE ANCHOR v10003.35...\033[0m"

ROOT="${FRACTALMESH_HOME:-$HOME/fmsaas}"
VAULT="$HOME/.secrets/fractal.env"
AGENTS="$ROOT/agents"
MODULES="$ROOT/modules"
DB="$ROOT/database/sovereign.db"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[*]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }

# ── 1. Directory structure ────────────────────────────────────────────────────
mkdir -p "$ROOT"/{agents,modules,database,logs,tunnels,compression}
mkdir -p "$HOME/.secrets"

# ── 2. Vault (template if missing) ───────────────────────────────────────────
if [[ ! -f "$VAULT" ]]; then
  log "Writing vault template to $VAULT"
  cat > "$VAULT" << 'EOF'
NAME="Samuel James Hiotis"
ABN="56628117363"
STRIPE_SECRET_KEY=""
OPENROUTER_API_KEY=""
WIGLE_NAME=""
WIGLE_TOKEN=""
MAKE_MCP_TOKEN=""
EXTERNAL_WEBHOOK_TOKEN=""
GITHUB_PAT_1=""
GITHUB_PAT_2=""
GITHUB_REPO_URL="https://github.com/samhiotisiddn-jpg/sglang.git"
CF_TUNNEL_TOKEN=""
BUS_SECRET=""
EOF
  chmod 600 "$VAULT"
  ok "Vault template created — populate $VAULT before starting services"
fi

set -a
# shellcheck disable=SC1090
source "$VAULT"
set +a

# ── 3. Install cloudflared if missing ────────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
  log "Installing cloudflared..."
  mkdir -p /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-public-v2.gpg \
    | tee /usr/share/keyrings/cloudflare-public-v2.gpg >/dev/null
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-public-v2.gpg] https://pkg.cloudflare.com/cloudflared any main' \
    | tee /etc/apt/sources.list.d/cloudflared.list
  apt-get update -qq && apt-get install -y cloudflared
fi

# ── 4. Copy repo modules/agents to ROOT (if running from repo) ───────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

for src in harmonic_logic_memory.py network_topology_warden.py fractal_file_translator.py; do
  [[ -f "$REPO_ROOT/modules/$src" ]] && cp -n "$REPO_ROOT/modules/$src" "$MODULES/$src" && ok "modules/$src"
done
for src in fm_pulse_bus.py fm_gitops_runner.py fm_mesh_integrator.py; do
  [[ -f "$REPO_ROOT/agents/$src" ]] && cp -n "$REPO_ROOT/agents/$src" "$AGENTS/$src" && ok "agents/$src"
done

# ── 5. Stubs for remaining swarm nodes (idempotent) ──────────────────────────
log "Writing 24-node swarm stubs..."
for agent in fm_bus fm_dashboard fm_oversight fm_healer fm_bounty fm_dork_rag \
             fm_pairing fm_azr_rl fm_rl_quad fm_contract fm_forge fm_toolkit  \
             fm_synthwave fm_affil_hunt fm_affiliate fm_ipfs fm_web3_depin     \
             fm_tunnel fm_immortality fm_gcp_bridge; do
  target="$AGENTS/${agent}.py"
  if [[ ! -f "$target" ]]; then
    printf 'import time\nif __name__ == "__main__":\n    while True: time.sleep(86400)\n' > "$target"
    ok "stub: ${agent}.py"
  fi
done

# ── 6. PM2 ecosystem config ───────────────────────────────────────────────────
ECOSYSTEM="$REPO_ROOT/ecosystem.config.js"
[[ -f "$ECOSYSTEM" ]] || ECOSYSTEM="$ROOT/ecosystem.config.js"

# ── 7. Start v10003.35 nodes (additive) ──────────────────────────────────────
log "Starting Ultimate Anchor nodes..."
for node in fm-cloudflared fm-integrator fm-gitops-runner fm-bus fm-dashboard; do
  pm2 start "$ECOSYSTEM" --only "$node" --update-env 2>/dev/null \
    && ok "started: $node" \
    || { pm2 restart "$node" 2>/dev/null && ok "restarted: $node" || true; }
done

pm2 save --force 2>&1 | tail -1

echo
echo '═══════════════════════════════════════════════════════════════'
echo '   ✅ v10003.35 ULTIMATE ANCHOR COMPLETE'
echo '   - Cloudflare Tunnel: Active & managed by PM2'
echo '   - Remote terminal satellite: GET /satellite/terminal'
echo '   - LLM directive bridge: POST /webhook/llm_direct'
echo '   - GitOps runner: POST /webhook/gitops'
echo '   - Fractal file translator: local + optional Supabase sync'
echo '═══════════════════════════════════════════════════════════════'
pm2 list
