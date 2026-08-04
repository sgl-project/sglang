#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FRACTALMESH OMEGA TITAN v40.0 — FULL DEPLOY (VAULT-SAFE)
# Termux/Linux compatible | Zero hardcoded credentials
# Operator: Samuel James Hiotis | ABN: 56 628 117 363
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

readonly R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' C='\033[0;36m' NC='\033[0m'

# ── Detect environment ────────────────────────────────────────────────────────
if [[ "${PREFIX:-}" == *"termux"* ]] || [[ -d "/data/data/com.termux" ]]; then
    ENV_TYPE="termux"
    HOME_BASE="$HOME"
    PYTHON="python3"
    PIP="pip3 install --break-system-packages"
    PM2="pm2"
elif command -v apt-get &>/dev/null; then
    ENV_TYPE="debian"
    HOME_BASE="$HOME"
    PYTHON="python3"
    PIP="pip3 install"
    PM2="pm2"
else
    ENV_TYPE="generic"
    HOME_BASE="$HOME"
    PYTHON="python3"
    PIP="pip3 install"
    PM2="pm2"
fi

ROOT="${FRACTALMESH_HOME:-$HOME_BASE/fmsaas}"
REPO="${REPO_ROOT:-$HOME_BASE/sglang}"
VAULT="$HOME_BASE/.secrets/fractal.env"
AGENTS="$REPO/fractalmesh/agents"
API="$REPO/fractalmesh/api"
SCRIPTS="$REPO/fractalmesh/scripts"

echo -e "${C}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${C}║  FRACTALMESH OMEGA TITAN v40.0 — DEPLOY                  ║${NC}"
echo -e "${C}║  Operator: Samuel James Hiotis | ABN 56 628 117 363       ║${NC}"
echo -e "${C}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${Y}[0] Environment: $ENV_TYPE | Root: $ROOT${NC}"

# ── Create directories ────────────────────────────────────────────────────────
echo -e "${C}[1] Creating directory structure...${NC}"
mkdir -p "$ROOT"/{database,dist,backups,logs}
mkdir -p "$(dirname "$VAULT")"

# ── Vault template (credentials go here manually — NEVER hardcode) ────────────
echo -e "${C}[2] Checking vault...${NC}"
if [[ ! -f "$VAULT" ]]; then
    echo -e "${Y}[VAULT] Creating vault template at $VAULT${NC}"
    cat > "$VAULT" << 'VAULTEOF'
# FractalMesh Sovereign Vault — Samuel James Hiotis | ABN 56 628 117 363
# SECURITY: chmod 600 this file. Never commit. Never paste into chat.
# Fill values after rotating all credentials.

# ── Core identity ──────────────────────────────────────────────────────────
OPERATOR_NAME="Samuel James Hiotis"
OPERATOR_EMAIL=""
OPERATOR_PHONE=""
ABN="56628117363"
# TFN: store in ATO secure system only — not here

# ── Internal bus ──────────────────────────────────────────────────────────
BUS_SECRET=""
DASHBOARD_TOKEN=""

# ── Email ─────────────────────────────────────────────────────────────────
GMAIL_USER=""
GMAIL_APP_PASS=""

# ── Payments ──────────────────────────────────────────────────────────────
STRIPE_SECRET_KEY=""
STRIPE_PUBLISHABLE_KEY=""
STRIPE_WEBHOOK_SECRET=""
STRIPE_PAYMENT_LINK=""

# ── Trading (rotate all keys that appeared in plaintext) ──────────────────
BLOFIN_API_KEY=""
BLOFIN_API_SECRET=""
KUCOIN_API_KEY=""
KUCOIN_API_SECRET=""
KUCOIN_API_PASSPHRASE=""
PIONEX_API_KEY=""
PIONEX_API_SECRET=""

# ── AI/LLM ────────────────────────────────────────────────────────────────
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
XAI_API_KEY=""

# ── Infra ─────────────────────────────────────────────────────────────────
CF_TUNNEL_TOKEN=""
EXTERNAL_WEBHOOK_TOKEN=""
SUPABASE_URL=""
SUPABASE_ANON_KEY=""

# ── OSINT ─────────────────────────────────────────────────────────────────
GOOGLE_CSE_API_KEY=""
GOOGLE_CSE_ID=""
WIGLE_API_NAME=""
WIGLE_API_TOKEN=""

# ── Open source bridges ───────────────────────────────────────────────────
OLLAMA_HOST="http://localhost:11434"
BTCPAY_ENABLED="false"

# ── Agent feature flags (set to "true" to enable live mode) ──────────────
ENABLE_AFFILIATE_LIVE="false"
ENABLE_AUTO_ADVERT="false"
ENABLE_CONTRACT_FORGE="false"
ENABLE_DORK_ENGINE="false"
ENABLE_OSINT_SPIDER="false"
ENABLE_IPFS_PIN="false"
ENABLE_NEGOTIATOR="false"
ENABLE_HEALER_RESTARTS="false"
ENABLE_WIGLE_QUERY="false"
ENABLE_STRIPE_GATEWAY="false"
ENABLE_MICROCHARGE="false"
ENABLE_DEVTO_PUBLISH="false"
ENABLE_WORKSPACE_SYNC="false"
VAULTEOF
    chmod 600 "$VAULT"
    echo -e "${G}[VAULT] Template created. Fill in values before starting agents.${NC}"
else
    echo -e "${G}[VAULT] Vault exists at $VAULT${NC}"
    # Enforce permissions
    chmod 600 "$VAULT"
fi

# ── Load vault into environment ───────────────────────────────────────────────
echo -e "${C}[3] Loading vault...${NC}"
set +u
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    [[ "$line" =~ ^([A-Z_]+)=(.*)$ ]] && export "${BASH_REMATCH[1]}"="${BASH_REMATCH[2]//\"/}"
done < "$VAULT"
set -u

# ── Fix Termux package issues ─────────────────────────────────────────────────
if [[ "$ENV_TYPE" == "termux" ]]; then
    echo -e "${C}[4] Termux dependency fixes...${NC}"
    # Fix broken dpkg state without upgrading binutils (which fails on some Android)
    apt-get --fix-broken install -y 2>/dev/null || true
    pkg install -y python git sqlite jq nodejs-lts openssl-tool 2>/dev/null || true
    # Rust needed for cryptography/ormsgpack
    pkg install -y rust 2>/dev/null || true
    echo -e "${G}    Termux fixes applied${NC}"
fi

# ── Python dependencies ───────────────────────────────────────────────────────
echo -e "${C}[5] Installing Python dependencies...${NC}"
$PIP flask requests psutil sqlalchemy python-dotenv 2>/dev/null || true
# ormsgpack/langgraph requires Rust — skip on ARM if unavailable
$PIP mcp fastmcp langchain 2>/dev/null || echo -e "${Y}    fastmcp/mcp skipped (build deps missing)${NC}"

# ── Sovereign DB initialisation ───────────────────────────────────────────────
echo -e "${C}[6] Initialising sovereign.db...${NC}"
$PYTHON - << 'PYEOF'
import os, sqlite3
root = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
db   = os.path.join(root, "database", "sovereign.db")
os.makedirs(os.path.dirname(db), exist_ok=True)
conn = sqlite3.connect(db)
conn.execute("PRAGMA journal_mode=WAL")
for stmt in [
    "CREATE TABLE IF NOT EXISTS pulse_log (id INTEGER PRIMARY KEY, source TEXT, event TEXT, priority REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE IF NOT EXISTS leads (id INTEGER PRIMARY KEY, industry TEXT, region TEXT, query_hash TEXT UNIQUE, raw_query TEXT, result_count INTEGER, source TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE IF NOT EXISTS revenue (id INTEGER PRIMARY KEY, source TEXT, charge_id TEXT UNIQUE, amount_aud REAL, currency TEXT, description TEXT, status TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE IF NOT EXISTS royalty_pools (pool_id TEXT PRIMARY KEY, label TEXT, pct REAL, aud_balance REAL, updated DATETIME DEFAULT CURRENT_TIMESTAMP)",
]:
    conn.execute(stmt)
conn.commit(); conn.close()
print(f"[DB] sovereign.db ready at {db}")
PYEOF

# ── PM2 ecosystem.config.js ───────────────────────────────────────────────────
echo -e "${C}[7] Writing PM2 ecosystem...${NC}"
cat > "$ROOT/ecosystem.config.js" << ECOSEOF
module.exports = {
  apps: [
    // ── Core infrastructure ──────────────────────────────────────────────
    { name: "fm-pulse-bus",       script: "$AGENTS/fm_pulse_bus.py",       interpreter: "python3", max_memory_restart: "120M" },
    { name: "fm-mesh-integrator", script: "$AGENTS/fm_mesh_integrator.py", interpreter: "python3", max_memory_restart: "120M" },
    { name: "fm-health-api",      script: "$API/health_app.py",            interpreter: "python3", env: { APP_PORT: "5057" }, max_memory_restart: "80M" },
    { name: "omni-dashboard",     script: "$API/omni_dashboard.py",        interpreter: "python3", env: { APP_PORT: "8090" }, max_memory_restart: "150M" },
    // ── Compliance & identity ────────────────────────────────────────────
    { name: "fm-sovereign-ops",   script: "$AGENTS/fm_sovereign_ops.py",   interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-lba-bridge",      script: "$AGENTS/fm_lba_bridge.py",      interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-oversight",       script: "$AGENTS/fm_oversight.py",       interpreter: "python3", max_memory_restart: "80M" },
    // ── Self-healing ─────────────────────────────────────────────────────
    { name: "fm-healer",          script: "$AGENTS/fm_healer.py",          interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-immortality",     script: "$AGENTS/fm_immortality.py",     interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-salvage-crew",    script: "$AGENTS/fm_salvage_crew.py",    interpreter: "python3", max_memory_restart: "80M" },
    // ── Revenue ──────────────────────────────────────────────────────────
    { name: "fm-stripe-mon",      script: "$AGENTS/fm_stripe_mon.py",      interpreter: "python3", env: { APP_PORT: "8091" }, max_memory_restart: "100M" },
    { name: "fm-stripe-gateway",  script: "$AGENTS/fm_stripe_gateway.py",  interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-live-tokenomics", script: "$AGENTS/fm_live_tokenomics.py", interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-micro-charge",    script: "$AGENTS/economics/fm_micro_charge.py", interpreter: "python3", max_memory_restart: "80M" },
    // ── Marketing & outreach ─────────────────────────────────────────────
    { name: "fm-negotiator",      script: "$AGENTS/fm_negotiator.py",      interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-auto-advert",     script: "$AGENTS/fm_auto_advert.py",     interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-affiliate",       script: "$AGENTS/fm_affiliate.py",       interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-devto",           script: "$AGENTS/marketing/fm_devto.py", interpreter: "python3", max_memory_restart: "80M" },
    // ── Intelligence ─────────────────────────────────────────────────────
    { name: "fm-osint-spider",    script: "$AGENTS/fm_osint_spider.py",    interpreter: "python3", max_memory_restart: "100M" },
    { name: "fm-dork-engine",     script: "$AGENTS/fm_dork_engine.py",     interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-wigle-oracle",    script: "$AGENTS/fm_wigle_oracle.py",    interpreter: "python3", max_memory_restart: "80M" },
    // ── System ───────────────────────────────────────────────────────────
    { name: "fm-toolkit",         script: "$AGENTS/fm_toolkit.py",         interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-email-listener",  script: "$AGENTS/fm_email_listener.py",  interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-samsung-warden",  script: "$AGENTS/fm_samsung_warden.py",  interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-domain",          script: "$AGENTS/fm_domain.py",          interpreter: "python3", max_memory_restart: "80M" },
    { name: "fm-enochian-hash",   script: "$AGENTS/fm_enochian_hash.py",   interpreter: "python3", max_memory_restart: "80M" },
    { name: "git-oracle",         script: "$AGENTS/system/git_oracle.py",  interpreter: "python3", max_memory_restart: "80M" },
    { name: "omni-graph",         script: "$AGENTS/system/omni_graph.py",  interpreter: "python3", max_memory_restart: "80M" },
  ]
};
ECOSEOF

echo -e "${G}[7] ecosystem.config.js written to $ROOT/${NC}"

# ── Launch ────────────────────────────────────────────────────────────────────
echo -e "${C}[8] Launching PM2 swarm...${NC}"
cd "$ROOT"

# Start only health + dashboard + core first; others can be added selectively
$PM2 start ecosystem.config.js --only "fm-pulse-bus,omni-dashboard,fm-health-api,fm-sovereign-ops,fm-healer,fm-salvage-crew" 2>/dev/null || \
$PM2 restart ecosystem.config.js --only "fm-pulse-bus,omni-dashboard,fm-health-api,fm-sovereign-ops,fm-healer,fm-salvage-crew" 2>/dev/null || true
$PM2 save 2>/dev/null || true

echo ""
echo -e "${G}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${G}║  FRACTALMESH OMEGA TITAN v40.0 DEPLOYED                  ║${NC}"
echo -e "${G}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${G}║  Omni-Dashboard:  http://127.0.0.1:8090                  ║${NC}"
echo -e "${G}║  Health API:      http://127.0.0.1:5057/health            ║${NC}"
echo -e "${G}║  Pulse Bus:       http://127.0.0.1:5060/health            ║${NC}"
echo -e "${G}║  Logs:            pm2 logs                                ║${NC}"
echo -e "${G}║  Full swarm:      pm2 start ecosystem.config.js           ║${NC}"
echo -e "${G}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${Y}║  ACTION: Fill vault at ~/.secrets/fractal.env             ║${NC}"
echo -e "${Y}║  ACTION: Rotate ALL credentials that appeared in chat     ║${NC}"
echo -e "${Y}║  ACTION: XYO seed phrase — move funds to new wallet NOW   ║${NC}"
echo -e "${G}╚══════════════════════════════════════════════════════════╝${NC}"
