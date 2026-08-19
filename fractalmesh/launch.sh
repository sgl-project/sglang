#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FRACTALMESH LIVE LAUNCH                                                     ║
# ║  One-command deploy — Termux/Linux                                           ║
# ║  Samuel James Hiotis | ABN 56628117363                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; R='\033[0;31m'
D='\033[0m';    B='\033[1m'
ok()  { echo -e "${G}[✓]${D} $*"; }
err() { echo -e "${R}[✗]${D} $*"; }
info(){ echo -e "${C}[→]${D} $*"; }
warn(){ echo -e "${Y}[!]${D} $*"; }
hdr() { echo -e "\n${B}${C}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${D}"; }

# ── Resolve root ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FRACTALMESH_HOME="${FRACTALMESH_HOME:-$SCRIPT_DIR}"
VAULT="$FRACTALMESH_HOME/.env"

echo -e "${B}${C}
╔══════════════════════════════════════════════════════════════╗
║  FRACTALMESH OMEGA — LIVE LAUNCH                              ║
║  Samuel James Hiotis | ABN 56628117363 | Albury NSW           ║
╚══════════════════════════════════════════════════════════════╝${D}"

hdr "1 — DEPENDENCIES"

# Python packages
info "Installing Python dependencies..."
pip3 install --quiet --break-system-packages \
    flask flask-cors python-dotenv requests \
    2>/dev/null || \
pip3 install --quiet \
    flask flask-cors python-dotenv requests \
    2>/dev/null || warn "pip had warnings — continuing"
ok "Python packages ready"

# Node serve (for dashboard)
if ! command -v serve &>/dev/null; then
    info "Installing serve..."
    npm install -g serve 2>/dev/null || warn "serve install failed — dashboard needs manual start"
fi
command -v serve &>/dev/null && ok "serve available" || warn "serve not found"

# PM2
if ! command -v pm2 &>/dev/null; then
    info "Installing PM2..."
    npm install -g pm2 2>/dev/null || {
        err "PM2 not available — install with: npm install -g pm2"
        exit 1
    }
fi
ok "PM2: $(pm2 --version)"

hdr "2 — VAULT"

if [ ! -f "$VAULT" ]; then
    info "Creating vault from template..."
    cat > "$VAULT" << 'ENVEOF'
# FractalMesh Vault — fill in real credentials
FLASK_PORT=5058
GEOSIGNAL_PORT=5057
DASH_PORT=8090

# AI (required for live chat)
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY_HERE

# Stripe (required for live checkout)
STRIPE_SECRET_KEY=sk_live_YOUR_STRIPE_SECRET_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_STRIPE_PUBLISHABLE_KEY_HERE
STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET_HERE
STRIPE_LINK_SIGNAL=https://buy.stripe.com/YOUR_SIGNAL_LINK
STRIPE_LINK_DASH=https://buy.stripe.com/YOUR_DASH_LINK
STRIPE_LINK_NFT=https://buy.stripe.com/YOUR_NFT_LINK
STRIPE_LINK_ENT=https://buy.stripe.com/YOUR_ENT_LINK

# Exchanges
KUCOIN_API_KEY=YOUR_KUCOIN_API_KEY_HERE
KUCOIN_API_SECRET=YOUR_KUCOIN_API_SECRET_HERE
KUCOIN_API_PASSPHRASE=YOUR_KUCOIN_PASSPHRASE_HERE
PIONEX_API_KEY=YOUR_PIONEX_API_KEY_HERE
PIONEX_API_SECRET=YOUR_PIONEX_API_SECRET_HERE

# Publishing
DEVTO_API_KEY=YOUR_DEVTO_API_KEY_HERE
ZENODO_TOKEN=YOUR_ZENODO_TOKEN_HERE

# AI Studio
AI_STUDIO_APP_URL=https://ai.studio/apps/bafddcde-c79c-4e7b-931e-d4d218e325de
AI_STUDIO_PRE_URL=https://ais-pre-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app
AI_STUDIO_DEV_URL=https://ais-dev-cfm5pqyfzngks2vm33hhfd-89456771264.asia-southeast1.run.app
ENVEOF
    chmod 600 "$VAULT"
    ok "Vault created at $VAULT"
    warn "Fill in credentials then re-run this script"
else
    ok "Vault exists"
fi

hdr "3 — LOGS"
mkdir -p "$FRACTALMESH_HOME/logs"
ok "Logs dir: $FRACTALMESH_HOME/logs"

hdr "4 — PM2 LAUNCH"

# Kill stale pm2 processes on our ports (non-fatal)
pm2 delete fm-pod fm-geosignal fm-trading fm-whitepaper fm-dashboard 2>/dev/null || true

info "Starting ecosystem..."
cd "$FRACTALMESH_HOME"
pm2 start ecosystem.config.js --env production
sleep 4

hdr "5 — VERIFICATION"

pm2 list

echo ""
chk() {
    local name=$1 url=$2
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 6 "$url" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo -e "${G}[✓]${D} $name → HTTP $STATUS"
    elif [ "$STATUS" = "000" ]; then
        echo -e "${R}[✗]${D} $name → not responding (may still be starting)"
    else
        echo -e "${Y}[!]${D} $name → HTTP $STATUS"
    fi
}

chk "fm-geosignal :5057" "http://localhost:5057/health"
chk "fm-pod       :5058" "http://localhost:5058/health"
chk "fm-dashboard :8090" "http://localhost:8090"

echo ""
echo -e "${G}${B}╔══════════════════════════════════════════════════════════════════════╗${D}"
echo -e "${G}${B}║  FRACTALMESH LIVE                                                     ║${D}"
echo -e "${G}╠══════════════════════════════════════════════════════════════════════╣${D}"
echo -e "${C}║  Dashboard →   http://localhost:8090                                 ║${D}"
echo -e "${C}║  API       →   http://localhost:5058/api/status                     ║${D}"
echo -e "${C}║  Signals   →   http://localhost:5057/api/signals                    ║${D}"
echo -e "${C}║  AI Studio →   http://localhost:5058/api/ai-studio                  ║${D}"
echo -e "${G}╠══════════════════════════════════════════════════════════════════════╣${D}"
echo -e "${Y}║  Logs:  pm2 logs                                                     ║${D}"
echo -e "${Y}║  Status: pm2 list                                                    ║${D}"
echo -e "${Y}║  Stop:   pm2 stop all                                                ║${D}"
echo -e "${G}╚══════════════════════════════════════════════════════════════════════╝${D}"
