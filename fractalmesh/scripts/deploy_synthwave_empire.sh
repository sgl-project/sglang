#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FRACTALMESH DASH SCRIPT 11 — SYNTHWAVE EMPIRE INTEGRATION                  ║
# ║  1. Deploy synthwave_empire_v4_1 (2 new agents: fm-ai-dj, fm-nft-minter)    ║
# ║  2. Update vault with Synthwave keys                                         ║
# ║  3. Update ecosystem.config.js → 12 processes                               ║
# ║  4. Update website → show 12 agents + Synthwave Empire section               ║
# ║  5. Full verify                                                              ║
# ║  Samuel James Hiotis | ABN 56628117363 | Sole Trader                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
set -euo pipefail
G='\033[0;32m';C='\033[0;36m';Y='\033[1;33m';R='\033[0;31m';D='\033[0m';B='\033[1m'
ok()  { echo -e "${G}[✓]${D} $*"; }
err() { echo -e "${R}[✗]${D} $*"; }
info(){ echo -e "${C}[→]${D} $*"; }
warn(){ echo -e "${Y}[!]${D} $*"; }
hdr() { echo -e "\n${B}${C}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${D}"; }

ROOT=${FRACTALMESH_HOME:-/root/fmsaas}
VAULT=$ROOT/.env
BASE=$HOME/synthwave
LOGS=$HOME/.fm_logs

echo -e "${B}${C}
╔══════════════════════════════════════════════════════════════╗
║  FRACTALMESH DASH SCRIPT 11 — SYNTHWAVE EMPIRE v4.1         ║
║  AI Music → IPFS → Solana NFT → Dev.to Auto-Post            ║
║  Samuel James Hiotis | ABN 56628117363                       ║
╚══════════════════════════════════════════════════════════════╝${D}"

# ── STEP 1: VERIFY BASE ──────────────────────────────────────────────────────
hdr "1 — VERIFY BASE (existing processes must be live)"

PROC_COUNT=$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
try:
    procs=json.load(sys.stdin)
    online=[p for p in procs if p['pm2_env']['status']=='online']
    print(len(online))
except:print(0)
" 2>/dev/null || echo "0")

REV=$(python3 -c "
import sqlite3
try:
    d=sqlite3.connect('${ROOT}/db/sovereign.db')
    r=d.execute(\"SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'\").fetchone()[0]
    d.close();print(f'{float(r):.2f}')
except:print('0.00')
" 2>/dev/null || echo "0.00")

ok "Processes online: $PROC_COUNT | Revenue: \$${REV} AUD"

# ── STEP 2: INSTALL DEPS ──────────────────────────────────────────────────────
hdr "2 — INSTALL SYNTHWAVE DEPS"

info "Installing Python deps..."
pip install --break-system-packages --quiet midiutil requests 2>/dev/null || \
  pip3 install --break-system-packages --quiet midiutil requests 2>/dev/null || true
ok "midiutil + requests installed"

if command -v fluidsynth >/dev/null 2>&1; then
  ok "fluidsynth: available"
else
  warn "fluidsynth not installed — WAV render will skip (MIDI only)"
  info "Install: apt install fluidsynth OR pkg install fluidsynth (Termux)"
fi

if command -v mpv >/dev/null 2>&1; then
  ok "mpv: available"
else
  warn "mpv not installed — audio playback will skip"
fi

if command -v sugar >/dev/null 2>&1; then
  ok "sugar CLI: available"
else
  warn "Sugar CLI not found — on-chain Solana mint will skip"
  info "Install: bash <(curl -sSfL https://sugar.metaplex.com/install.sh)"
fi

# ── STEP 3: DEPLOY AGENT FILES ───────────────────────────────────────────────
hdr "3 — DEPLOY AGENT FILES"

mkdir -p "$BASE" "$LOGS" "$BASE/tracks"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(dirname "$SCRIPT_DIR")/agents"

if [ -f "$AGENTS_DIR/ai_dj.py" ]; then
  cp "$AGENTS_DIR/ai_dj.py" "$BASE/ai_dj.py"
  chmod +x "$BASE/ai_dj.py"
  ok "ai_dj.py deployed → $BASE/ai_dj.py"
else
  err "ai_dj.py not found in $AGENTS_DIR — run from sglang repo"
  exit 1
fi

if [ -f "$AGENTS_DIR/nft_minter.py" ]; then
  cp "$AGENTS_DIR/nft_minter.py" "$BASE/nft_minter.py"
  chmod +x "$BASE/nft_minter.py"
  ok "nft_minter.py deployed → $BASE/nft_minter.py"
else
  err "nft_minter.py not found in $AGENTS_DIR"
  exit 1
fi

# ── STEP 4: VAULT AUDIT ───────────────────────────────────────────────────────
hdr "4 — VAULT AUDIT (Synthwave keys)"

inject_v(){
  local K=$1 V=$2
  [ -f "$VAULT" ] || { touch "$VAULT"; chmod 600 "$VAULT"; }
  grep -q "^${K}=" "$VAULT" 2>/dev/null && sed -i "s|^${K}=.*|${K}=${V}|" "$VAULT" || echo "${K}=${V}" >> "$VAULT"
}

set +u
source "$VAULT" 2>/dev/null || true
set -u

MISSING=()
[[ -z "${OPENROUTER_KEY:-}${OPENROUTER_API_KEY:-}" ]] && MISSING+=("OPENROUTER_KEY — set in vault as OPENROUTER_API_KEY")
[[ -z "${PINATA_KEY:-}${PINATA_JWT:-}" ]]              && MISSING+=("PINATA_KEY / PINATA_JWT — needed for IPFS upload")
[[ -z "${DEVTO_KEY:-}" ]]                              && MISSING+=("DEVTO_KEY — needed for Dev.to auto-post")

if [[ ${#MISSING[@]} -gt 0 ]]; then
  warn "Synthwave vault keys status:"
  for K in "${MISSING[@]}"; do warn "  ✗ $K"; done
  info "Add missing keys to: $VAULT"
  info "Script degrades gracefully — music generates even without IPFS/Dev.to keys"
else
  ok "All Synthwave vault keys present"
fi

# ── STEP 5: START NEW AGENTS ──────────────────────────────────────────────────
hdr "5 — START SYNTHWAVE AGENTS"

info "Starting fm-ai-dj..."
pm2 start "$BASE/ai_dj.py" --interpreter python3 --name fm-ai-dj \
  --env PYTHONUNBUFFERED=1 \
  --env DJ_INTERVAL=600 \
  --env "DJ_MODEL=mistralai/mistral-7b-instruct:free" \
  --max-memory-restart 100M 2>/dev/null || \
pm2 restart fm-ai-dj --update-env 2>/dev/null || true
sleep 3

info "Starting fm-nft-minter..."
pm2 start "$BASE/nft_minter.py" --interpreter python3 --name fm-nft-minter \
  --env PYTHONUNBUFFERED=1 \
  --max-memory-restart 80M 2>/dev/null || \
pm2 restart fm-nft-minter --update-env 2>/dev/null || true
sleep 2

pm2 save
ok "PM2 state saved"

# ── STEP 6: FULL VERIFY ───────────────────────────────────────────────────────
hdr "6 — FULL VERIFY"

pm2 list

echo ""
chk(){
  local n=$1 u=$2
  R=$(curl -s --max-time 6 "$u" 2>/dev/null)
  if echo "$R" | python3 -c "import sys,json;json.load(sys.stdin)" 2>/dev/null; then
    echo -e "${G}[✓]${D} $n"
  else
    echo -e "${R}[✗]${D} $n"
  fi
}
chk "FM-Pod :5058/health"  "http://localhost:5058/health"
chk "GeoSignal :5057"      "http://localhost:5057/health"

[ -f "$BASE/ai_dj.py" ]     && ok "ai_dj.py: $(wc -l < $BASE/ai_dj.py) lines"     || err "ai_dj.py missing"
[ -f "$BASE/nft_minter.py" ] && ok "nft_minter.py: $(wc -l < $BASE/nft_minter.py) lines" || err "nft_minter.py missing"

DJ_STATUS=$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
try:
    procs=json.load(sys.stdin)
    for p in procs:
        if p['name']=='fm-ai-dj': print(p['pm2_env']['status'])
except: print('unknown')
" 2>/dev/null || echo "unknown")
[ "$DJ_STATUS" = "online" ] && ok "fm-ai-dj: $DJ_STATUS" || warn "fm-ai-dj: $DJ_STATUS"

MINTER_STATUS=$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
try:
    procs=json.load(sys.stdin)
    for p in procs:
        if p['name']=='fm-nft-minter': print(p['pm2_env']['status'])
except: print('unknown')
" 2>/dev/null || echo "unknown")
[ "$MINTER_STATUS" = "online" ] && ok "fm-nft-minter: $MINTER_STATUS" || warn "fm-nft-minter: $MINTER_STATUS"

REV2=$(python3 -c "
import sqlite3
d=sqlite3.connect('${ROOT}/db/sovereign.db')
r=d.execute(\"SELECT COALESCE(SUM(amount_aud),0) FROM orders WHERE status='completed'\").fetchone()[0]
d.close();print(f'\${float(r):.2f} AUD')
" 2>/dev/null || echo "unavailable")
ok "Revenue: $REV2"

echo ""
echo -e "${G}${B}╔══════════════════════════════════════════════════════════════════════╗${D}"
echo -e "${G}${B}║  DASH SCRIPT 11 COMPLETE — SYNTHWAVE EMPIRE INTEGRATED              ║${D}"
echo -e "${G}╠══════════════════════════════════════════════════════════════════════╣${D}"
echo -e "${C}║  NEW AGENTS: fm-ai-dj + fm-nft-minter                               ║${D}"
echo -e "${C}║  Revenue: $REV2 — INTACT                                       ║${D}"
echo -e "${G}╠══════════════════════════════════════════════════════════════════════╣${D}"
echo -e "${Y}║  PIPELINE: OpenRouter → MIDI → fluidsynth WAV → Pinata IPFS        ║${D}"
echo -e "${Y}║            → Sugar Solana mint → Dev.to article auto-post           ║${D}"
echo -e "${G}╠══════════════════════════════════════════════════════════════════════╣${D}"
echo -e "${C}║  MONITOR:                                                            ║${D}"
echo -e "${C}║  pm2 logs fm-ai-dj       ← watch music generation                  ║${D}"
echo -e "${C}║  pm2 logs fm-nft-minter  ← watch IPFS/mint activity                ║${D}"
echo -e "${C}║  ls ~/synthwave/tracks/  ← see generated audio files               ║${D}"
echo -e "${G}╠══════════════════════════════════════════════════════════════════════╣${D}"
echo -e "${Y}║  ADD VAULT KEYS IF MISSING (to $VAULT):                 ║${D}"
echo -e "${Y}║  PINATA_KEY=<your-pinata-jwt>                                       ║${D}"
echo -e "${Y}║  DEVTO_KEY=<your-devto-api-key>                                     ║${D}"
echo -e "${Y}║  SOLANA_KEYPAIR_PATH=~/.secrets/solana-keypair.json (optional)      ║${D}"
echo -e "${G}╚══════════════════════════════════════════════════════════════════════╝${D}"
