#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# FRACTALMESH OMEGA TITAN [v113.3] — UNIFIED SOVEREIGN SHELL
# ARM64 Termux / Proot-Debian | Samuel James Hiotis | ABN 56 628 117 363
# Usage: fm [ignite|stop|restart|status|report|sync|broadcast|monetise|logs|help]
# ==============================================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT="${FRACTALMESH_HOME:-$HOME/fmsaas}"
REPO="${REPO_ROOT:-$HOME/sglang}"
VAULT="$HOME/.secrets/fractal.env"
LOG_DIR="$ROOT/logs"
DB="$ROOT/database/sovereign.db"
ECO="$REPO/fractalmesh/ecosystem.config.js"

# ── ANSI colours ───────────────────────────────────────────────────────────────
R="\033[0;31m"; G="\033[0;32m"; Y="\033[1;33m"
C="\033[0;36m"; B="\033[1;34m"; NC="\033[0m"

log()  { echo -e "${C}[$(date +%H:%M:%S)]${NC} $1"; }
ok()   { echo -e "${G}[+]${NC} $1"; }
warn() { echo -e "${Y}[!]${NC} $1"; }
err()  { echo -e "${R}[X]${NC} $1"; exit 1; }

# ── Architecture notice ────────────────────────────────────────────────────────
[[ "$(uname -m)" == "aarch64" ]] && log "ARM64 Termux node confirmed." \
    || warn "Non-aarch64 environment — standard compatibility mode."

# ── Ensure base dirs ───────────────────────────────────────────────────────────
mkdir -p "$ROOT/database" "$LOG_DIR" "$HOME/.secrets"

# ── Dependency check ───────────────────────────────────────────────────────────
for cmd in python3 pm2; do
    command -v "$cmd" >/dev/null 2>&1 || warn "Missing: $cmd"
done

# ── Load vault ─────────────────────────────────────────────────────────────────
if [[ -f "$VAULT" ]]; then
    set -a; source "$VAULT"; set +a
    ok "Vault loaded: $VAULT"
else
    warn "Vault not found at $VAULT — live features will run in dry-run mode."
fi

# ── Guardian: hash all repo agents ────────────────────────────────────────────
_guardian() {
    log "Guardian integrity check..."
    local hash_file="$LOG_DIR/integrity.hash"
    find "$REPO/fractalmesh/agents" -type f -name "*.py" \
        -exec sha256sum {} + > "$hash_file" 2>/dev/null
    local count
    count=$(wc -l < "$hash_file")
    ok "Integrity snapshot: $count agent files — $hash_file"
}

# ── SQLite scalar helper ───────────────────────────────────────────────────────
_sq() {
    [[ -f "$DB" ]] || { echo "0"; return; }
    python3 -c "
import sqlite3, sys
try:
    c=sqlite3.connect('$DB',timeout=5)
    r=c.execute(sys.argv[1]).fetchone()
    print(r[0] if r and r[0] is not None else 0)
    c.close()
except: print(0)
" "$1" 2>/dev/null
}

# ══════════════════════════════════════════════════════════════════════════════
CMD="${1:-status}"

case "$CMD" in

# ── ignite: full swarm via ecosystem.config.js ────────────────────────────────
ignite)
    log "Igniting FractalMesh Omega Titan swarm..."
    [[ -f "$ECO" ]] || err "ecosystem.config.js not found at $ECO"
    _guardian
    pm2 start "$ECO" --env production
    pm2 save --force
    ok "Swarm LIVE — $(pm2 jlist 2>/dev/null | python3 -c \
        'import json,sys; p=json.load(sys.stdin); print(len(p))' 2>/dev/null || echo "?") agents"
    ;;

# ── stop: suspend all agents ──────────────────────────────────────────────────
stop)
    log "Suspending swarm — low-power drift mode..."
    pm2 stop all
    ok "All agents suspended."
    ;;

# ── restart: rolling restart ──────────────────────────────────────────────────
restart)
    TARGET="${2:-all}"
    log "Restarting: $TARGET"
    pm2 restart "$TARGET"
    ok "Restart complete."
    ;;

# ── status: pm2 list + grid header ────────────────────────────────────────────
status)
    pm2 list
    echo -e "\n${B}━━━ Grid Status ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  Principal : Samuel James Hiotis"
    echo -e "  ABN       : 56 628 117 363 | Sole Trader"
    echo -e "  Node      : Albury Anchor — $(hostname)"
    echo -e "  Arch      : $(uname -m)"
    echo -e "  Nexus UI  : http://localhost:8095"
    echo -e "  Vault     : $( [[ -f "$VAULT" ]] && echo "loaded" || echo "MISSING" )"
    echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    ;;

# ── report: business snapshot from sovereign.db ───────────────────────────────
report)
    echo -e "\n${B}━━━ FractalMesh Business Report ━━━━━━━━━━━━━━━━━━━${NC}"
    printf "  %-28s %s\n" "Revenue (all time):" \
        "A\$$(  _sq "SELECT COALESCE(SUM(amount_aud),0) FROM revenue")"
    printf "  %-28s %s\n" "Orders paid:" \
        "$( _sq "SELECT COUNT(*) FROM orders WHERE status='paid'")"
    printf "  %-28s %s\n" "Active products:" \
        "$( _sq "SELECT COUNT(*) FROM products WHERE active=1")"
    printf "  %-28s %s\n" "Leads:" \
        "$( _sq "SELECT COUNT(*) FROM leads")"
    printf "  %-28s %s\n" "Affiliate programs:" \
        "$( _sq "SELECT COUNT(*) FROM affiliates WHERE status='active'")"
    printf "  %-28s %s\n" "Affiliate earned:" \
        "A\$$(  _sq "SELECT COALESCE(SUM(amount),0) FROM affiliate_conversions WHERE status!='rejected'")"
    printf "  %-28s %s\n" "Affiliate clicks / 24h:" \
        "$( _sq "SELECT COUNT(*) FROM affiliate_clicks WHERE ts>datetime('now','-1 day')")"
    printf "  %-28s %s\n" "Content pieces:" \
        "$( _sq "SELECT COUNT(*) FROM content_pieces")"
    printf "  %-28s %s\n" "Drip sequences active:" \
        "$( _sq "SELECT COUNT(*) FROM drip_sequences WHERE status='active'")"
    printf "  %-28s %s\n" "Methane anomalies:" \
        "$( _sq "SELECT COUNT(*) FROM methane_readings WHERE is_anomaly=1")"
    printf "  %-28s %s\n" "AIS open alerts:" \
        "$( _sq "SELECT COUNT(*) FROM ais_alerts WHERE resolved=0")"
    printf "  %-28s %s\n" "IP portfolio:" \
        "A\$$(_sq "SELECT COALESCE(SUM(value_estimate_aud),0) FROM ip_registry")"
    echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    ;;

# ── sync: push analytics cycle + flush research cache ─────────────────────────
sync)
    log "Running research & analytics sync..."
    pm2 restart research-agent 2>/dev/null || warn "research-agent not running"
    pm2 restart affiliate-manager 2>/dev/null || warn "affiliate-manager not running"
    log "Flushing research cache (entries older than 2h)..."
    python3 - <<'PYEOF'
import sqlite3, os
db = os.path.join(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")),
                  "database", "sovereign.db")
if os.path.exists(db):
    c = sqlite3.connect(db, timeout=5)
    r = c.execute(
        "DELETE FROM research_cache WHERE ts < datetime('now','-2 hours')"
    ).rowcount
    c.commit(); c.close()
    print(f"  Pruned {r} stale cache entries.")
else:
    print("  DB not found — skipping cache flush.")
PYEOF
    ok "Sync complete."
    ;;

# ── broadcast: trigger content generation + drip cycle ───────────────────────
broadcast)
    log "Executing broadcast cycle..."
    pm2 restart content-generator 2>/dev/null || warn "content-generator not running"
    pm2 restart fm-drip-agent    2>/dev/null || warn "fm-drip-agent not running"
    log "Restarting OSINT spider for fresh leads..."
    pm2 restart fm-osint-spider  2>/dev/null || warn "fm-osint-spider not running"
    ok "Broadcast cycle triggered — content + drip + lead discovery."
    ;;

# ── monetise: restart revenue-critical agents ─────────────────────────────────
monetise)
    log "Activating revenue loops..."
    for agent in fm-negotiator billing-api affiliate-manager fm-methane-reports; do
        pm2 restart "$agent" 2>/dev/null && ok "$agent restarted" \
            || warn "$agent not in pm2 list"
    done
    ok "Revenue stack at full pressure."
    ;;

# ── logs: tail combined output ────────────────────────────────────────────────
logs)
    AGENT="${2:-}"
    if [[ -n "$AGENT" ]]; then
        pm2 logs "$AGENT" --lines 80
    else
        pm2 logs --lines 40
    fi
    ;;

# ── help ──────────────────────────────────────────────────────────────────────
help|*)
    echo -e "\n${B}FractalMesh Omega Titan [v113.3]${NC}"
    echo -e "Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader\n"
    echo -e "  ${G}ignite${NC}              Start full 49-agent swarm via ecosystem.config.js"
    echo -e "  ${G}stop${NC}                Suspend all agents (low-power mode)"
    echo -e "  ${G}restart [agent]${NC}     Restart one agent or 'all'"
    echo -e "  ${G}status${NC}              pm2 list + grid header"
    echo -e "  ${G}report${NC}              Business snapshot from sovereign.db"
    echo -e "  ${G}sync${NC}                Flush research cache, restart analytics agents"
    echo -e "  ${G}broadcast${NC}           Trigger content gen + drip + OSINT cycle"
    echo -e "  ${G}monetise${NC}            Restart revenue-critical agents"
    echo -e "  ${G}logs [agent]${NC}        Tail pm2 logs (all or specific agent)"
    echo ""
    ;;
esac

# ── Auto-alias ─────────────────────────────────────────────────────────────────
SCRIPT_PATH="$(realpath "$0")"
BASHRC="${HOME}/.bashrc"
if ! grep -q "alias fm=" "$BASHRC" 2>/dev/null; then
    echo "alias fm='bash $SCRIPT_PATH'" >> "$BASHRC"
    log "Alias 'fm' added to $BASHRC"
fi
