#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FRACTALMESH UNIFIED CONTROL  v600.0                                        ║
# ║  Principal: Samuel James Hiotis | ABN: 56628117363 | Albury NSW            ║
# ║  Strategy: 14-agent sovereign AI stack + Nexus universal gateway            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
G="\033[0;32m"; Y="\033[0;33m"; R="\033[0;31m"; C="\033[0;36m"; B="\033[1;34m"; D="\033[0m"
ok()  { echo -e "${G}✅ $*${D}"; }
inf() { echo -e "${C}ℹ  $*${D}"; }
wrn() { echo -e "${Y}⚠  $*${D}"; }
err() { echo -e "${R}❌ $*${D}"; }

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT="${FRACTALMESH_HOME:-$HOME/fmsaas}"
VAULT="$ROOT/.env"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"

# ─── Load vault into environment ──────────────────────────────────────────────
load_vault() {
    local vaults=("$HOME/.secrets/fractal.env" "$VAULT" "$HOME/.env")
    local loaded=0
    for v in "${vaults[@]}"; do
        [ -f "$v" ] || continue
        while IFS= read -r line; do
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            [[ "$line" =~ = ]] || continue
            key="${line%%=*}"; val="${line#*=}"
            [[ "$val" == YOUR_* ]] && continue
            export "$key=$val" 2>/dev/null || true
            ((loaded++)) || true
        done < "$v"
    done
    inf "Vault loaded: $loaded variables"
}

# ─── Inject/update vault key ──────────────────────────────────────────────────
inject_vault() {
    local key="$1" val="$2"
    mkdir -p "$(dirname "$VAULT")"
    touch "$VAULT"
    if grep -q "^${key}=" "$VAULT" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$VAULT"
    else
        echo "${key}=${val}" >> "$VAULT"
    fi
}

# ─── Memory check ─────────────────────────────────────────────────────────────
get_memory_mb() {
    pm2 jlist 2>/dev/null | python3 -c "
import sys,json
try:
    ps=json.load(sys.stdin)
    print(int(sum(p.get('monit',{}).get('memory',0) for p in ps)/1024/1024))
except Exception:
    print(0)
" 2>/dev/null || echo "0"
}

get_agent_count() {
    pm2 jlist 2>/dev/null | python3 -c "
import sys,json
try: print(len([p for p in json.load(sys.stdin) if p.get('pm2_env',{}).get('status')=='online']))
except: print(0)" 2>/dev/null || echo "0"
}

# ─── System summary ───────────────────────────────────────────────────────────
system_summary() {
    echo ""
    echo -e "${B}╔═══════════════════════════════════════════════════════╗${D}"
    echo -e "${B}║  FRACTALMESH NEXUS v600  |  Samuel James Hiotis       ║${D}"
    echo -e "${B}║  ABN 56628117363 | Albury NSW 2640                    ║${D}"
    echo -e "${B}╚═══════════════════════════════════════════════════════╝${D}"
    echo ""

    local agents; agents=$(get_agent_count)
    local mem;    mem=$(get_memory_mb)
    echo -e "  Agents online : ${G}${agents}/16${D}  |  Memory : ${Y}${mem}MB / 250MB${D}"
    echo ""

    # Health checks
    echo -e "${C}📡 ENDPOINT HEALTH:${D}"
    local endpoints=(
        "localhost:5057/health:fm-geosignal"
        "localhost:5058/health:fm-pod"
        "localhost:5060/health:fm-analytics"
        "localhost:5061/health:fm-notes"
        "localhost:5062/health:fm-tunnel"
        "localhost:8000/health:nexus-gateway"
        "localhost:8090:fm-dashboard"
    )
    for ep in "${endpoints[@]}"; do
        local url="${ep%%:*:*}:${ep#*:}"; url="${url%%:*}"
        local name="${ep##*:}"
        local addr="${ep%:*}"
        if curl -sf --max-time 2 "http://${addr}" >/dev/null 2>&1; then
            echo -e "  ${G}✅${D} http://${addr}  (${name})"
        else
            echo -e "  ${R}❌${D} http://${addr}  (${name})"
        fi
    done

    # Revenue snapshot
    echo ""
    echo -e "${C}💰 REVENUE SNAPSHOT:${D}"
    local rev; rev=$(curl -sf --max-time 3 "http://localhost:5058/api/revenue/mrr" 2>/dev/null || echo '{}')
    local mrr; mrr=$(echo "$rev" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('mrr_fmt','N/A'))" 2>/dev/null || echo "N/A")
    local arr; arr=$(echo "$rev" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('arr_fmt','N/A'))" 2>/dev/null || echo "N/A")
    local subs; subs=$(echo "$rev" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_subs','N/A'))" 2>/dev/null || echo "N/A")
    echo -e "  MRR: ${G}${mrr}${D}  |  ARR: ${G}${arr}${D}  |  Active subs: ${G}${subs}${D}"

    # Tunnel URL
    local tunnel; tunnel=$(curl -sf --max-time 2 "http://localhost:5062/api/tunnel/url" 2>/dev/null || echo '{}')
    local pub_url; pub_url=$(echo "$tunnel" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url','none'))" 2>/dev/null || echo "none")
    if [ "$pub_url" != "none" ] && [ -n "$pub_url" ]; then
        echo ""
        echo -e "${C}🌐 PUBLIC URL:${D} ${G}${pub_url}${D}"
    fi

    echo ""
    pm2 list 2>/dev/null || true
    echo ""
}

# ─── date-env: reload vault + PM2 restart ─────────────────────────────────────
date_env() {
    inf "Reloading vault and restarting all agents with updated environment..."
    load_vault
    # Export all vault vars to PM2 environment file
    local env_file="$ROOT/.pm2_env"
    cp "$VAULT" "$env_file" 2>/dev/null || true
    pm2 restart all --update-env 2>/dev/null || true
    pm2 save 2>/dev/null || true
    ok "Environment reloaded | All agents restarted with fresh env"
}

# ─── start: bring up all PM2 processes ────────────────────────────────────────
start_all() {
    load_vault
    mkdir -p "$LOG_DIR" "$ROOT/db" "$ROOT/www" "$ROOT/agents"
    # Copy agents from repo if needed
    if [ -d "$REPO_DIR/agents" ]; then
        cp -n "$REPO_DIR/agents/"*.py "$ROOT/agents/" 2>/dev/null || true
        ok "Agents synced from $REPO_DIR/agents/"
    fi
    # Copy ecosystem config
    [ -f "$REPO_DIR/ecosystem.config.js" ] && cp "$REPO_DIR/ecosystem.config.js" "$ROOT/"
    # Copy www
    [ -d "$REPO_DIR/www" ] && cp -r "$REPO_DIR/www/." "$ROOT/www/"
    # Start
    pm2 start "$ROOT/ecosystem.config.js" --env production 2>/dev/null || \
    pm2 restart all --update-env 2>/dev/null || true
    pm2 save 2>/dev/null || true
    ok "All agents started via PM2"
}

# ─── merge: install Nexus gateway ─────────────────────────────────────────────
deploy_nexus() {
    inf "Deploying Nexus universal gateway..."
    mkdir -p "$ROOT/agents"
    if [ -f "$REPO_DIR/agents/gateway.py" ]; then
        cp "$REPO_DIR/agents/gateway.py" "$ROOT/agents/"
        ok "gateway.py deployed"
    fi
    # Ensure gateway in PM2
    if ! pm2 list | grep -q "unified-gateway"; then
        pm2 start "$ROOT/agents/gateway.py" \
            --name "unified-gateway" \
            --interpreter python3 \
            --max-memory-restart 16M \
            -- --port 8000 2>/dev/null || true
    else
        pm2 restart unified-gateway 2>/dev/null || true
    fi
    pm2 save 2>/dev/null || true
    ok "Nexus gateway deployed on :8000"
}

# ─── optimize: memory safety ──────────────────────────────────────────────────
optimize() {
    inf "Running memory optimization..."
    local current; current=$(get_memory_mb)
    echo -e "  Current memory: ${Y}${current}MB${D} / 250MB"
    if [ "$current" -gt 220 ]; then
        wrn "High memory — applying QBM limits..."
        pm2 restart fm-dashboard --max-memory-restart 25M 2>/dev/null || true
        pm2 restart fm-pod       --max-memory-restart 25M 2>/dev/null || true
        pm2 restart fm-geosignal --max-memory-restart 25M 2>/dev/null || true
        pm2 restart fm-dorking   --max-memory-restart 20M 2>/dev/null || true
        pm2 restart fm-figma     --max-memory-restart 20M 2>/dev/null || true
        pm2 save 2>/dev/null || true
        sleep 3
        local new; new=$(get_memory_mb)
        ok "Memory optimized: ${current}MB → ${new}MB"
    else
        ok "Memory within budget: ${current}MB"
    fi
}

# ─── test-ai: invoke Nexus AI generation ──────────────────────────────────────
test_ai() {
    inf "Testing Nexus AI generation..."
    if curl -sf --max-time 3 "http://localhost:8000/health" >/dev/null 2>&1; then
        curl -s -X POST http://localhost:8000/ai/generate \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "FractalMesh system status: 14 agents, MRR $217/mo. Provide 3-sentence analysis.",
                "max_tokens": 150
            }' | python3 -m json.tool 2>/dev/null || echo "Response received"
    else
        err "Nexus gateway offline. Run: $0 merge"
    fi
}

# ─── setup-alias: install fm command ──────────────────────────────────────────
setup_alias() {
    local script_path; script_path="$(realpath "${BASH_SOURCE[0]}")"
    local alias_line="alias fm='bash ${script_path}'"
    for rcfile in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile"; do
        [ -f "$rcfile" ] || continue
        if ! grep -q "alias fm=" "$rcfile" 2>/dev/null; then
            echo "" >> "$rcfile"
            echo "# FractalMesh unified control" >> "$rcfile"
            echo "$alias_line" >> "$rcfile"
            ok "Alias added to $rcfile"
        fi
    done
    inf "Run: source ~/.bashrc  (then use 'fm' from anywhere)"
}

# ─── full: complete initialization ────────────────────────────────────────────
full() {
    inf "Full FractalMesh initialization..."
    load_vault
    start_all
    deploy_nexus
    optimize
    ok "Full initialization complete"
}

# ─── help ─────────────────────────────────────────────────────────────────────
show_help() {
    echo -e "${G}FractalMesh Unified Control v600.0${D}"
    echo -e "${Y}Principal: Samuel James Hiotis | ABN 56628117363${D}"
    echo ""
    echo -e "${C}COMMANDS:${D}"
    echo "  fm status          # System status + health checks + revenue"
    echo "  fm date-env        # Reload vault + restart all PM2 with updated env"
    echo "  fm start           # Sync agents from repo + pm2 start ecosystem"
    echo "  fm merge           # Install/update Nexus universal gateway (:8000)"
    echo "  fm full            # Complete init: env + start + nexus + optimize"
    echo "  fm optimize        # Memory optimization (QBM limits)"
    echo "  fm test-ai         # Test AI generation via Nexus gateway"
    echo "  fm alias           # Install 'fm' shortcut to ~/.bashrc"
    echo ""
    echo -e "${C}AGENT PORTS:${D}"
    echo "  :5057  fm-geosignal       (live signals + NASA)"
    echo "  :5058  fm-pod             (master API + Stripe)"
    echo "  :5060  fm-analytics       (MRR/ARR/LTV)"
    echo "  :5061  fm-notes           (Firebase IP registrar)"
    echo "  :5062  fm-tunnel          (public URL)"
    echo "  :8000  unified-gateway    (Nexus AI proxy)"
    echo "  :8090  fm-dashboard       (web UI)"
    echo ""
    echo -e "${C}ZERO-CAPITAL MONETIZATION:${D}"
    echo "  • Fractal Signal Feed   $499 AUD | $49/mo"
    echo "  • Sovereign Dashboard   $299 AUD | $29/mo"
    echo "  • Geo-Intelligence Feed $349 AUD | $35/mo  (WiGLE+NASA)"
    echo "  • Synthwave Empire NFTs $149 AUD | $15/mo  (7% royalties)"
    echo "  • Enterprise Bundle     $899 AUD | $89/mo"
    echo "  • Coupons: ALBURY20, LAUNCH50, FRACTAL10, NEXUS25"
}

# ─── Router ───────────────────────────────────────────────────────────────────
case "${1:-status}" in
    status|"")    system_summary ;;
    date-env|env) date_env;  system_summary ;;
    start|up)     start_all; system_summary ;;
    merge|nexus)  deploy_nexus; system_summary ;;
    full|complete)full ;;
    optimize)     optimize; system_summary ;;
    test-ai|ai)   test_ai ;;
    alias)        setup_alias ;;
    help|--help|-h) show_help ;;
    *)
        err "Unknown command: $1"
        echo "Run '$0 help' for available commands"
        exit 1
        ;;
esac
