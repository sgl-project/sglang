#!/usr/bin/env bash
# Cloudflare tunnel wrapper — sources vault so CF_TUNNEL_TOKEN is available,
# then execs cloudflared (exec replaces the shell so PM2 tracks the right PID).
set -euo pipefail

VAULT="${HOME}/.secrets/fractal.env"
if [[ -f "$VAULT" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$VAULT"
  set +a
fi

: "${CF_TUNNEL_TOKEN:?CF_TUNNEL_TOKEN not set — populate $VAULT}"

exec /usr/bin/cloudflared tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN"
