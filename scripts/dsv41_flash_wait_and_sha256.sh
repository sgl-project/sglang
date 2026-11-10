#!/usr/bin/env bash
# Wait for Engram shards 47/48, place them, then openssl sha256 vs official LFS oids.
set -euo pipefail

MODEL="${MODEL_PATH:-$HOME/models/DeepSeek-V4.1-Flash}"
SUMS="${SUMS:-$MODEL/SHA256SUMS}"
LOG="${LOG:-$MODEL/SHA256SUMS.openssl.log}"
GOT="${GOT:-$MODEL/SHA256SUMS.got}"
DL="$MODEL/.cache/huggingface/download"

INC47="$DL/95KtgLlFZaOvegAH2UrU43IvGLI=.824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f.fb9cf4cf.incomplete"
INC48="$DL/EjlT4LZa81dCKY4mCC22O7kqCQM=.976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed.220e02a5.incomplete"
WANT47=101535150936
WANT48=101537926640
DEST47="$MODEL/model-00047-of-00048.safetensors"
DEST48="$MODEL/model-00048-of-00048.safetensors"
CURL47="${CURL47_PID:-11557}"
CURL48="${CURL48_PID:-11558}"

exec > >(tee -a "$LOG") 2>&1

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
say() { echo "[$(ts)] $*"; }

wait_pid() {
  local pid="$1" label="$2"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    say "$label pid ${pid:-none} not running"
    return 0
  fi
  say "waiting for $label pid $pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
  say "$label pid $pid exited"
}

place() {
  local inc="$1" dest="$2" want="$3" name="$4"
  if [[ -f "$dest" ]]; then
    local have
    have=$(stat -c '%s' "$dest")
    if [[ "$have" == "$want" ]]; then
      say "$name already in place ($have bytes)"
      return 0
    fi
    say "ERROR $name exists with size $have want $want"
    return 1
  fi
  if [[ ! -f "$inc" ]]; then
    say "ERROR missing incomplete for $name: $inc"
    return 1
  fi
  local have
  have=$(stat -c '%s' "$inc")
  if [[ "$have" != "$want" ]]; then
    say "ERROR $name incomplete size $have want $want — not moving"
    return 1
  fi
  mv -n -- "$inc" "$dest"
  say "placed $name ($have bytes)"
}

say "wait+openssl start"

wait_pid "$CURL47" shard47
wait_pid "$CURL48" shard48
# If the recorded PIDs were recycled/gone, still wait until any matching curl dies.
while pgrep -f 'curl .*model-0004[78]-of-00048.safetensors' >/dev/null; do
  say "curl still running for 47/48"
  sleep 20
done
say "no curl left for 47/48"

place "$INC47" "$DEST47" "$WANT47" model-00047-of-00048.safetensors
place "$INC48" "$DEST48" "$WANT48" model-00048-of-00048.safetensors

if [[ ! -f "$SUMS" ]]; then
  say "ERROR missing $SUMS"
  exit 2
fi

say "openssl sha256 vs $SUMS"
: > "$GOT"
fail=0
missing=0
ok=0
while read -r want name; do
  [[ -z "${want:-}" || "$want" == \#* ]] && continue
  if [[ ! -f "$MODEL/$name" ]]; then
    say "MISSING  $name"
    missing=$((missing + 1))
    continue
  fi
  have=$(stat -c '%s' "$MODEL/$name")
  say "HASHING  $name ($have bytes)"
  # openssl sha256 prints: SHA256(path)= <hex>
  got=$(ionice -c3 openssl sha256 -- "$MODEL/$name" | awk -F'= ' '{print $NF}')
  echo "$got  $name" >> "$GOT"
  if [[ "$got" != "$want" ]]; then
    say "MISMATCH $name"
    say "  want $want"
    say "  got  $got"
    fail=$((fail + 1))
  else
    say "OK       $name"
    ok=$((ok + 1))
  fi
done < "$SUMS"

say "RESULT ok=$ok missing=$missing hash_fail=$fail"
if (( fail > 0 || missing > 0 )); then
  exit 1
fi
exit 0
