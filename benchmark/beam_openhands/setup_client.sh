#!/usr/bin/env bash
# Isolated client on this host: a venv holding exactly client-requirements.txt (evalscope pinned by commit), validated
# after install, plus a tokenizer-only view of the served checkpoint. Writes only under WORK_DIR; the serving
# interpreter and the server are never touched.
#
#   SITE_ENV=/path/to/site.env bash setup_client.sh
#
# An existing venv that passes the pin check is kept. One that fails is refused; FORCE=1 moves it aside (never deletes).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

mkdir -p "$WORK_DIR/logs"
say "client setup: work dir $WORK_DIR, base interpreter $CLIENT_BASE_PYTHON ($(uname -m))"
"$CLIENT_BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || die "$CLIENT_BASE_PYTHON is not CPython 3.12 (the pinned set was frozen on 3.12)"
case "$("$CLIENT_BASE_PYTHON" -c 'import sys; print(sys.prefix)')" in
    "$VENV") die "CLIENT_BASE_PYTHON points into the client venv itself" ;;
esac

check() { "$PY" "$HERE/sweep_tool.py" client-check --requirements "$HERE/client-requirements.txt" --out "$CLIENT_MANIFEST"; }

if [ -e "$VENV" ] && ! check; then
    [ "${FORCE:-0}" = 1 ] || die "$VENV does not match client-requirements.txt; rerun with FORCE=1 to move it aside"
    stale="$VENV.stale-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$VENV" "$stale"
    say "moved the old venv to $stale"
fi
if [ ! -e "$VENV" ]; then
    say "creating $VENV and installing the closed pinned set (--no-deps)"
    "$CLIENT_BASE_PYTHON" -m venv "$VENV"
    if ! "$PY" -m pip install --no-deps --no-input --disable-pip-version-check \
        -r "$HERE/client-requirements.txt" 2>&1 | tee "$WORK_DIR/logs/pip-install.log"; then
        die "pip install failed; see $WORK_DIR/logs/pip-install.log"
    fi
    check || die "the fresh venv does not match client-requirements.txt"
fi

"$PY" "$HERE/sweep_tool.py" tokenizer-view --model-path "$MODEL_PATH" --view "$TOKENIZER_VIEW" \
    --revision "${MODEL_REVISION:-}" --out "$TOKENIZER_MANIFEST" || die "tokenizer view check failed"
"$PY" -m pip freeze --all --disable-pip-version-check > "$WORK_DIR/logs/pip-freeze.txt"
say "client ready: $CLIENT_MANIFEST, $TOKENIZER_MANIFEST"
