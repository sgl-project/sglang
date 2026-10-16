# Shared settings for setup_client.sh, prepare_dataset.sh and run_sweep.sh; sourced, not executed.
# Site values come from $SITE_ENV (default: site.env beside this file, git-ignored); see site.env.example.

HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SITE_ENV=${SITE_ENV:-$HERE/site.env}
if [ ! -f "$SITE_ENV" ]; then
    echo "missing site settings $SITE_ENV (copy site.env.example and fill it in)" >&2
    return 1 2>/dev/null || exit 1
fi
set -a
# shellcheck disable=SC1090
source "$SITE_ENV"
set +a
for _v in BASE_URL SERVED_MODEL_NAME MODEL_PATH CLIENT_BASE_PYTHON HF_HOME; do
    if [ -z "${!_v:-}" ] || [[ "${!_v}" == *"<"* ]]; then
        echo "$_v is unset or still a placeholder in $SITE_ENV" >&2
        return 1 2>/dev/null || exit 1
    fi
done
# Container images often export hub-cache variables that outrank HF_HOME; pin every cache under HF_HOME.
export HF_HOME HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
unset TRANSFORMERS_CACHE
# Serving overlays must not shadow the closed client environment.
unset PYTHONPATH PYTHONHOME

WORK_DIR=${WORK_DIR:-$HERE}
# Keep every generated client artifact/cache in this Beam checkout.
python3 - "$HERE" "$WORK_DIR" "$HF_HOME" <<'PYCODE'
import pathlib, sys
repo = pathlib.Path(sys.argv[1]).resolve().parents[1]
for value in sys.argv[2:]:
    if not pathlib.Path(value).resolve().is_relative_to(repo):
        raise SystemExit(f"artifact/cache path must be inside Beam checkout: {value}")
PYCODE
[ $? = 0 ] || return 1
mkdir -p "$WORK_DIR/tmp" "$WORK_DIR/cache"
export TMPDIR=$WORK_DIR/tmp PIP_CACHE_DIR=$WORK_DIR/cache/pip XDG_CACHE_HOME=$WORK_DIR/cache
export PYTHONPYCACHEPREFIX=$WORK_DIR/cache/pycache
VENV=$WORK_DIR/venv
PY=$VENV/bin/python
TOKENIZER_VIEW=$WORK_DIR/tokenizer-view
CLIENT_MANIFEST=$WORK_DIR/client-manifest.json
TOKENIZER_MANIFEST=$WORK_DIR/tokenizer-view.json
NUM_CONVERSATIONS=${NUM_CONVERSATIONS:-1024}
DATASET=${DATASET:-$WORK_DIR/datasets/openhands-padded-$NUM_CONVERSATIONS.json}
RESULTS_DIR=${RESULTS_DIR:-$WORK_DIR/results}
SERVER_PROFILE=${SERVER_PROFILE:-$HERE/server_profile.json}
# GPUs one server instance spans; per-GPU columns divide by it and server-check requires it to match the server.
N_GPUS=${N_GPUS:-4}
if ! [[ $N_GPUS =~ ^[1-9][0-9]*$ ]]; then
    echo "N_GPUS must be a positive integer, got '$N_GPUS'" >&2
    return 1 2>/dev/null || exit 1
fi

# The reference workload; a different value here is a different benchmark, so none of these is a knob.
TURNS=13
FIRST_TURN_TOKENS=74160
SUBSEQUENT_TURN_TOKENS=753
OSL=220
NUM_MULT=2
WARMUP_CONVS=2

# Beam replay includes assistant history without separate thinking fields. Keep this
# identical for dataset rendering and every request; no reasoning-budget override.
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-'{"allow_missing_thinking":true}'}

export TZ=UTC PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
# A client that aborts would otherwise leave a multi-GB core file wherever kernel.core_pattern points.
ulimit -c 0
say() { echo "=== $(date -u +%FT%TZ) $* ==="; }
die() { say "FATAL: $*"; exit 1; }
