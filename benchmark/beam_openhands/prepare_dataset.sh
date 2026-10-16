#!/usr/bin/env bash
# One-time dataset build: NUM_CONVERSATIONS (default 1024) padded 13-turn conversations, tokenized with the served
# checkpoint's tokenizer and rendered with the chat_template_kwargs that the sweep sends. It reads the two public
# datasets' parquet files at the pinned revisions below. An existing dataset is validated, never overwritten.
#
#   SITE_ENV=/path/to/site.env bash prepare_dataset.sh
#
# The default ladder needs 2 x 128 + 2 = 258 conversations; NUM_CONVERSATIONS=258 builds only that minimum.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

TRAJECTORIES_REVISION=35455389ab51bf5e2306bfd436ef72d0f98bf882   # nebius/SWE-rebench-openhands-trajectories
OPENSCIENCE_REVISION=174b02c9cdf231f220765b2a1d5ece4550921894    # nvidia/OpenScienceReasoning-2
MIN_FREE_GIB=20

[ -x "$PY" ] && [ -f "$TOKENIZER_MANIFEST" ] || die "run setup_client.sh first"
CONTEXT_LENGTH=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["resolved_args"]["context_length"])' "$SERVER_PROFILE")

check_dataset() {
    "$PY" "$HERE/sweep_tool.py" dataset-check --dataset "$DATASET" --tokenizer-manifest "$TOKENIZER_MANIFEST" \
        --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" --need "$NUM_CONVERSATIONS" --turns "$TURNS" \
        --first-turn-length "$FIRST_TURN_TOKENS" --subsequent-turn-length "$SUBSEQUENT_TURN_TOKENS" --osl "$OSL" \
        --render-sample 8 --out "$DATASET.check.json"
}

if [ -e "$DATASET" ]; then
    check_dataset || die "$DATASET exists but fails validation; move it aside to rebuild (it is never overwritten)"
    say "dataset already built and valid: $DATASET"
    exit 0
fi
mkdir -p "$(dirname "$DATASET")" "$WORK_DIR/logs"
free_gib=$(df -B1 --output=avail "$(dirname "$DATASET")" | awk 'NR==2{printf "%d", $1/1073741824}')
[ "$free_gib" -ge "$MIN_FREE_GIB" ] || die "only ${free_gib} GiB free next to $DATASET"

log="$WORK_DIR/logs/build-$(basename "$DATASET" .json)-$(date -u +%Y%m%dT%H%M%SZ).log"
say "building $NUM_CONVERSATIONS conversations x $TURNS turns -> $DATASET (log $log)"
if ! env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
    "$PY" "$HERE/build_openhands_padded_dataset.py" \
    --tokenizer "$TOKENIZER_VIEW" --tokenizer-manifest "$TOKENIZER_MANIFEST" \
    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
    --dataset-revision "$TRAJECTORIES_REVISION" --pad-dataset-revision "$OPENSCIENCE_REVISION" \
    --pad-source openscience --first-turn-length "$FIRST_TURN_TOKENS" --subsequent-turn-length "$SUBSEQUENT_TURN_TOKENS" \
    --num-turns "$TURNS" --osl "$OSL" --context-length "$CONTEXT_LENGTH" --number "$NUM_CONVERSATIONS" \
    --output-path "$DATASET" 2>&1 | tee "$log"; then
    die "dataset build failed; see $log"
fi
check_dataset || die "the new dataset fails validation"
say "dataset ready: $DATASET ($(sha256sum "$DATASET" | cut -c1-16))"
