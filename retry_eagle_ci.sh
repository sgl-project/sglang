#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-local}"
TEST_MODULE="${TEST_MODULE:-registered.spec.eagle.test_eagle_infer_b}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"

REPO="sgl-project/sglang"
BRANCH="${BRANCH:-$(git branch --show-current)}"
WORKFLOW="pr-test.yml"
STAGES=("stage-b-test-large-1-gpu" "stage-b-test-small-1-gpu")

echo "Eagle Retry | mode=$MODE test=$TEST_MODULE max=$MAX_ITERATIONS"
echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "---"

run_local() {
    for i in $(seq 1 "$MAX_ITERATIONS"); do
        echo "#$i  START  $(date '+%H:%M:%S')"
        cd test/
        python -m unittest "$TEST_MODULE" -v 2>&1
        EXIT_CODE=$?
        cd ..

        if [ $EXIT_CODE -ne 0 ]; then
            echo "#$i  CRASH  exit=$EXIT_CODE  $(date '+%H:%M:%S')"
            echo "CRASH DETECTED on iteration $i — $(date '+%Y-%m-%d %H:%M:%S')"
            exit 0
        fi

        echo "#$i  PASS   $(date '+%H:%M:%S')"
        sleep "$SLEEP_BETWEEN"
    done
    echo "No crash after $MAX_ITERATIONS iterations."
    exit 1
}

run_ci() {
    for i in $(seq 1 "$MAX_ITERATIONS"); do
        echo "#$i  TRIGGER  $(date '+%H:%M:%S')"

        for stage in "${STAGES[@]}"; do
            gh workflow run "$WORKFLOW" --repo "$REPO" --ref "$BRANCH" \
                -f target_stage="$stage" 2>/dev/null
            sleep 5
        done

        sleep 15
        RUN_IDS=()
        RUN_NAMES=()
        for attempt in $(seq 1 5); do
            RUN_IDS=()
            RUN_NAMES=()
            LINES=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" \
                --branch "$BRANCH" --limit "${#STAGES[@]}" \
                --json databaseId,displayTitle \
                -q '.[] | "\(.databaseId) \(.displayTitle)"' 2>/dev/null || true)
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue
                RUN_IDS+=("${line%% *}")
                RUN_NAMES+=("${line#* }")
            done <<< "$LINES"
            [[ ${#RUN_IDS[@]} -ge ${#STAGES[@]} ]] && break
            sleep 10
        done

        for idx in "${!RUN_IDS[@]}"; do
            echo "#$i  START   [${RUN_NAMES[$idx]:-?}] https://github.com/$REPO/actions/runs/${RUN_IDS[$idx]}  $(date '+%H:%M:%S')"
        done

        CRASHED=false
        for rid in "${RUN_IDS[@]}"; do
            RUN_URL="https://github.com/$REPO/actions/runs/$rid"
            while true; do
                STATUS=$(gh run view "$rid" --repo "$REPO" --json status,conclusion \
                    -q '[.status,.conclusion] | join(",")' 2>/dev/null || echo "unknown,")
                case "$STATUS" in
                    completed,success)
                        echo "#$i  PASS    $RUN_URL  $(date '+%H:%M:%S')"
                        break
                        ;;
                    completed,*)
                        CONCLUSION="${STATUS#completed,}"
                        echo "#$i  FAIL($CONCLUSION)  $RUN_URL  $(date '+%H:%M:%S')"
                        CRASHED=true
                        break
                        ;;
                    *)
                        sleep 60
                        ;;
                esac
            done
        done

        if $CRASHED; then
            echo "CRASH DETECTED on iteration $i — $(date '+%Y-%m-%d %H:%M:%S')"
            exit 0
        fi

        sleep "$SLEEP_BETWEEN"
    done
    echo "No crash after $MAX_ITERATIONS iterations."
    exit 1
}

case "$MODE" in
    local) run_local ;;
    ci)    run_ci ;;
    *)     echo "Unknown MODE=$MODE (use 'local' or 'ci')"; exit 1 ;;
esac
