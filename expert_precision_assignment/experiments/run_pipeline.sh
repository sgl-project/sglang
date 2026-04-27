#!/usr/bin/env bash
# One entrypoint for the full eval pipeline driven by a recipe YAML.
#
# Stages (in order):
#   prep     → pipeline/prompt/prepare_prompts_<task>.py --recipe <yaml>
#              writes pipeline/prompt/<task>_<variant>.jsonl + .meta.jsonl
#   calib    → pipeline/kv_calib/run_calib.sh <task>_<variant>
#              (KV calibration on BF16 baseline)
#              writes data/kv_calib/calib_<task>_<variant>_n<N>.jsonl + .json
#   gen      → pipeline/gen_config/gen_all.py --task <task>_<variant> --calib_json <path>
#              writes data/configs/<task>_<variant>/mc{mc}/...
#   sweep    → pipeline/run_sweep.sh <task>_<variant>   (6×11 grid)
#              writes data/results/<task>_<variant>/mc{mc}_{variant}.jsonl
#   score    → pipeline/scoring/score_traces_<task>.py per trace
#              writes data/results/<task>_<variant>/mc{mc}_{variant}.scores.json
#   collect  → pipeline/collect_result/collect_results.py
#              writes data/results/<task>_<variant>/summary.csv
#
# Usage:
#   bash run_pipeline.sh recipe/yamls/ifbench_nothink.yaml               # full pipeline
#   bash run_pipeline.sh recipe/yamls/ifbench_nothink.yaml --stage prep  # only prep
#   bash run_pipeline.sh recipe/yamls/ifbench_nothink.yaml --stage calib
#   bash run_pipeline.sh recipe/yamls/ifbench_nothink.yaml --stages prep,calib,gen
#   bash run_pipeline.sh recipe/yamls/ifbench_nothink.yaml --from sweep  # sweep onwards
#
# Env overrides still work per-stage — e.g. you can pass NUM_PROMPTS=16 to
# override the recipe's calibration.num_prompts when debugging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PIPELINE_DIR="$SCRIPT_DIR/pipeline"
DATA_DIR="$SCRIPT_DIR/data"

usage() {
    sed -n '1,32p' "$0" | sed 's/^#\{1,2\} \{0,1\}//' >&2
    exit 2
}

[ $# -ge 1 ] || usage
case "$1" in
    --help|-h) usage ;;
esac
RECIPE="$1"; shift
[ -f "$RECIPE" ] || { echo "Recipe not found: $RECIPE" >&2; exit 2; }

# Bootstrap python — just enough to parse the recipe.  The real runtime
# python is re-resolved below, using the recipe's `runtime.python` field.
# Resolve to an absolute path so the executability check below is meaningful.
BOOTSTRAP_PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

ALL_STAGES=(prep calib gen sweep score collect)
STAGES_REQ=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage)   STAGES_REQ="$2"; shift 2 ;;
        --stages)  STAGES_REQ="$2"; shift 2 ;;
        --from)    STAGES_REQ="from:$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

# Decide which stages to run.
if [ -z "$STAGES_REQ" ]; then
    STAGES=("${ALL_STAGES[@]}")
elif [[ "$STAGES_REQ" == from:* ]]; then
    from="${STAGES_REQ#from:}"
    STAGES=()
    active=0
    for s in "${ALL_STAGES[@]}"; do
        [ "$s" = "$from" ] && active=1
        [ $active -eq 1 ] && STAGES+=("$s")
    done
    [ ${#STAGES[@]} -eq 0 ] && { echo "Unknown --from stage: $from" >&2; exit 2; }
else
    IFS=',' read -r -a STAGES <<< "$STAGES_REQ"
fi

# Pull recipe fields into RECIPE_* env vars.
eval "$("$BOOTSTRAP_PYTHON" "$PIPELINE_DIR/recipe.py" "$RECIPE" env)"

# Resolve the real runtime python: user $PYTHON env > recipe.runtime.python > bootstrap default.
PYTHON="${PYTHON:-${RECIPE_RUNTIME__PYTHON:-$BOOTSTRAP_PYTHON}}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: runtime python not executable: $PYTHON" >&2
    echo "  Fix: set PYTHON=/path/to/venv/bin/python, or runtime.python in $RECIPE" >&2
    exit 1
fi
if ! "$PYTHON" -c "import sglang" >/dev/null 2>&1; then
    echo "ERROR: sglang not importable under $PYTHON" >&2
    echo "  Fix: set PYTHON=/path/to/venv with sglang, or runtime.python in $RECIPE" >&2
    exit 1
fi

# Export runtime knobs so child scripts (run_calib.sh, run_sweep.sh, gen_all.py, scorers)
# inherit the recipe's choices without each having to re-parse the YAML.
export PYTHON
export RECIPE_RUNTIME__PYTHON RECIPE_RUNTIME__MODEL_PATH RECIPE_RUNTIME__HOST

TASK="$RECIPE_TASK"
NAME="$RECIPE_NAME"
CALIB_N="${RECIPE_CALIBRATION__NUM_PROMPTS}"
PROMPTS_JSONL="$PIPELINE_DIR/prompt/${NAME}.jsonl"
# run_calib.sh nests its outputs under data/kv_calib/<name>/.
CALIB_DIR="$DATA_DIR/kv_calib/${NAME}"
CALIB_JSON="$CALIB_DIR/calib.json"
CALIB_TRACE="$CALIB_DIR/details_mc${RECIPE_CALIBRATION__MC}.jsonl"
RESULTS_DIR="$DATA_DIR/results/${NAME}"
SCORER="$PIPELINE_DIR/scoring/score_traces_${TASK}.py"
META_JSONL="$PIPELINE_DIR/prompt/${NAME}.meta.jsonl"
PREP_SCRIPT="$PIPELINE_DIR/prompt/prepare_prompts_${TASK}.py"
# Tasks without a prepare_prompts script (e.g., gsm8k) run in bench_eval
# mode — lm-eval pulls the dataset itself, so prep is a no-op and the
# calib/score stages must not gate on the openai-mode prompts file.
HAS_PREP=0
[ -f "$PREP_SCRIPT" ] && HAS_PREP=1

echo "============================================================"
echo "  recipe         $RECIPE"
echo "  name           $NAME   (task=$TASK  variant=$RECIPE_VARIANT)"
echo "  stages         ${STAGES[*]}"
echo "  prompts →      $PROMPTS_JSONL"
echo "  calib   →      $CALIB_JSON  ($CALIB_TRACE)"
echo "  configs →      data/configs/${NAME}/"
echo "  results →      $RESULTS_DIR/"
echo "============================================================"

run_stage() {
    local stage="$1"
    echo ""
    echo "=================== STAGE: $stage ==========================="
    case "$stage" in
        prep)
            if [ "$HAS_PREP" = "1" ]; then
                "$PYTHON" "$PREP_SCRIPT" --recipe "$RECIPE"
            else
                echo "  no prepare_prompts_${TASK}.py — bench_eval mode pulls the dataset itself, skipping prep."
            fi
            ;;
        calib)
            if [ "$HAS_PREP" = "1" ] && [ ! -f "$PROMPTS_JSONL" ]; then
                echo "Run 'prep' first: missing $PROMPTS_JSONL" >&2
                exit 3
            fi
            # Pass calibration knobs via env (run_calib.sh already reads these).
            # BENCH_TASK lets bench_eval get the lm-eval task name (TASK) while
            # the run_calib script still scopes outputs by NAME (compound).
            # CALIB_LIMIT caps bench_eval docs to recipe.calibration.num_prompts.
            CALIB_MC="${CALIB_MC:-$RECIPE_CALIBRATION__MC}" \
            NUM_PROMPTS="${NUM_PROMPTS:-$CALIB_N}" \
            CALIB_LIMIT="${CALIB_LIMIT:-$CALIB_N}" \
            CALIB_GPU="${CALIB_GPU:-$RECIPE_CALIBRATION__GPU}" \
            CALIB_PORT="${CALIB_PORT:-$RECIPE_CALIBRATION__PORT}" \
            CALIB_NCCL_PORT="${CALIB_NCCL_PORT:-$RECIPE_CALIBRATION__NCCL_PORT}" \
            BENCH_TASK="${BENCH_TASK:-$TASK}" \
                bash "$PIPELINE_DIR/kv_calib/run_calib.sh" "$NAME"
            ;;
        gen)
            [ -f "$CALIB_JSON" ] || { echo "Run 'calib' first: missing $CALIB_JSON" >&2; exit 3; }
            # Forward recipe knobs so gen_heter_configs.py / gen_dyna_variants.py
            # honor the YAML's sweep.mc_list and sweep.variants.
            MC_LIST="${MC_LIST:-${RECIPE_SWEEP__MC_LIST:-}}" \
            VARIANTS="${VARIANTS:-${RECIPE_SWEEP__VARIANTS:-}}" \
                "$PYTHON" "$PIPELINE_DIR/gen_config/gen_all.py" --task "$NAME" --calib_json "$CALIB_JSON"
            ;;
        sweep)
            if [ "$HAS_PREP" = "1" ] && [ ! -f "$PROMPTS_JSONL" ]; then
                echo "Run 'prep' first: missing $PROMPTS_JSONL" >&2
                exit 3
            fi
            # Recipe convention: sweep.num_prompts=0 means "all".  Shell's
            # ${x:-default} keeps the literal "0" because it's non-empty, so
            # translate it here explicitly before exporting to run_sweep.sh.
            sweep_np="${OPENAI_NUM_PROMPTS:-${RECIPE_SWEEP__NUM_PROMPTS:-999999}}"
            [ "$sweep_np" = "0" ] && sweep_np=999999
            # Recipe sweep.limit=0 means "no cap" (full task) — translate
            # to empty so run_sweep.sh's `[ -n "$LIMIT" ]` skips the flag.
            sweep_lim="${LIMIT:-${RECIPE_SWEEP__LIMIT:-}}"
            [ "$sweep_lim" = "0" ] && sweep_lim=""
            MC_LIST="${MC_LIST:-${RECIPE_SWEEP__MC_LIST:-}}" \
            VARIANTS="${VARIANTS:-$RECIPE_SWEEP__VARIANTS}" \
            GPUS="${GPUS:-$RECIPE_SWEEP__GPUS}" \
            OPENAI_NUM_PROMPTS="$sweep_np" \
            LIMIT="$sweep_lim" \
            BENCH_TASK="${BENCH_TASK:-$TASK}" \
                bash "$PIPELINE_DIR/run_sweep.sh" "$NAME"
            ;;
        score)
            # Opt-out via recipe: scoring.enabled=false turns this stage into
            # a no-op. `collect` still runs fine — the summary CSV just won't
            # have score_* columns.
            if [ "${RECIPE_SCORING__ENABLED:-1}" = "0" ]; then
                echo "  scoring.enabled=false in recipe — skipping score stage."
                return 0
            fi
            [ -d "$RESULTS_DIR" ] || { echo "Run 'sweep' first: missing $RESULTS_DIR" >&2; exit 3; }
            [ -f "$SCORER" ] || { echo "No scorer for task=$TASK at $SCORER" >&2; exit 3; }
            [ -f "$META_JSONL" ] || { echo "Missing meta: $META_JSONL" >&2; exit 3; }
            # Extra CLI args forwarded from recipe.scoring.extra_args.
            local scorer_extra=()
            if [ -n "${RECIPE_SCORING__EXTRA_ARGS:-}" ]; then
                read -r -a scorer_extra <<< "$RECIPE_SCORING__EXTRA_ARGS"
            fi
            # Score every mc*_*.jsonl that doesn't have a fresh sidecar.
            shopt -s nullglob
            local traces=("$RESULTS_DIR"/mc*_*.jsonl)
            shopt -u nullglob
            if [ ${#traces[@]} -eq 0 ]; then
                echo "No traces found in $RESULTS_DIR — did sweep produce output?"
                return 0
            fi
            for t in "${traces[@]}"; do
                sidecar="${t%.jsonl}.scores.json"
                if [ -f "$sidecar" ] && [ "$sidecar" -nt "$t" ]; then
                    echo "  [up-to-date] $(basename "$sidecar")"
                    continue
                fi
                echo "  scoring $(basename "$t") ..."
                "$PYTHON" "$SCORER" --trace "$t" --meta "$META_JSONL" \
                    "${scorer_extra[@]}" || \
                    echo "    (scorer FAILED for $t — continuing)" >&2
            done
            ;;
        collect)
            [ -d "$RESULTS_DIR" ] || { echo "Missing $RESULTS_DIR" >&2; exit 3; }
            "$PYTHON" "$PIPELINE_DIR/collect_result/collect_results.py" \
                --results_dir "$RESULTS_DIR" \
                --out_csv "$RESULTS_DIR/summary.csv"
            echo "  → $RESULTS_DIR/summary.csv"
            ;;
        *) echo "Unknown stage: $stage" >&2; exit 2 ;;
    esac
}

for s in "${STAGES[@]}"; do
    run_stage "$s"
done

echo ""
echo "============================================================"
echo "  Pipeline complete for $NAME"
[ -f "$RESULTS_DIR/summary.csv" ] && echo "  Summary CSV: $RESULTS_DIR/summary.csv"
echo "============================================================"
