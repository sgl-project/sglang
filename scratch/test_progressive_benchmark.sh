#!/usr/bin/env bash
# =============================================================================
# Progressive Resolution Benchmark Suite — FLUX.1-dev
# =============================================================================
#
# PURPOSE
#   Measure wall-clock denoising time for every relevant combination of:
#     - progressive mode vs fullres
#     - DIT CPU offload on/off (key variable: offload dilutes quadratic speedup)
#     - TeaCache on/off
#     - TeaCache + progressive compatibility
#
# GROUPS
#   A  Baseline: fullres vs progressive, NO offload, NO optimizations
#      → shows real compute speedup, comparable to paper numbers
#   B  Fullres with TeaCache (GPU-resident)
#      → best-effort fullres throughput with all available opts
#   C  Progressive + TeaCache compatibility test (GPU-resident)
#      → does progressive generation break with TeaCache enabled?
#
# USAGE
#   bash scratch/test_progressive_benchmark.sh [--steps N] [--seed N] [--group A|B|C|all]
#
# REQUIREMENTS
#   GPU with ≥40 GB VRAM (FLUX transformer ~22 GB; no-offload needs full residency)
#
# REPRODUCIBILITY
#   Every run uses the same --seed, --steps, --prompt, and --attention-backend.
#   Runs are sequential (one sglang process at a time) so GPU is fully idle
#   between runs.  Timing is wall-clock from "Pixel data generated successfully".
#
# =============================================================================
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

FLUX_MODEL="/miele/brian/modelscope/black-forest-labs/FLUX.1-dev"
STEPS=50
SEED=42
GROUP="all"
PROMPT="A serene mountain lake at golden hour, photorealistic"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  STEPS="$2"; shift 2 ;;
        --seed)   SEED="$2";  shift 2 ;;
        --group)  GROUP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRATCH_DIR/results/bench_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "run_id\ttotal_s\tdenoise_s\tavg_step_s" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Results: $RESULTS_DIR"
echo "Config: steps=$STEPS seed=$SEED"
echo ""

# ---------------------------------------------------------------------------
# run_gen <label> <extra_flags...>
# Runs sglang generate, saves output, records timing.
# All runs: --dit-cpu-offload false (transformer GPU-resident).
# Exception: Cache-Dit runs omit --dit-cpu-offload false because Cache-Dit
# manages its own residency and disallows the flag.
# ---------------------------------------------------------------------------
run_gen() {
    local label="$1"; shift
    local use_offload_flag=true
    local new_args=()
    for arg in "$@"; do
        [[ "$arg" == "--no-offload-flag" ]] && use_offload_flag=false || new_args+=("$arg")
    done
    set -- "${new_args[@]}"

    local outfile="$RESULTS_DIR/${label}.png"
    local logfile="$RESULTS_DIR/${label}.log"
    echo ""
    echo "─── $label ──────────────────────────────────────────────────"

    local offload_flags=()
    $use_offload_flag && offload_flags=(--dit-cpu-offload false)

    time sglang generate \
        --model-path "$FLUX_MODEL" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        "${offload_flags[@]}" \
        "$@" \
        2>&1 | tee "$logfile"

    local total_s denoise_s avg_s
    total_s=$(grep -oP "generated successfully in \K[\d.]+" "$logfile" || echo "NA")
    # DiT denoising loop time only (excludes text encoding, VAE decode, model load).
    # Progressive runs log "Progressive denoising done in X.XXs" directly.
    # Fullres runs: denoising_start/end in DenoisingStage cover the step loop only;
    # "average time per step: X.XXs" × n_steps gives the same window.
    avg_s=$(grep -oP "average time per step: \K[\d.]+" "$logfile" || echo "NA")
    if prog_done=$(grep -oP "Progressive denoising done in \K[\d.]+" "$logfile" 2>/dev/null); then
        denoise_s="$prog_done"
    elif [[ "$avg_s" != "NA" ]]; then
        denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l)
    else
        denoise_s="NA"
    fi
    echo -e "${label}\t${total_s}\t${denoise_s}\t${avg_s}" >> "$TIMING_LOG"
    echo "  ✓  total=${total_s}s  denoise_loop=${denoise_s}s  avg=${avg_s}s/step"
}

# =============================================================================
# GROUP A — Pure baseline: fullres vs progressive, GPU-resident, no opts
# =============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "A" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  GROUP A  Pure baseline — GPU-resident, no optimizations    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # A1: fullres (reference for speedup calculations)
    run_gen "A1_fullres"

    # A2: progressive dct_rewind L1 δ=0.01 (standard, transition ~step 18/50)
    run_gen "A2_prog_L1_d0.01" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta 0.01

    # A3: progressive dct_rewind L1 δ=0.05 (more aggressive, transition ~step 28/50)
    run_gen "A3_prog_L1_d0.05" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta 0.05

    # A4: progressive dct_rewind L2 δ=0.01 (3-stage: 32→64→128 latent)
    run_gen "A4_prog_L2_d0.01" \
        --progressive-mode dct_rewind \
        --progressive-levels 2 \
        --progressive-delta 0.01
fi

# =============================================================================
# GROUP B — Fullres + TeaCache (GPU-resident)
# =============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "B" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  GROUP B  Fullres + TeaCache                                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # B1: fullres + TeaCache (default threshold, ~1.5x expected)
    # TeaCache skips redundant transformer steps using temporal similarity.
    # Compatible with progressive? TeaCache state is per-step; if we reset it
    # at stage transitions it should work. Tested in Group C.
    run_gen "B1_fullres_teacache" \
        --enable-teacache
fi

# =============================================================================
# GROUP C — Progressive + TeaCache compatibility
# =============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "C" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  GROUP C  Progressive + TeaCache (compatibility test)       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # C1: progressive L1 + TeaCache
    # Expected: either works (if TeaCache resets at transition) or produces
    # wrong output / error (sequence-length mismatch in cached states).
    run_gen "C1_prog_L1_d0.01_teacache" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta 0.01 \
        --enable-teacache

    run_gen "C2_prog_L1_d0.05_teacache" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta 0.05 \
        --enable-teacache
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK COMPLETE                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo ""
echo "Timing (raw):"
column -t -s $'\t' "$TIMING_LOG"
echo ""

# Speedup from DiT denoising loop only (column 3 = denoise_s).
# This excludes text encoding, model loading, VAE decode — those are fixed
# overhead and identical across all runs.  Speedup should only reflect
# how much faster the DiT forward passes run.
A1_denoise=$(grep "^A1_fullres" "$TIMING_LOG" | awk -F'\t' '{print $3}')
if [[ -n "$A1_denoise" && "$A1_denoise" != "NA" ]]; then
    echo "Speedup vs A1_fullres DiT loop (${A1_denoise}s):"
    printf "  %-42s  %8s  %8s  %8s\n" "run_id" "total_s" "denoise_s" "speedup"
    printf "  %-42s  %8s  %8s  %8s\n" "------" "-------" "---------" "-------"
    while IFS=$'\t' read -r run_id total denoise avg; do
        [[ "$run_id" == "run_id" || "$denoise" == "NA" || "$denoise" == "denoise_s" ]] && continue
        speedup=$(echo "scale=2; $A1_denoise / $denoise" | bc 2>/dev/null || echo "?")
        printf "  %-42s  %8.1f  %9.2f  %7sx\n" "$run_id" "${total:-0}" "${denoise:-0}" "$speedup"
    done < "$TIMING_LOG"
fi

echo ""
echo "Logs:    $RESULTS_DIR/*.log"
echo "Montage: montage $RESULTS_DIR/*.png -geometry 512x512+2+2 $RESULTS_DIR/montage.png"
