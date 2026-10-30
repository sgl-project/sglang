#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Uses the visible GPUs; this script never requests or allocates GPUs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_ovis_image_matrix.sh [--quick] [--dry-run] [--max-gpus N] [CASE ...]

Required environment: NATIVE_PYTHON REFERENCE_PYTHON MODEL_PATH RESULTS
Optional environment: HEIGHT WIDTH STEPS SEED (defaults: 1024 1024 50 42)
--quick defaults to 512x512, four steps; explicit environment values take priority.
VAE tiling/spatial cases always use 1024x1536 to exceed the native tile threshold.
--dry-run prints commands without importing Torch, creating files, or using GPUs.
--max-gpus limits cases to N visible GPUs; it does not allocate GPUs.
--list prints the available case names. With no CASE arguments, run the full matrix.
EOF
}

all_cases=(
    components reference single no-cfg batch2 prompt-batch component layerwise
    tiny-tp2 tiny-ulysses2 tiny-ring2 tiny-tp2-sp2
    tp2 ulysses2 ring2 cfg2 tp2-sp2 vae-tiled vae-spatial http
)
quick=false
dry_run=false
max_gpus=0
selected=()
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --list) printf '%s\n' "${all_cases[@]}"; exit 0 ;;
        --quick) quick=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --max-gpus)
            if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s\n' '--max-gpus needs a positive integer' >&2
                exit 2
            fi
            max_gpus=$2
            shift 2
            ;;
        --*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
        *) selected+=("$1"); shift ;;
    esac
done
explicit_selection=true
if ((${#selected[@]} == 0)); then
    explicit_selection=false
    selected=("${all_cases[@]}")
fi
for name in "${selected[@]}"; do
    known=false
    for allowed in "${all_cases[@]}"; do
        if [[ "$name" == "$allowed" ]]; then known=true; break; fi
    done
    if [[ "$known" == false ]]; then
        printf 'Unknown case: %s (use --list)\n' "$name" >&2
        exit 2
    fi
done

: "${NATIVE_PYTHON:?Set NATIVE_PYTHON to the native environment Python}"
: "${REFERENCE_PYTHON:?Set REFERENCE_PYTHON to the reference environment Python}"
: "${MODEL_PATH:?Set MODEL_PATH to the pinned local Diffusers directory}"
: "${RESULTS:?Set RESULTS to an artifact directory outside the checkout}"
for executable in "$NATIVE_PYTHON" "$REFERENCE_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        printf 'Python is not executable: %s\n' "$executable" >&2
        exit 2
    fi
done
for component in scheduler tokenizer text_encoder transformer vae; do
    if [[ ! -d "$MODEL_PATH/$component" ]]; then
        printf 'Missing model directory: %s/%s\n' "$MODEL_PATH" "$component" >&2
        exit 2
    fi
done
if [[ ! -f "$MODEL_PATH/model_index.json" ]]; then
    printf 'Missing model_index.json: %s\n' "$MODEL_PATH" >&2
    exit 2
fi

default_size=1024
default_steps=50
if [[ "$quick" == true ]]; then default_size=512; default_steps=4; fi
height=${HEIGHT:-$default_size}
width=${WIDTH:-$default_size}
steps=${STEPS:-$default_steps}
seed=${SEED:-42}
for integer in "$height" "$width" "$steps"; do
    if [[ ! "$integer" =~ ^[1-9][0-9]*$ ]]; then
        printf 'HEIGHT, WIDTH, and STEPS must be positive integers\n' >&2
        exit 2
    fi
done
if [[ ! "$seed" =~ ^[0-9]+$ ]] || ((height % 16 || width % 16)); then
    printf 'SEED must be nonnegative; HEIGHT and WIDTH must be multiples of 16\n' >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
cd "$repo_root"
export PATH="$(dirname -- "$NATIVE_PYTHON"):$PATH"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
validator="$repo_root/test/manual/diffusion/validate_ovis_image.py"
tiny_validator="$repo_root/test/manual/diffusion/check_ovis_image_parallel.py"
run_dir="$RESULTS/$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$dry_run" == true ]]; then
    gpu_limit=${max_gpus:-0}
    if ((gpu_limit == 0)); then gpu_limit=4; fi
else
    visible_gpus=$("$NATIVE_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
    if ((visible_gpus < 1)); then
        printf 'No visible GPUs; run on a GPU host or use --dry-run\n' >&2
        exit 2
    fi
    gpu_limit=$visible_gpus
    if ((max_gpus > 0 && max_gpus < gpu_limit)); then gpu_limit=$max_gpus; fi
    mkdir -p "$run_dir/logs"
    git rev-parse HEAD > "$run_dir/native-revision.txt"
    git status --short > "$run_dir/source-status.txt"
    git diff --binary HEAD > "$run_dir/source.patch"
    "$NATIVE_PYTHON" -m pip freeze > "$run_dir/native-freeze.txt"
    "$REFERENCE_PYTHON" -m pip freeze > "$run_dir/reference-freeze.txt"
    "$NATIVE_PYTHON" - <<'PY' > "$run_dir/gpus.json"
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "devices": [
        {"index": i, "name": torch.cuda.get_device_name(i),
         "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ],
}, indent=2))
PY
    printf 'case\trequired_gpus\tstatus\tartifact\n' > "$run_dir/status.tsv"
fi
printf 'Results: %s\nProfile: %sx%s, %s steps, seed %s; GPU limit %s\n' \
    "$run_dir" "$height" "$width" "$steps" "$seed" "$gpu_limit"

run_logged() {
    local label=$1
    shift
    if [[ "$dry_run" == true ]]; then
        printf '%s: ' "$label"
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    printf 'Running %s (log: %s/logs/%s.log)\n' "$label" "$run_dir" "$label"
    "$@" > "$run_dir/logs/$label.log" 2>&1
}

status() {
    local display=$3
    if [[ "$dry_run" == false ]]; then
        printf '%s\t%s\t%s\t%s\n' "$@" >> "$run_dir/status.tsv"
    elif [[ "$display" == pass ]]; then
        display=planned
    fi
    printf '%s: %s\n' "$1" "$display"
}

declare -A reference_status=()
ensure_reference() {
    local profile=$1
    shift
    local known=${reference_status[$profile]:-}
    if [[ "$known" == pass ]]; then return 0; fi
    if [[ "$known" == fail ]]; then return 1; fi
    if run_logged "reference-$profile" "$REFERENCE_PYTHON" "$validator" \
        --mode reference "${common[@]}" "$@" --output "$run_dir/reference-$profile"; then
        reference_status[$profile]=pass
        status "reference-$profile" 1 pass "$run_dir/reference-$profile"
        return 0
    fi
    reference_status[$profile]=fail
    status "reference-$profile" 1 fail "$run_dir/logs/reference-$profile.log"
    return 1
}

failures=0
for name in "${selected[@]}"; do
    required=1
    topology=()
    extra=()
    profile=default
    case_height=$height
    case_width=$width
    case "$name" in
        tiny-tp2|tp2) required=2; topology=(--tp 2) ;;
        tiny-ulysses2|ulysses2) required=2; topology=(--ulysses 2) ;;
        tiny-ring2|ring2) required=2; topology=(--ring 2) ;;
        tiny-tp2-sp2|tp2-sp2) required=4; topology=(--tp 2 --ulysses 2) ;;
        cfg2) required=2; topology=(--cfg 2) ;;
        component) extra=(--offload component) ;;
        layerwise) extra=(--offload layerwise) ;;
        no-cfg) profile=no-cfg; extra=(--guidance 1) ;;
        batch2) profile=batch2; extra=(--outputs 2) ;;
        prompt-batch)
            profile=prompt-batch
            extra=(--outputs 2 --second-prompt 'A blue sailboat on a calm sea, watercolor painting.')
            ;;
        vae-tiled)
            profile=vae-tiled
            case_height=1024
            case_width=1536
            extra=(--vae-tiling)
            ;;
        vae-spatial)
            required=2
            topology=(--ulysses 2)
            profile=vae-tiled
            case_height=1024
            case_width=1536
            extra=(--vae-tiling --vae-sp)
            ;;
    esac
    common=(--model-path "$MODEL_PATH" --height "$case_height" --width "$case_width" --steps "$steps" --seed "$seed")
    if ((required > gpu_limit)); then
        status "$name" "$required" skipped "needs $required GPUs; limit is $gpu_limit"
        if [[ "$explicit_selection" == true ]]; then failures=$((failures + 1)); fi
        continue
    fi
    if [[ "$name" == components ]]; then
        if run_logged "$name" "$NATIVE_PYTHON" -m pytest -q \
            python/sglang/multimodal_gen/test/unit/test_ovis_image.py \
            python/sglang/multimodal_gen/test/unit/test_ovis_image_config.py; then
            status "$name" 1 pass "$run_dir/logs/$name.log"
        else
            status "$name" 1 fail "$run_dir/logs/$name.log"
            failures=$((failures + 1))
        fi
        continue
    fi
    if [[ "$name" == http ]]; then
        if run_logged "$name" "$NATIVE_PYTHON" -m pytest -s -q \
            python/sglang/multimodal_gen/test/server/test_server_ovis_image.py; then
            status "$name" 1 pass "$run_dir/logs/$name.log"
        else
            status "$name" 1 fail "$run_dir/logs/$name.log"
            failures=$((failures + 1))
        fi
        continue
    fi
    if [[ "$name" == tiny-* ]]; then
        if run_logged "$name" "$NATIVE_PYTHON" -m torch.distributed.run \
            --standalone --nproc_per_node "$required" "$tiny_validator" "${topology[@]}"; then
            status "$name" "$required" pass "$run_dir/logs/$name.log"
        else
            status "$name" "$required" fail "$run_dir/logs/$name.log"
            failures=$((failures + 1))
        fi
        continue
    fi
    reference_extra=("${extra[@]}")
    if [[ "$name" == component || "$name" == layerwise ]]; then reference_extra=(); fi
    if [[ "$name" == vae-spatial ]]; then reference_extra=(--vae-tiling); fi
    if ! ensure_reference "$profile" "${reference_extra[@]}"; then
        status "$name" "$required" fail 'matching reference failed'
        failures=$((failures + 1))
        continue
    fi
    if [[ "$name" == reference ]]; then continue; fi
    attention=torch_sdpa
    if [[ "$name" == ring2 ]]; then attention=fa; fi
    output="$run_dir/$name"
    if run_logged "$name" "$NATIVE_PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node "$required" "$validator" --mode native \
        "${common[@]}" "${topology[@]}" "${extra[@]}" --attention "$attention" --output "$output"; then
        if run_logged "$name-compare" env CUDA_VISIBLE_DEVICES= "$REFERENCE_PYTHON" "$validator" \
            --mode compare --reference "$run_dir/reference-$profile" --output "$output"; then
            baseline=""
            if [[ "$profile" == default && "$name" != single ]]; then
                baseline="$run_dir/single"
            elif [[ "$name" == vae-spatial ]]; then
                baseline="$run_dir/vae-tiled"
            fi
            if [[ -n "$baseline" && ( "$dry_run" == true || -f "$baseline/tensors.pt" ) ]]; then
                if run_logged "$name-single-card-compare" env CUDA_VISIBLE_DEVICES= \
                    "$REFERENCE_PYTHON" "$validator" --mode compare \
                    --reference "$baseline" --output "$output" \
                    --comparison-name single-card-comparison.json; then
                    status "$name" "$required" pass "$output/comparison.json"
                else
                    status "$name" "$required" fail "$run_dir/logs/$name-single-card-compare.log"
                    failures=$((failures + 1))
                fi
            else
                status "$name" "$required" pass "$output/comparison.json"
            fi
        else
            status "$name" "$required" fail "$run_dir/logs/$name-compare.log"
            failures=$((failures + 1))
        fi
    else
        status "$name" "$required" fail "$run_dir/logs/$name.log"
        failures=$((failures + 1))
    fi
done
if ((failures)); then
    printf 'Failed or unavailable selected cases: %s\n' "$failures" >&2
    exit 1
fi
