#!/bin/sh
set -eu

# Usage: ./benchmark/xpress/serve.sh [xpress|dflash|dspark|target] [extra server flags...]
# Run each mode on the same GPU, with the same target model and server flags.
mode="${1:-xpress}"
if [ "$#" -gt 0 ]; then
    shift
fi

script_dir="$(CDPATH= cd "$(dirname "$0")" && pwd)"
if [ -d "$script_dir/../../python/sglang" ]; then
    repo_root="$(CDPATH= cd "$script_dir/../.." && pwd)"
elif [ -d "$script_dir/python/sglang" ]; then
    repo_root="$script_dir"
elif [ -n "${SGLANG_REPO:-}" ] && [ -d "$SGLANG_REPO/python/sglang" ]; then
    repo_root="$(CDPATH= cd "$SGLANG_REPO" && pwd)"
else
    echo "Cannot find SGLang source; set SGLANG_REPO to its repository root." >&2
    exit 1
fi
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2
python_bin="${PYTHON_BIN:-python3}"
target_model="${TARGET_MODEL:-}"
port="${PORT:-30000}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

if [ -z "$target_model" ] || [ ! -f "$target_model/config.json" ]; then
    echo "Set TARGET_MODEL to a local target model directory containing config.json." >&2
    exit 2
fi

# A partial source sync can pair a new binding.py with an older spec.py.
# Check their imported names before spending time loading the target model.
"$python_bin" - "$repo_root" <<'PY'
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "python/sglang"
binding = root / "srt/kernel_plugins/nvidia/binding.py"
spec = root / "kernels/plugin_kernels/nvidia/spec.py"
if binding.is_file() and spec.is_file():
    binding_tree = ast.parse(binding.read_text())
    spec_tree = ast.parse(spec.read_text())
    imported = {
        alias.name
        for node in binding_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "sglang.kernels.plugin_kernels.nvidia.spec"
        for alias in node.names
    }
    defined = set()
    for node in spec_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    missing = sorted(imported - defined)
    if missing:
        raise SystemExit(
            f"Inconsistent SGLang source in {root}: {spec} lacks {missing} "
            "required by binding.py. Sync both plugin files from the same "
            "SGLang revision before starting the server."
        )

eplb_importer = root / "srt/eplb/expert_distribution.py"
eplb_kernel = root / "kernels/ops/moe/eplb_map_record.py"
if (
    eplb_importer.is_file()
    and "from sglang.kernels.ops.moe.eplb_map_record import" in eplb_importer.read_text()
    and not eplb_kernel.is_file()
):
    raise SystemExit(
        f"Inconsistent SGLang source in {root}: {eplb_importer} imports "
        f"missing {eplb_kernel}. Sync the complete SGLang checkout, including "
        "python/sglang/kernels, before starting the server."
    )
PY

case "$mode" in
    xpress)
        draft_model="${XPRESS_DRAFT:-}"
        if [ -z "$draft_model" ] || [ ! -f "$draft_model/config.json" ]; then
            echo "Set XPRESS_DRAFT to a local XPress checkpoint containing config.json." >&2
            exit 2
        fi
        set -- \
            --speculative-algorithm DFLASH \
            --speculative-draft-model-path "$draft_model" \
            --speculative-num-draft-tokens 16 \
            "$@"
        ;;
    dflash)
        draft_model="${DFLASH_DRAFT:-${XPRESS_DRAFT:-}}"
        if [ -z "$draft_model" ] || [ ! -f "$draft_model/config.json" ]; then
            echo "Set DFLASH_DRAFT to a local DFlash checkpoint containing config.json." >&2
            exit 2
        fi
        if [ -z "${DFLASH_DRAFT:-}" ]; then
            echo "Using XPRESS_DRAFT as the DFlash checkpoint; prefer DFLASH_DRAFT." >&2
        fi
        set -- \
            --speculative-algorithm DFLASH \
            --speculative-draft-model-path "$draft_model" \
            --speculative-num-draft-tokens 16 \
            "$@"
        ;;
    dspark)
        draft_model="${DSPARK_DRAFT:-}"
        if [ -z "$draft_model" ] || [ ! -f "$draft_model/config.json" ]; then
            echo "Set DSPARK_DRAFT to a local DSpark checkpoint containing config.json." >&2
            exit 2
        fi
        set -- \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$draft_model" \
            --speculative-dspark-block-size 15 \
            "$@"
        ;;
    target)
        ;;
    *)
        echo "Usage: $0 [xpress|dflash|dspark|target] [extra server flags...]" >&2
        exit 2
        ;;
esac

if [ "$mode" = xpress ] || [ "$mode" = dflash ]; then
    "$python_bin" - "$mode" "$draft_model/config.json" <<'PY'
import json
import sys

mode, config_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as f:
    architectures = json.load(f).get("architectures") or []
is_xpress = any("XPress" in name for name in architectures)
if mode == "dflash" and is_xpress:
    raise SystemExit(
        f"{config_path} declares an XPress architecture; use a plain DFlash "
        "checkpoint for the dflash baseline."
    )
if mode == "xpress" and not is_xpress:
    raise SystemExit(
        f"{config_path} does not declare an XPress architecture; check XPRESS_DRAFT."
    )
PY
fi

echo "Starting $mode: target=$target_model draft=${draft_model:-none} port=$port gpu=$CUDA_VISIBLE_DEVICES" >&2
exec "$python_bin" -m sglang.launch_server \
    --model-path "$target_model" \
    --host 127.0.0.1 \
    --port "$port" \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --context-length 8192 \
    --trust-remote-code \
    "$@"
