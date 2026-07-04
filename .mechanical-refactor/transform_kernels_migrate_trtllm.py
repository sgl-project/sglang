#!/usr/bin/env python3
"""Reproducible transform: migrate the experimental trtllm_lora_temp/triton_ops
kernels into sglang.kernels.ops.{gemm,moe}.trtllm_lora_temp.* (RFC #29630).

These are experimental variants of the main LoRA kernels with identical file
names, so they go into a `trtllm_lora_temp` subpackage inside each group to
avoid colliding with the already-migrated production kernels. Consumers live
inside sglang.srt.lora.trtllm_lora_temp and import via the package __init__
re-export; they are rewritten per-symbol (preserving `as` aliases).

Usage:
  python3 transform_kernels_migrate_trtllm.py apply
  python3 transform_kernels_migrate_trtllm.py            # verify reproducibility
"""
import re
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from mechanical_refactor_verify_utils import (  # noqa: E402
    exec_command,
    git_add_and_commit,
    verify_mechanical_refactor,
)

BASE_COMMIT = "ff2845953f"  # main lora migration commit
TARGET_COMMIT = "dc83c6784726b246b1fb3bd90c1af4163e40a33a"

OLD_PKG = "sglang.srt.lora.trtllm_lora_temp.triton_ops"
GEMM_STEMS = [
    "gate_up_lora_b",
    "kernel_utils",
    "kv_b_lora_absorbed",
    "qkv_lora_b",
    "sgemm_lora_a",
    "sgemm_lora_b",
]
MOE_STEMS = ["virtual_experts"]

SYMBOL_MODULE = {
    "gate_up_lora_b_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.gate_up_lora_b",
    "qkv_lora_b_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.qkv_lora_b",
    "sgemm_lora_a_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a",
    "sgemm_lora_b_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_b",
    "step_a_q_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed",
    "step_b_q_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed",
    "step_a_v_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed",
    "step_b_v_fwd": "sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed",
    "get_pdl_launch_metadata": "sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils",
    "merged_experts_fused_moe_lora_add": "sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts",
    "fused_sanitize_expert_ids": "sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts",
}

SUBPKG_INIT = (
    '"""Experimental TRT-LLM LoRA kernel variants (gated by '
    "``SGLANG_EXPERIMENTAL_LORA_OPTI`` / ``lora_envs``).\n\n"
    "Migrated from ``sglang.srt.lora.trtllm_lora_temp.triton_ops`` (RFC #29630)."
    '"""\n'
)


def module_map():
    m = {}
    for stem in GEMM_STEMS:
        m[f"{OLD_PKG}.{stem}"] = f"sglang.kernels.ops.gemm.trtllm_lora_temp.{stem}"
    for stem in MOE_STEMS:
        m[f"{OLD_PKG}.{stem}"] = f"sglang.kernels.ops.moe.trtllm_lora_temp.{stem}"
    return m


def rewrite_package_imports(text):
    def parse_specs(blob):
        return [s.strip() for s in blob.replace("\n", " ").split(",") if s.strip()]

    def emit(indent, specs):
        by_module = {}
        for spec in specs:
            base = spec.split(" as ")[0].strip()
            mod = SYMBOL_MODULE[base]
            by_module.setdefault(mod, []).append(spec)
        lines = []
        for mod in sorted(by_module):
            lines.append(f"{indent}from {mod} import {', '.join(by_module[mod])}")
        return "\n".join(lines)

    text = re.sub(
        r"(?m)^([ \t]*)from " + re.escape(OLD_PKG) + r" import \(([^)]*)\)",
        lambda m: emit(m.group(1), parse_specs(m.group(2))),
        text,
    )
    text = re.sub(
        r"(?m)^([ \t]*)from " + re.escape(OLD_PKG) + r" import ([^\(\n]+)$",
        lambda m: emit(m.group(1), parse_specs(m.group(2))),
        text,
    )
    return text


def rewrite_module_paths(text, mmap):
    for old_mod in sorted(mmap, key=len, reverse=True):
        text = re.sub(re.escape(old_mod) + r"(?![A-Za-z0-9_])", mmap[old_mod], text)
    return text


def transform(dir_root: Path) -> None:
    mmap = module_map()

    # Create the group subpackages first.
    for group in ["gemm", "moe"]:
        subpkg = dir_root / f"python/sglang/kernels/ops/{group}/trtllm_lora_temp"
        subpkg.mkdir(parents=True, exist_ok=True)
        (subpkg / "__init__.py").write_text(SUBPKG_INIT)

    # Move files.
    for stem in GEMM_STEMS:
        exec_command(
            f"git mv python/sglang/srt/lora/trtllm_lora_temp/triton_ops/{stem}.py "
            f"python/sglang/kernels/ops/gemm/trtllm_lora_temp/{stem}.py",
            cwd=str(dir_root),
        )
    for stem in MOE_STEMS:
        exec_command(
            f"git mv python/sglang/srt/lora/trtllm_lora_temp/triton_ops/{stem}.py "
            f"python/sglang/kernels/ops/moe/trtllm_lora_temp/{stem}.py",
            cwd=str(dir_root),
        )

    exec_command(
        "git rm python/sglang/srt/lora/trtllm_lora_temp/triton_ops/__init__.py",
        cwd=str(dir_root),
    )

    for root_name in ["python/sglang", "test", "benchmark"]:
        root = dir_root / root_name
        for path in root.rglob("*.py"):
            text = path.read_text()
            new_text = rewrite_package_imports(text)
            new_text = rewrite_module_paths(new_text, mmap)
            if new_text != text:
                path.write_text(new_text)

    src = dir_root / "python/sglang/srt/lora/trtllm_lora_temp/triton_ops"
    if src.exists():
        pycache = src / "__pycache__"
        if pycache.exists():
            for f in pycache.iterdir():
                f.unlink()
            pycache.rmdir()
        if not any(src.iterdir()):
            src.rmdir()

    git_add_and_commit(
        "[Kernel] Migrate trtllm_lora_temp/triton_ops into "
        "sglang.kernels.ops.{gemm,moe}.trtllm_lora_temp",
        cwd=str(dir_root),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        repo_root = exec_command("git rev-parse --show-toplevel")
        transform(Path(repo_root))
    else:
        verify_mechanical_refactor(BASE_COMMIT, TARGET_COMMIT, transform)
