#!/usr/bin/env python3
"""Reproducible transform: migrate lora/triton_ops kernels into
sglang.kernels.ops.{gemm,moe} (RFC #29630).

lora is separated from the "clean cluster" because its consumers import symbols
from the package __init__ re-export (splitting across gemm/moe), and because the
csgmv_configs tuning-config dir is looked up via __file__ and must move with
lora_tuning_config.py.

Usage:
  python3 transform_kernels_migrate_lora.py apply
  python3 transform_kernels_migrate_lora.py           # verify reproducibility
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

BASE_COMMIT = "595d2c6ac8"  # clean-cluster migration commit
TARGET_COMMIT = "ff2845953f79f121d7e456851f4cd55e9cd98b81"

GEMM_STEMS = [
    "chunked_embedding_lora_a",
    "chunked_sgmv_expand",
    "chunked_sgmv_shrink",
    "embedding_lora_a",
    "gate_up_lora_b",
    "kernel_utils",
    "kv_b_lora_absorbed",
    "lora_tuning_config",
    "qkv_lora_b",
    "sgemm_lora_a",
    "sgemm_lora_b",
]
MOE_STEMS = ["fused_moe_lora_kernel", "virtual_experts"]

OLD_PKG = "sglang.srt.lora.triton_ops"

# Public symbol -> new module path (for the package-import re-export splits).
SYMBOL_MODULE = {
    "chunked_embedding_lora_a_forward": "sglang.kernels.ops.gemm.chunked_embedding_lora_a",
    "chunked_sgmv_lora_expand_forward": "sglang.kernels.ops.gemm.chunked_sgmv_expand",
    "chunked_sgmv_lora_shrink_forward": "sglang.kernels.ops.gemm.chunked_sgmv_shrink",
    "embedding_lora_a_fwd": "sglang.kernels.ops.gemm.embedding_lora_a",
    "gate_up_lora_b_fwd": "sglang.kernels.ops.gemm.gate_up_lora_b",
    "qkv_lora_b_fwd": "sglang.kernels.ops.gemm.qkv_lora_b",
    "sgemm_lora_a_fwd": "sglang.kernels.ops.gemm.sgemm_lora_a",
    "sgemm_lora_b_fwd": "sglang.kernels.ops.gemm.sgemm_lora_b",
    "step_a_q_fwd": "sglang.kernels.ops.gemm.kv_b_lora_absorbed",
    "step_b_q_fwd": "sglang.kernels.ops.gemm.kv_b_lora_absorbed",
    "step_a_v_fwd": "sglang.kernels.ops.gemm.kv_b_lora_absorbed",
    "step_b_v_fwd": "sglang.kernels.ops.gemm.kv_b_lora_absorbed",
    "fused_moe_lora": "sglang.kernels.ops.moe.fused_moe_lora_kernel",
    "merged_experts_fused_moe_lora_add": "sglang.kernels.ops.moe.virtual_experts",
}
# Names that are submodules of the old package (import as `from PARENT import name`).
SUBMODULE_PARENT = {"lora_tuning_config": "sglang.kernels.ops.gemm"}


def module_map():
    m = {}
    for stem in GEMM_STEMS:
        m[f"{OLD_PKG}.{stem}"] = f"sglang.kernels.ops.gemm.{stem}"
    for stem in MOE_STEMS:
        m[f"{OLD_PKG}.{stem}"] = f"sglang.kernels.ops.moe.{stem}"
    return m


def rewrite_package_imports(text):
    """Rewrite `from sglang.srt.lora.triton_ops import ...` (the __init__
    re-export) into explicit per-module imports, preserving indentation."""

    def emit(indent, names):
        by_module = {}
        extra_lines = []
        for name in names:
            if name in SUBMODULE_PARENT:
                extra_lines.append(
                    f"{indent}from {SUBMODULE_PARENT[name]} import {name}"
                )
            else:
                mod = SYMBOL_MODULE[name]
                by_module.setdefault(mod, []).append(name)
        lines = []
        for mod in sorted(by_module):
            syms = ", ".join(sorted(by_module[mod]))
            lines.append(f"{indent}from {mod} import {syms}")
        lines.extend(extra_lines)
        return "\n".join(lines)

    def parse_names(blob):
        return [n.strip() for n in blob.replace("\n", " ").split(",") if n.strip()]

    # Multi-line parenthesized form.
    def paren_sub(match):
        indent, blob = match.group(1), match.group(2)
        return emit(indent, parse_names(blob))

    text = re.sub(
        r"(?m)^([ \t]*)from "
        + re.escape(OLD_PKG)
        + r" import \(([^)]*)\)",
        paren_sub,
        text,
    )

    # Single-line form (no parentheses).
    def line_sub(match):
        indent, blob = match.group(1), match.group(2)
        return emit(indent, parse_names(blob))

    text = re.sub(
        r"(?m)^([ \t]*)from " + re.escape(OLD_PKG) + r" import ([^\(\n]+)$",
        line_sub,
        text,
    )
    return text


def rewrite_module_paths(text, mmap):
    for old_mod in sorted(mmap, key=len, reverse=True):
        text = re.sub(
            re.escape(old_mod) + r"(?![A-Za-z0-9_])", mmap[old_mod], text
        )
    return text


# Benchmark tune script builds the config output dir from path components.
BENCH_PATH_OLD = '        "srt",\n        "lora",\n        "triton_ops",'
BENCH_PATH_NEW = '        "kernels",\n        "ops",\n        "gemm",'


def transform(dir_root: Path) -> None:
    mmap = module_map()

    # Step 1: move .py files.
    for stem in GEMM_STEMS:
        exec_command(
            f"git mv python/sglang/srt/lora/triton_ops/{stem}.py "
            f"python/sglang/kernels/ops/gemm/{stem}.py",
            cwd=str(dir_root),
        )
    for stem in MOE_STEMS:
        exec_command(
            f"git mv python/sglang/srt/lora/triton_ops/{stem}.py "
            f"python/sglang/kernels/ops/moe/{stem}.py",
            cwd=str(dir_root),
        )

    # Step 2: move the tuning-config dir (looked up via __file__).
    exec_command(
        "git mv python/sglang/srt/lora/triton_ops/csgmv_configs "
        "python/sglang/kernels/ops/gemm/csgmv_configs",
        cwd=str(dir_root),
    )

    # Step 3: delete the now-stale package __init__ (re-export shim).
    exec_command(
        "git rm python/sglang/srt/lora/triton_ops/__init__.py", cwd=str(dir_root)
    )

    # Step 4: rewrite references across python/sglang, test/, benchmark/.
    for root_name in ["python/sglang", "test", "benchmark"]:
        root = dir_root / root_name
        for path in root.rglob("*.py"):
            text = path.read_text()
            new_text = rewrite_package_imports(text)
            new_text = rewrite_module_paths(new_text, mmap)
            new_text = new_text.replace(BENCH_PATH_OLD, BENCH_PATH_NEW)
            new_text = new_text.replace(
                "python/sglang/srt/lora/triton_ops/",
                "python/sglang/kernels/ops/gemm/",
            )
            if new_text != text:
                path.write_text(new_text)

    # Step 5: drop the now-empty source dir.
    src = dir_root / "python/sglang/srt/lora/triton_ops"
    if src.exists():
        for junk in src.rglob("*.pyc"):
            junk.unlink()
        pycache = src / "__pycache__"
        if pycache.exists():
            for f in pycache.iterdir():
                f.unlink()
            pycache.rmdir()
        if not any(src.iterdir()):
            src.rmdir()

    git_add_and_commit(
        "[Kernel] Migrate lora/triton_ops kernels into sglang.kernels.ops.{gemm,moe}",
        cwd=str(dir_root),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        repo_root = exec_command("git rev-parse --show-toplevel")
        transform(Path(repo_root))
    else:
        verify_mechanical_refactor(BASE_COMMIT, TARGET_COMMIT, transform)
