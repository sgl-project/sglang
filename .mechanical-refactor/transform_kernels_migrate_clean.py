#!/usr/bin/env python3
"""Reproducible transform: migrate scattered triton_ops kernels (clean cluster)
into the sglang.kernels.ops.<group> namespace (RFC #29630).

Clean cluster = all triton_ops dirs whose consumers use module-level imports:
  layers/attention/triton_ops  -> kvcache + attention
  mem_cache/triton_ops         -> kvcache + memory
  model_executor/triton_ops    -> attention
  layers/triton_ops            -> activation
  constrained/triton_ops       -> grammar
  speculative/triton_ops       -> speculative
Also deletes dead code models/triton_ops/deepseek_v4.py.
lora/triton_ops is handled separately (package-split imports + config dir).
trtllm_lora_temp is intentionally NOT touched (12 active external referrers).

Usage:
  python3 transform_kernels_migrate_clean.py apply     # apply to current repo
  python3 transform_kernels_migrate_clean.py           # verify reproducibility
"""
import os
import re
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from mechanical_refactor_verify_utils import (  # noqa: E402
    exec_command,
    git_add_and_commit,
    verify_mechanical_refactor,
)

BASE_COMMIT = "ddb63f09cd"
TARGET_COMMIT = "0e901d03895c37017b2a63e42b8bdedb324bc10d"

# group -> {old_dir_rel_to_srt: [module_stems]}
SPEC = {
    "kvcache": {
        "layers/attention/triton_ops": [
            "cache_ops",
            "kv_indices",
            "rope_cache",
            "trtllm_fp8_kv_kernel",
            "trtllm_mha_page_table",
            "aiter_unified_attention",
        ],
        "mem_cache/triton_ops": ["cache_move", "mla_buffer"],
    },
    "attention": {
        "layers/attention/triton_ops": [
            "decode_attention",
            "extend_attention",
            "prefill_attention",
            "merge_state",
            "metadata",
            "dsa_metadata",
            "rocm_mla_decode_rope",
            "verify_splitkv",
            "pad",
        ],
        "model_executor/triton_ops": ["position"],
    },
    "memory": {
        "mem_cache/triton_ops": ["allocator", "common", "virtual_slot"],
    },
    "activation": {
        "layers/triton_ops": ["softcap"],
    },
    "grammar": {
        "constrained/triton_ops": ["bitmask_ops", "token_filter_ops"],
    },
    "speculative": {
        "speculative/triton_ops": [
            "cache_locs",
            "dflash",
            "eagle",
            "fused_kv_materialize",
            "gather_spec_extras",
            "multi_layer_eagle",
            "spec_tree",
        ],
    },
}

# __init__.py files in emptied source dirs that must be removed.
INIT_REMOVE = [
    "python/sglang/srt/mem_cache/triton_ops/__init__.py",
    "python/sglang/srt/speculative/triton_ops/__init__.py",
]

DEAD_CODE = ["python/sglang/srt/models/triton_ops/deepseek_v4.py"]

# Consumers that import a *submodule by name* from the old package (not a
# symbol) — handled explicitly since the module map keys on dotted paths.
SPECIAL_REPLACEMENTS = [
    (
        "from sglang.srt.layers.attention.triton_ops import extend_attention",
        "from sglang.kernels.ops.attention import extend_attention",
    ),
    # kvcache group docstring described the pre-move location; update it.
    (
        "This group wraps the Triton ``reshape_and_cache`` launcher that currently lives\n"
        "under ``sglang.srt.layers.attention.triton_ops``. Only a thin wrapper is added\n"
        "here; physically moving the Triton source into this package is deferred to a\n"
        "later phase (RFC #29630) to keep this change low-risk.",
        "This group wraps the Triton ``reshape_and_cache`` launcher, whose implementation\n"
        "now lives in this package (``sglang.kernels.ops.kvcache.cache_ops``) after being\n"
        "migrated out of ``sglang.srt.layers.attention.triton_ops`` (RFC #29630).",
    ),
]


def build_maps():
    moves = []  # (old_rel, new_rel)
    module_map = {}  # old_module -> new_module
    for group, dirs in SPEC.items():
        for old_dir, stems in dirs.items():
            for stem in stems:
                old_rel = f"python/sglang/srt/{old_dir}/{stem}.py"
                new_rel = f"python/sglang/kernels/ops/{group}/{stem}.py"
                old_mod = "sglang.srt." + old_dir.replace("/", ".") + f".{stem}"
                new_mod = f"sglang.kernels.ops.{group}.{stem}"
                moves.append((old_rel, new_rel))
                module_map[old_mod] = new_mod
    return moves, module_map


def rewrite_text(text, module_map):
    # Longest keys first; only match when NOT followed by an identifier char so
    # e.g. "...triton_ops.metadata" never matches "...triton_ops.dsa_metadata".
    for old_mod in sorted(module_map, key=len, reverse=True):
        text = re.sub(
            re.escape(old_mod) + r"(?![A-Za-z0-9_])",
            module_map[old_mod],
            text,
        )
    for old, new in SPECIAL_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def transform(dir_root: Path) -> None:
    moves, module_map = build_maps()

    # Step 1: move files (git mv preserves rename detection).
    for old_rel, new_rel in moves:
        exec_command(f"git mv {old_rel} {new_rel}", cwd=str(dir_root))

    # Step 2: delete dead code + emptied package __init__ files.
    for rel in DEAD_CODE + INIT_REMOVE:
        exec_command(f"git rm {rel}", cwd=str(dir_root))

    # Step 3: rewrite all references across python/sglang, test/, benchmark/.
    roots = [
        dir_root / "python" / "sglang",
        dir_root / "test",
        dir_root / "benchmark",
    ]
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text()
            new_text = rewrite_text(text, module_map)
            if new_text != text:
                path.write_text(new_text)

    # Step 4: drop now-empty source dirs (git already untracked their files).
    for d in [
        "python/sglang/srt/layers/attention/triton_ops",
        "python/sglang/srt/mem_cache/triton_ops",
        "python/sglang/srt/model_executor/triton_ops",
        "python/sglang/srt/layers/triton_ops",
        "python/sglang/srt/constrained/triton_ops",
        "python/sglang/srt/speculative/triton_ops",
        "python/sglang/srt/models/triton_ops",
    ]:
        abs_d = dir_root / d
        if abs_d.exists():
            for junk in abs_d.rglob("*.pyc"):
                junk.unlink()
            pycache = abs_d / "__pycache__"
            if pycache.exists():
                for f in pycache.iterdir():
                    f.unlink()
                pycache.rmdir()
            if not any(abs_d.iterdir()):
                abs_d.rmdir()

    git_add_and_commit(
        "[Kernel] Migrate triton_ops kernels into sglang.kernels.ops.* (clean cluster)",
        cwd=str(dir_root),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        repo_root = exec_command("git rev-parse --show-toplevel")
        transform(Path(repo_root))
    else:
        verify_mechanical_refactor(BASE_COMMIT, TARGET_COMMIT, transform)
