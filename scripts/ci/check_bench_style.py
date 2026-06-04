#!/usr/bin/env python3
"""
Pre-commit hook: enforce the unified `marker` benchmark style for JIT kernel
benchmarks under python/sglang/jit_kernel/benchmark/.

Rules (AST-based, no import needed — runs without a GPU):
  1. No hand-rolled timing: forbid triton.testing.{do_bench,do_bench_cudagraph,
     perf_report,Benchmark} and time.perf_counter()/time.time() timing loops.
     Time kernels through `marker.do_bench` instead.
  2. Every `@marker.kernel(...)` must declare correctness intent: either
     `reference=<impl>` (compared via torch.testing.assert_close) or an explicit
     `correctness=False, reason="..."` opt-out.

LEGACY_ALLOWLIST holds files predating this style. It is a ratchet: it may only
shrink on whole-tree runs. New bench files are not exempt; migrate a file and
drop it from the list.
"""

import ast
import os
import sys

BENCH_ROOT = "python/sglang/jit_kernel/benchmark"

# Files predating the unified marker style. Migrate to the class contract and
# remove from this list — never add to it.
LEGACY_ALLOWLIST = {
    "bench_add_constant.py",
    "bench_awq_dequantize.py",
    "bench_clamp_position.py",
    "bench_concat_mla.py",
    "bench_custom_all_reduce.py",
    "bench_dsv4_fp4_indexer.py",
    "bench_fused_qknorm_rope.py",
    "bench_hadamard.py",
    "bench_hicache.py",
    "bench_hisparse.py",
    "bench_mla_kv_pack_quantize_fp8.py",
    "bench_mxfp8_moe.py",
    "bench_ngram_compute_decode.py",
    "bench_norm.py",
    "bench_nvfp4_blockwise_moe.py",
    "bench_nvfp4_quant.py",
    "bench_nvfp4_scaled_mm.py",
    "bench_per_tensor_quant_fp8.py",
    "bench_per_token_group_quant_8bit.py",
    "bench_qknorm_across_heads.py",
    "bench_renorm.py",
    "bench_resolve_future_token_ids.py",
    "bench_rope.py",
    "bench_set_mla_kv_buffer.py",
    "bench_tp_qknorm.py",
    "diffusion/bench_diffusion_nvfp4_scaled_mm.py",
    "diffusion/bench_fused_norm_scale_shift.py",
    "diffusion/bench_group_norm_silu.py",
    "diffusion/bench_norm_impls.py",
    "diffusion/bench_qknorm_rope.py",
    "diffusion/bench_qwen_image_modulation.py",
    "kv_canary/bench_plan.py",
    "kv_canary/bench_scatter_req_token_ids.py",
    "kv_canary/bench_verify.py",
    "kv_canary/bench_write.py",
}

_FORBIDDEN_TRITON = {"do_bench", "do_bench_cudagraph", "perf_report", "Benchmark"}
_FORBIDDEN_TIME = {"perf_counter", "time"}


def _attr_chain(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _check_file(path: str) -> list:
    errors = []
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    imports_marker = False
    has_kernel = False

    for node in ast.walk(tree):
        # rule 1: forbidden timing primitives
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain = _attr_chain(node.func)
            if (
                chain.startswith("triton.testing.")
                and node.func.attr in _FORBIDDEN_TRITON
            ):
                errors.append(
                    f"  L{node.lineno}: uses {chain}; time kernels via marker.do_bench"
                )
            if chain in ("time.perf_counter", "time.time"):
                errors.append(
                    f"  L{node.lineno}: uses {chain} for timing; use marker.do_bench"
                )

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = " ".join(a.name for a in node.names)
            if "marker" in mod or "marker" in names:
                imports_marker = True

        # rule 2: @marker.kernel(...) correctness contract
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if not (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "kernel"
                ):
                    continue
                has_kernel = True
                ref = _kw(dec, "reference")
                corr = _kw(dec, "correctness")
                reason = _kw(dec, "reason")
                opt_out = isinstance(corr, ast.Constant) and corr.value is False
                if opt_out:
                    if not (isinstance(reason, ast.Constant) and reason.value):
                        errors.append(
                            f"  L{dec.lineno}: @marker.kernel on {node.name} has "
                            f"correctness=False but no reason=..."
                        )
                elif ref is None:
                    errors.append(
                        f"  L{dec.lineno}: @marker.kernel on {node.name} must set "
                        f"reference=<impl> (or correctness=False, reason=...)"
                    )

    if not imports_marker:
        errors.append(
            "  does not import marker; benchmarks must use the marker harness"
        )
    _ = has_kernel  # informational; function-form benches are still permitted
    return errors


def _is_bench_file(path: str) -> bool:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(BENCH_ROOT)).replace(
        os.sep, "/"
    )
    return (
        not rel.startswith("../")
        and not os.path.isabs(rel)
        and os.path.basename(path).startswith("bench_")
        and path.endswith(".py")
    )


def _files_from_args(args: list[str]) -> list[str]:
    return sorted(
        path for path in args if os.path.isfile(path) and _is_bench_file(path)
    )


def _all_bench_files() -> list[str]:
    files = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(BENCH_ROOT)
        for name in names
        if name.startswith("bench_") and name.endswith(".py")
    )
    return files


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    files = _files_from_args(argv) if argv else _all_bench_files()
    check_stale_allowlist = not argv
    failed = {}
    stale_allowlist = []

    for path in files:
        rel = os.path.relpath(path, BENCH_ROOT)
        try:
            errors = _check_file(path)
        except SyntaxError as e:
            failed[path] = [f"  syntax error: {e}"]
            continue
        if rel in LEGACY_ALLOWLIST:
            if check_stale_allowlist and not errors:
                stale_allowlist.append(rel)
            continue
        if errors:
            failed[path] = errors

    if failed:
        print(
            "ERROR: benchmark style violations (see scripts/ci/check_bench_style.py):"
        )
        for path, errs in failed.items():
            print(f"\n{path}")
            print("\n".join(errs))
        print()

    if stale_allowlist:
        print(
            "NOTE: these files now conform — remove them from LEGACY_ALLOWLIST "
            "in scripts/ci/check_bench_style.py (the ratchet only shrinks):"
        )
        for rel in stale_allowlist:
            print(f"  {rel}")
        print()

    return 1 if (failed or stale_allowlist) else 0


if __name__ == "__main__":
    sys.exit(main())
