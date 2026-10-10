#!/usr/bin/env python3
"""Pre-build the AITER JIT modules that the AMD nightly suites need.

AITER compiles its kernel modules lazily, on first use. That is fine when the CI
image ships a matching prebuilt AITER, but the AITER Scout workflow forces a
source rebuild in every job (AITER_COMMIT_OVERRIDE), which empties the JIT cache.
The first request that reaches a cold module then blocks the scheduler loop for
as long as the compile takes -- measured at 363s for
module_gemm_a8w8_blockscale_bpreshuffle_cktile on gfx950, above the 300s
scheduler watchdog, which SIGQUITs the server mid-test.

Running this after an AITER rebuild moves that cost into the install step, where
no watchdog is running. Modules are built by name through the same code path as
AITER's compile_ops decorator, so no kernel arguments or tensor layouts are
needed here.

MODULES is the union of the modules observed in the AMD 8-GPU nightly jobs that
repeatedly time out under Scout (DeepSeek-V4 Flash/Pro, DeepSeek-R1 hicache and
MXFP4, Qwen3-235B MXFP4, GLM-5.1, MiniMax-M2.7). Unknown names are skipped, so
the list can lag behind AITER without breaking the job.
"""

import inspect
import os
import sys
import time

os.environ.setdefault("SGLANG_USE_AITER", "1")

MODULES = [
    "module_aiter_core",
    # Blockscale FP8 GEMM: the DeepSeek-V4 hang, and by far the costliest pair.
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
    "module_gemm_a8w8_blockscale_bpreshuffle",
    "module_gemm_a8w8_blockscale_bpreshuffle_asm",
    "module_gemm_a8w8_blockscale_cktile",
    "module_gemm_a16w16_asm",
    "module_gemm_common",
    # Attention / MLA.
    "module_mhc",
    "module_mla_asm",
    "module_mla_metadata",
    "module_mla_reduce",
    "module_pa_sparse_prefill_opus",
    "module_ps_metadata",
    # MoE.
    "module_moe_asm",
    "module_moe_fmoe_asm",
    "module_moe_opus",
    "module_moe_sorting_opus",
    "module_moe_topk",
    "module_moe_topksoftmax_asm",
    # Norm / rope / quant / elementwise.
    "module_activation",
    "module_cache",
    "module_custom",
    "module_custom_all_reduce",
    "module_deepgemm_opus",
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    "module_norm",
    "module_quant",
    "module_rmsnorm_quant",
    "module_rope_2c_cached_positions_fwd",
    "module_sample",
]


def is_built(core, md_name):
    """True if the module's .so is already on disk and valid for this arch.

    Checked on disk rather than via core.get_module(): the ASM modules are loaded
    through ctypes, so importing them as Python extensions raises even when they
    are built.
    """
    jit_dir = getattr(core, "this_dir", None)
    if jit_dir is None:
        try:
            core.get_module(md_name)
            return True
        except Exception:  # noqa: BLE001
            return False
    if not os.path.exists(os.path.join(jit_dir, f"{md_name}.so")):
        return False
    needs_arch_rebuild = getattr(core, "_needs_arch_rebuild", None)
    return not (needs_arch_rebuild is not None and needs_arch_rebuild(md_name))


def build(core, md_name):
    """Build one module by name. Returns 'cached', 'built' or an error string."""
    if is_built(core, md_name):
        return "cached"

    try:
        args = core.get_args_of_build(md_name)
    except Exception as e:  # noqa: BLE001
        return f"unknown module: {e}"

    # get_args_of_build returns a superset of build_module's parameters, and the
    # set has changed across AITER versions, so pass only what it accepts.
    accepted = inspect.signature(core.build_module).parameters
    kwargs = {k: v for k, v in args.items() if k in accepted and k != "md_name"}

    hip_clang_path = args.get("hip_clang_path")
    prev_hip_clang_path = None
    if hip_clang_path is not None and os.path.exists(hip_clang_path):
        prev_hip_clang_path = os.environ.get("HIP_CLANG_PATH")
        os.environ["HIP_CLANG_PATH"] = hip_clang_path
    try:
        core.build_module(md_name, **kwargs)
        return "built"
    except Exception as e:  # noqa: BLE001
        return f"build failed: {e}"
    finally:
        if hip_clang_path is not None and os.path.exists(hip_clang_path):
            if prev_hip_clang_path is None:
                os.environ.pop("HIP_CLANG_PATH", None)
            else:
                os.environ["HIP_CLANG_PATH"] = prev_hip_clang_path


def main():
    try:
        from aiter.jit import core
    except ImportError as e:
        print(f"AITER not importable ({e}); skipping warmup")
        return 0

    print(f"Warming up {len(MODULES)} AITER JIT modules")
    counts = {}
    start = time.perf_counter()
    for md_name in MODULES:
        t0 = time.perf_counter()
        status = build(core, md_name)
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        print(f"  {md_name}: {status} ({time.perf_counter() - t0:.1f}s)", flush=True)

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"AITER warmup finished in {time.perf_counter() - start:.1f}s ({summary})")
    # Never fail the job: a cold module only costs time later, and AITER module
    # names move around between releases.
    return 0


if __name__ == "__main__":
    sys.exit(main())
