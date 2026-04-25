"""Bitwise equivalence check between flashinfer's _dequant_mxfp4_on_device and
sglang's locally-copied _dequant_mxfp4 (in w4a16_deepseek.py).

Neither file is imported as a module (the flashinfer test file has heavy deps,
the sglang file imports CUDA-specific bits). We textually extract the two
symbols from each source file and exec them into isolated namespaces.

Run (CPU-only, no CUDA required):
    uv run python sunrise/verify_dequant_mxfp4.py

Override reference path with:
    FLASHINFER_SUNRISE_TEST_FILE=/some/other/path python sunrise/verify_dequant_mxfp4.py
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Callable

import torch


_SCRIPT = Path(__file__).resolve()
_SGLANG_REPO_ROOT = _SCRIPT.parent.parent  # sunrise/ -> repo root
OUR_FILE = (
    _SGLANG_REPO_ROOT / "python/sglang/srt/layers/quantization/w4a16_deepseek.py"
)
# Default assumes the standard NDA workspace layout where flashinfer-sunrise
# sits as a sibling of the sglang worktrees directory
# (ws_nda/flashinfer-sunrise and ws_nda/worktrees/<this-repo>).
_DEFAULT_REF = (
    _SGLANG_REPO_ROOT.parent.parent
    / "flashinfer-sunrise/tests/moe/test_trtllm_cutlass_fused_moe.py"
)
REF_FILE = Path(os.environ.get("FLASHINFER_SUNRISE_TEST_FILE", str(_DEFAULT_REF)))


def _extract_symbols(path: Path, lut_name: str, fn_name: str) -> dict:
    """Parse `path` with ast, pull out the `lut_name` Assign and the `fn_name`
    FunctionDef, exec them in an isolated namespace, and return the namespace."""
    source = path.read_text()
    tree = ast.parse(source)
    wanted_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == lut_name:
                    wanted_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == fn_name:
            wanted_nodes.append(node)
    if len(wanted_nodes) != 2:
        raise RuntimeError(
            f"Expected to find 1 Assign for {lut_name!r} and 1 FunctionDef for {fn_name!r} in {path}; "
            f"got {len(wanted_nodes)} matching nodes."
        )
    module = ast.Module(body=wanted_nodes, type_ignores=[])
    code = compile(module, filename=str(path), mode="exec")
    ns: dict = {"torch": torch}
    exec(code, ns)
    return ns


def main() -> int:
    print(f"REF: {REF_FILE}")
    print(f"OUR: {OUR_FILE}")
    if not REF_FILE.exists():
        print(f"ERROR: reference file not found: {REF_FILE}", file=sys.stderr)
        return 2
    if not OUR_FILE.exists():
        print(f"ERROR: local file not found: {OUR_FILE}", file=sys.stderr)
        return 2

    ref_ns = _extract_symbols(REF_FILE, "_MXFP4_LUT", "_dequant_mxfp4_on_device")
    our_ns = _extract_symbols(OUR_FILE, "_MXFP4_LUT", "_dequant_mxfp4")

    ref_lut = ref_ns["_MXFP4_LUT"]
    our_lut = our_ns["_MXFP4_LUT"]
    assert ref_lut == our_lut, f"LUT mismatch: ref={ref_lut} our={our_lut}"
    print(f"LUT equal: {ref_lut == our_lut} (len={len(ref_lut)})", flush=True)

    ref_fn: Callable = ref_ns["_dequant_mxfp4_on_device"]
    our_fn: Callable = our_ns["_dequant_mxfp4"]

    torch.manual_seed(0)

    # Shapes: last dim = K/2 (so K = 2*last_dim); K must be divisible by 32 for
    # the UE8M0 scale derivation. The (4, 1024, 2048) shape is a DSv4-realistic
    # slice (K=4096 = DSv4 hidden dim); full (256, 4096, 2048) OOMs a laptop
    # during fp32 intermediate allocation.
    shapes = [
        (2, 2, 16),           # minimal: K=32
        (4, 8, 32),           # K=64
        (8, 256, 64),         # K=128
        (256, 64, 256),       # full e=256, K=512
        (4, 1024, 2048),      # DSv4-shaped slice: K=4096
    ]

    all_ok = True
    mismatches: list[tuple] = []
    for shape in shapes:
        K_half = shape[-1]
        K = 2 * K_half
        assert K % 32 == 0, f"K={K} not divisible by 32 for shape {shape}"
        scale_shape = shape[:-1] + (K // 32,)

        w_fp4 = torch.randint(0, 256, shape, dtype=torch.uint8)
        # Scale UE8M0 byte 255 produces exp2(128)=inf, and inf*0 (FP4 zero
        # nibbles) → NaN, which torch.equal treats as unequal-to-itself. Cap
        # to <255 in this loop; NaN-position agreement is verified separately
        # below.
        w_scale = torch.randint(0, 255, scale_shape, dtype=torch.uint8)

        ref_out = ref_fn(w_fp4, w_scale)
        our_out = our_fn(w_fp4, w_scale_ue8m0_u8=w_scale)

        ok = torch.equal(ref_out, our_out)
        assert not ref_out.isnan().any(), "ref_out has NaN — scale cap failed"
        assert not our_out.isnan().any(), "our_out has NaN — scale cap failed"
        if not ok:
            diff = (ref_out.float() - our_out.float()).abs()
            num_mismatch = int((ref_out != our_out).sum().item())
            max_abs_diff = float(diff.max().item())
            mismatches.append((shape, num_mismatch, max_abs_diff))
            all_ok = False
            print(
                f"MISMATCH shape={shape} numel={ref_out.numel()} "
                f"num_mismatch={num_mismatch} max_abs_diff={max_abs_diff} "
                f"ref_dtype={ref_out.dtype} our_dtype={our_out.dtype} "
                f"ref_shape={tuple(ref_out.shape)} our_shape={tuple(our_out.shape)}"
            )
        else:
            print(
                f"OK shape={shape} numel={ref_out.numel()} "
                f"dtype={ref_out.dtype} out_shape={tuple(ref_out.shape)}",
                flush=True,
            )
        del ref_out, our_out, w_fp4, w_scale

    # NaN-edge: scale=255 path (inf*0 → NaN). Both fns should produce NaN at
    # identical positions and identical finite values elsewhere.
    print("--- NaN-edge shape (scale includes 255) ---", flush=True)
    shape_nan = (4, 8, 32)
    scale_shape_nan = shape_nan[:-1] + (shape_nan[-1] * 2 // 32,)
    w_fp4 = torch.randint(0, 256, shape_nan, dtype=torch.uint8)
    w_scale = torch.randint(0, 256, scale_shape_nan, dtype=torch.uint8)
    ref_out = ref_fn(w_fp4, w_scale)
    our_out = our_fn(w_fp4, w_scale_ue8m0_u8=w_scale)
    both_nan = ref_out.isnan() & our_out.isnan()
    eq_or_both_nan = (ref_out == our_out) | both_nan
    nan_ok = bool(eq_or_both_nan.all().item()) and torch.equal(
        ref_out.isnan(), our_out.isnan()
    )
    print(
        f"NaN-edge agree (incl NaN positions): {nan_ok} "
        f"(ref_nans={int(ref_out.isnan().sum())}, our_nans={int(our_out.isnan().sum())})",
        flush=True,
    )
    if not nan_ok:
        all_ok = False

    if all_ok:
        print("ALL SHAPES BITWISE EQUAL")
        return 0
    else:
        print(f"FAIL: {len(mismatches)} shape(s) mismatched: {mismatches}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
