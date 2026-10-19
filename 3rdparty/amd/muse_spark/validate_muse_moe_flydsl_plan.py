#!/usr/bin/env python3
"""CPU-only structural checks for the Muse MoE FlyDSL performance plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

SOURCE = Path("sglang_meta_common/sglang_meta_common/ops/muse_moe_flydsl.py")
SELECTOR_NAMES = {
    "_SMALL_M",
    "_MID_M",
    "_LARGE_M",
    "_SMALL_M_FP4",
    "_MID_M_FP4",
    "_LARGE_M_FP4",
    "_pick_flydsl_config",
}


def selector_from_text(text: str):
    tree = ast.parse(text)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in SELECTOR_NAMES for t in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in SELECTOR_NAMES:
            selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_pick_flydsl_config"]


def git_text(revision: str) -> str:
    return subprocess.run(
        ["git", "show", f"{revision}:{SOURCE.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def lds_bytes(
    dtype: str, block_m: int, block_n: int, block_k: int, halves: int, stages: int
) -> dict[str, int]:
    activation = stages * block_m * block_k * 2
    if dtype == "mxfp4":
        weight = stages * halves * block_n * block_k // 2
        scales = stages * halves * block_n * block_k // 32
    else:
        weight = stages * halves * block_n * block_k * 2
        scales = 0
    return {
        "activation": activation,
        "weight": weight,
        "scales": scales,
        "total": activation + weight + scales,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="a29522f")
    parser.add_argument("--pre-change", default="a29522f^")
    parser.add_argument("--emit-manifest", action="store_true")
    args = parser.parse_args()
    current_text = SOURCE.read_text()
    compile(current_text, str(SOURCE), "exec")
    current, baseline = (
        selector_from_text(current_text),
        selector_from_text(git_text(args.baseline)),
    )
    pre_change = selector_from_text(git_text(args.pre_change))

    for m in (4, 8, 16, 32, 64, 128):
        for mxfp4 in (False, True):
            for topk, experts in ((2, 8), (8, 32)):
                actual = current(m, topk, experts, mxfp4)
                expected = baseline(m, topk, experts, mxfp4)
                if mxfp4 or m > 16:
                    assert actual == expected, (m, mxfp4, expected, actual)
                    assert actual == pre_change(m, topk, experts, mxfp4), (
                        m,
                        mxfp4,
                        actual,
                    )
                elif m <= 4:
                    assert actual["BLOCK_N"] == 16
                    assert actual["FLYDSL_K_UNROLL"] == 4
                    assert actual["FLYDSL_SPLIT_K"] == 4
                else:
                    assert actual["BLOCK_N"] == 64 and actual["num_warps"] == 4
                    assert actual["GEMM1_FLYDSL_K_UNROLL"] == 2
                    assert actual["GEMM2_FLYDSL_K_UNROLL"] == 4
                assert "GEMM2_BLOCK_N" not in actual, (m, actual)
    for m in (256, 512, 1024):
        cfg = current(m, 2, 8, True)
        assert cfg["GEMM2_BLOCK_N"] == 256, (m, cfg)
        assert cfg["FLYDSL_K_UNROLL"] == 16 and cfg["num_warps"] == 2, (m, cfg)
    for m in (129, 1024):
        assert current(m, 8, 32, True).get("GEMM2_BLOCK_N") == 256
        assert "GEMM2_BLOCK_N" not in current(m, 8, 32, False)
    assert "GEMM2_BLOCK_N" not in current(1025, 8, 32, True)

    status_stream = sys.stderr if args.emit_manifest else sys.stdout
    print(
        f"selector: MXFP4 boundaries match {args.baseline} and {args.pre_change}; BF16 M=4 split-K and M=8/16 candidates checked",
        file=status_stream,
    )
    print(
        "selector: MXFP4 M=256/512/1024 uses GEMM2 BN256, unroll16, warps2",
        file=status_stream,
    )
    for dtype in ("bf16", "mxfp4"):
        for stage, bn, halves in (("gemm1", 128, 2), ("gemm2", 256, 1)):
            sizes = [
                lds_bytes(dtype, 16, bn, bk, halves, 2)["total"] // 1024
                for bk in (64, 128, 512)
            ]
            print(
                f"{dtype} {stage} BN={bn}: BK64/128/512 = {sizes} KiB",
                file=status_stream,
            )
    if args.emit_manifest:
        base = [
            "python",
            "/home/yichiche/flydsl-work/bench_muse_moe_flydsl.py",
            "--rows",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
            "256",
            "--experts",
            "32",
        ]
        manifest = []
        for dtype in ("bf16", "mxfp4"):
            for local_slots in (1, 2, 4, 8):
                common = base + ["--dtype", dtype, "--local-slots", str(local_slots)]
                if dtype == "mxfp4":
                    common += ["--random-scales"]
                manifest.append(
                    {
                        "format": dtype,
                        "stage": "control",
                        "local_slots": local_slots,
                        "argv": common,
                    }
                )
                if dtype == "bf16":
                    for stage in ("gemm1", "gemm2"):
                        for bn in (32, 64, 128):
                            argv = common + [
                                f"--{stage}-block-n",
                                str(bn),
                                f"--{stage}-num-warps",
                                "4",
                            ]
                            manifest.append(
                                {
                                    "format": dtype,
                                    "stage": stage,
                                    "local_slots": local_slots,
                                    "candidate": {"block_n": bn, "warps": 4},
                                    "argv": argv,
                                }
                            )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
