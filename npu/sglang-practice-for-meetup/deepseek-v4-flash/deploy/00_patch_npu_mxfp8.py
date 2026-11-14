#!/usr/bin/env python3
"""Patch the meetup image's SGLang NPU FP8 path for DeepSeek-V4-Flash.

One defect, one fix:
  SGLang's Ascend-950 MXFP8 weight re-layout (`w8a8_mxfp8.py::_layout_npu_arch35_ue8m0_weights`)
  validates the checkpoint scale shape BEFORE flattening it. DeepSeek-V4's grouped `wo_a` projection
  stores block scales as [G, N/128, K/128] (e.g. (8, 8, 32)) while the routine expects the flattened
  2-D block grid (e.g. (64, 32)) — same values, same order, just a different view.

Idempotent. Usage:  python3 deploy/00_patch_npu_mxfp8.py
"""
import pathlib
import sys

CANDIDATES = [
    "/sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/quantization/w8a8_mxfp8.py",
    "/sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/quantization/linear_utils.py",
]
MARKER = "MEETUP_FLATTEN_GROUPED_SCALES"
ANCHOR = "    checkpoint_scale = layer.weight_scale_inv.data\n"
INSERT = ANCHOR + """    # MEETUP_FLATTEN_GROUPED_SCALES: DSv4 grouped wo_a keeps scales as [G, N/128, K/128];
    # the 950 MXFP8 layout wants the flattened 2-D block grid (identical values and order).
    try:
        _exp = expected_scale_shape
    except NameError:
        _exp = ((n_dim + block_n - 1) // block_n, (k_dim + block_k - 1) // block_k)
    if (tuple(checkpoint_scale.shape) != tuple(_exp)
            and checkpoint_scale.numel() == int(_exp[0]) * int(_exp[1])):
        checkpoint_scale = checkpoint_scale.reshape(tuple(_exp))
        print('[patch] flattened grouped scale to', tuple(_exp))
"""


def main() -> int:
    for cand in CANDIDATES:
        p = pathlib.Path(cand)
        if not p.exists():
            continue
        s = p.read_text()
        if MARKER in s:
            print(f"already patched: {p}")
            return 0
        if ANCHOR not in s:
            print(f"anchor not found in {p}; inspect manually around 'Unexpected UE8M0 scale shape'")
            continue
        p.write_text(s.replace(ANCHOR, INSERT, 1))
        print(f"patched: {p}")
        return 0
    print("could not locate the layout routine; is SGLang installed where expected?")
    return 2


if __name__ == "__main__":
    sys.exit(main())
