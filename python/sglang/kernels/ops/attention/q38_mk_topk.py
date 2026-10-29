"""Exact M4 QSA selection: native short rows, cluster radix long rows.

One kernel is captured. Runtime lengths select the owner short path or the
full cluster long path. Native scratch is used only when the row fits4096.
"""
import os
from pathlib import Path

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args


@cache_once
def _module(items: int, zero_start: bool):
    pdl = is_arch_support_pdl()
    root = Path(__file__).resolve().parents[2] / "jit/csrc/attention"
    args = make_cpp_args(items, pdl, zero_start)
    long_args = make_cpp_args(items, 8, pdl, True, 32, True)
    flags = {"extra_cuda_cflags": ["-DQ38_PERF_SHORT_SCAN=1"]} if os.environ.get("SGLANG_Q38C_TOPK_SHORT_SCAN", "0") == "1" else {}
    if flags:
        import logging
        logging.getLogger(__name__).info("STACK8_TOPK_SHORT_SCAN_COMPILE items=%d zero_start=%s arch=%s", items, zero_start, torch.cuda.get_device_capability())
    return load_jit(
        "q38_mk_topk_single_launch", *args,
        cuda_files=[str(root / "q38_mk_topk.cuh")],
        cuda_wrappers=[("select_rows", f"q38_mk_topk<{long_args}>")],
        **flags,
    )


def topk_mk(scores, lengths, topk=512, row_starts=None):
    assert scores.dtype == torch.float32 and scores.stride(1) == 1 and topk == 512
    assert lengths.dtype == torch.int32
    assert row_starts is None or row_starts.dtype == torch.int32
    items = max(1, (scores.shape[1] + 4095) // 4096)
    output = torch.empty((scores.shape[0], 512), device=scores.device, dtype=torch.int32)
    module = _module(items, row_starts is None)
    module.select_rows(scores, lengths, row_starts, output)
    return output
