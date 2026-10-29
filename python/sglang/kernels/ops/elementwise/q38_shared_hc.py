# SPDX-License-Identifier: Apache-2.0
"""FN BF16 shared-output push into the HC consumer (Q35 hop/push adaptation).

Keep the shared MLP and routed expert kernels native. Preserve the intermediate
BF16 rounding after shared+routed, then the native HC partial-dot order.
"""
import logging
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

log = logging.getLogger(__name__)
_seen = set()

@triton.jit
def _shared_hc(H, W, S, R, RES, P, O, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    x = tl.arange(0, B)
    h = tl.load(H + row * D + x, x < D, 0).to(tl.float32)
    w = tl.load(W + x, x < D, 0).to(tl.float32)
    # Same shape/warps/reduction as native fused_gate_sigmoid_mul_add.
    sg = tl.sigmoid(tl.sum(h * w, 0))
    s = tl.load(S + row * D + x, x < D, 0).to(tl.float32)
    r = tl.load(R + row * D + x, x < D, 0).to(tl.float32)
    # Native gate/add contracts to FMA and stores BF16 before HC reads it.
    block = tl.fma(sg, s, r).to(tl.bfloat16).to(tl.float32)
    total = tl.full((), 0, tl.float32)
    for k in tl.static_range(8):
        total = total + tl.load(P + (row * 8 + k) * 4 + branch)
    gate = tl.div_rn(2.0, 1.0 + libdevice.exp(-total / 4.0))
    res = tl.load(RES + (row * 4 + branch) * D + x, x < D, 0).to(tl.float32)
    y = tl.fma(gate, block, res)
    tl.store(O + (row * 4 + branch) * D + x, y, x < D)


def shared_hc(hidden, weight, shared, routed, residual, partials, layer_id):
    rows, d = hidden.shape
    assert d == 2560 and residual.shape == (rows, 10240)
    assert all(t.dtype == torch.bfloat16 for t in (hidden, weight, shared, routed, residual))
    out = torch.empty_like(residual)
    _shared_hc[(rows, 4)](hidden, weight, shared, routed, residual, partials, out,
                          D=d, B=4096, num_warps=16, enable_fp_fusion=True)
    key = (layer_id, rows)
    if key not in _seen:
        _seen.add(key)
        log.info('Q38_SHARED_OUT_V3_ENGAGED layer=%s rows=%s BF16 shared+route->HC; native GEMMs', layer_id, rows)
    return out
