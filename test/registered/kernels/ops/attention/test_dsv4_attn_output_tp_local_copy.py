"""DeepSeek-V4 TP-local head copy of the mixed-step attention output
(SGLANG_DSV4_ATTN_OUTPUT_TP_LOCAL_COPY=1).

The segment buffer's real heads [0, n_real) must be bitwise identical after the
TP-local copy and after the full copy, for the prefill head rows and the verify
tail rows. The model's consumer slice is `o[:, 0:n_local_heads, :]` on every TP
rank (deepseek_v4.py: `tp_slice = slice(0, self.n_local_heads)`), so the same check
covers all ranks. This tests only the copy kernel; that the model never reads the
skipped heads rests on that inline slice.
"""

import sys

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import copy_tp_local_heads
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

dev = "cuda"
ROWS_PRE, ROWS_VER, H_PAD, H_REAL, DV = 3840, 120, 64, 32, 512
N_TOTAL = ROWS_PRE + ROWS_VER
TP_SLICE = slice(0, H_REAL)


def make_inputs():
    torch.manual_seed(0)
    o_head = torch.randn(ROWS_PRE, H_PAD, DV, device=dev, dtype=torch.bfloat16)
    # decode kernel layout [b, 1, H, DV], reshaped as the backend does
    o_tail = torch.randn(ROWS_VER, 1, H_PAD, DV, device=dev, dtype=torch.bfloat16)
    tail = o_tail.reshape(ROWS_VER, H_PAD, DV)
    return o_head, tail


def test_tp_local_heads_bitwise():
    o_head, tail = make_inputs()
    out_full = torch.empty(4096, H_PAD, DV, device=dev, dtype=torch.bfloat16)[:N_TOTAL]
    out_half = torch.empty(4096, H_PAD, DV, device=dev, dtype=torch.bfloat16)[:N_TOTAL]

    out_full[:ROWS_PRE].copy_(o_head)
    out_full[ROWS_PRE:].copy_(tail)
    copy_tp_local_heads(out_half[:ROWS_PRE], o_head, H_REAL)
    copy_tp_local_heads(out_half[ROWS_PRE:], tail, H_REAL)

    assert torch.equal(out_full[:, :H_REAL], out_half[:, :H_REAL])
    assert torch.equal(out_half[:ROWS_PRE, :H_REAL], o_head[:, :H_REAL])
    assert torch.equal(out_half[ROWS_PRE:, :H_REAL], tail[:, :H_REAL])
    # the model's consumer view: o[:, tp_slice, :]
    assert torch.equal(out_full[:, TP_SLICE, :], out_half[:, TP_SLICE, :])


def test_stale_heads_never_read():
    # Poison the heads the copy does not write; the TP-local slice the consumer reads
    # must be NaN-free and bitwise equal to the source.
    o_head, tail = make_inputs()
    src = torch.cat([o_head, tail], dim=0)
    dst = torch.full((4096, H_PAD, DV), float("nan"), device=dev, dtype=torch.bfloat16)[
        :N_TOTAL
    ]
    copy_tp_local_heads(dst[:ROWS_PRE], o_head, H_REAL)
    copy_tp_local_heads(dst[ROWS_PRE:], tail, H_REAL)

    # the consumer slice is inline in deepseek_v4.py (`o = o[:, tp_slice, :]`)
    o = dst[:, TP_SLICE, :]
    assert not o.isnan().any()
    assert torch.equal(o, src[:, TP_SLICE, :])
    assert dst[:, H_REAL:].isnan().all()  # the skipped heads really are stale


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
