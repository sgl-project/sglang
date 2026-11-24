"""HC mix up projection fused with its sigmoid gated mean for prefill row counts (TP1, hc 4, hidden 2560, rank 320).

``out[m, j] = bf16(sum_g sigmoid(bf16(logit[m, 4 j + g])) * x[m, g * 2560 + j] / 4)``, ``logit = t @ w_up_perm.T``
in FP32 over K ascending: the torch.compile chain's GEMM2 + sigmoid * mean, without writing and re-reading the
[rows, 10240] logits. A persistent tcgen05 kernel (hc_up_gate_tc_src): a TMA warp (t and W_up per K step, the
tile's normalized streams into one of two shared-memory buffers), an MMA warp (two TMEM accumulators) and 16
gated-mean warps. MODE_BITWISE (default) is bitwise to the compiled chain; MODE_TANH takes the sigmoid as
0.5 tanh(x / 2) + 0.5 (no faster on GB300, kept for comparison).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HC, HS, LOWRANK = 4, 2560, 320
# Kernel modes: 0 bitwise to the compiled chain (Triton's sigmoid); MODE_TANH the sigmoid as 0.5 tanh(x / 2) + 0.5.
MODE_BITWISE = 0
MODE_TANH = 4


def _source_path() -> str:
    import hashlib
    import os

    from sglang.kernels.jit.utils.compile.cache import cache_root
    from sglang.kernels.ops.gemm.hc_up_gate_tc_src import CUDA_SOURCE

    data = CUDA_SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = cache_root() / "inline-src"
    path = directory / f"hc_up_gate_tc-{digest}.cuh"
    if not path.is_file() or path.read_bytes() != data:
        directory.mkdir(parents=True, exist_ok=True)
        staged = directory / f".{path.name}.{os.getpid()}.tmp"
        staged.write_bytes(data)
        os.replace(staged, path)
    return str(path)


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "hc_up_gate_tc",
        cuda_files=[_source_path()],
        cuda_wrappers=[(f"run{m}", f"HcUpGateTcKernel<{m}>::run") for m in (MODE_BITWISE, MODE_TANH)],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


_NUM_SMS: dict = {}


def warmup() -> None:
    _jit_module()


def hc_up_gate_tc(
    t: torch.Tensor,
    w_up_permuted: torch.Tensor,
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    mode: int = 0,
    num_ctas: Optional[int] = None,
) -> torch.Tensor:
    """``t`` [rows, 320] contiguous bf16 (silu'd low-rank activations), ``w_up_permuted`` [10240, >= 320]
    (hc_mix_splitk.permute_pad_up_weight's layout, rows j*4 + g), ``x`` [rows, 10240] the normalized streams."""
    rows = x.shape[0]
    if out is None:
        out = torch.empty((rows, HS), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    if num_ctas is None:
        dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
        num_ctas = _NUM_SMS.get(dev)
        if num_ctas is None:
            num_ctas = _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    module = _jit_module()
    run = module.run4 if mode == MODE_TANH else module.run0
    run(t, w_up_permuted, x, out, int(num_ctas))
    return out
