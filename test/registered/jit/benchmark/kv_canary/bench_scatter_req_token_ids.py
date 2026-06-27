from __future__ import annotations

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.kv_canary.scatter_req_token_ids import (
    launch_scatter_req_token_ids_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)
# AMD mirrors the CUDA nightly registration (nightly-only, no per-PR suite).
# Note: amd_ci_exec.sh sets SGLANG_IS_IN_CI, so this runs the CI-reduced range
# (_BS_AXIS_CI/_SEQ_LEN_AXIS_CI via get_benchmark_range), same as CUDA nightly.
register_amd_ci(est_time=180, suite="nightly-amd-kernel-1-gpu", nightly=True)


_BS_SEQ_FULL: list[tuple[int, int]] = [
    (bs, seq_len) for bs in (1, 8, 64, 256) for seq_len in (128, 512, 2048, 8192)
]
_BS_SEQ_CI: list[tuple[int, int]] = [
    (bs, seq_len) for bs in (1, 64) for seq_len in (512, 2048)
]


def _build_inputs(*, bs: int, seq_len: int, device: torch.device) -> dict:
    max_reqs = max(bs + 1, 4)
    max_context_len = max(seq_len + 1, 1)
    total_tokens = bs * seq_len

    flat = torch.randint(
        low=0,
        high=1 << 30,
        size=(total_tokens,),
        dtype=torch.int64,
        device=device,
    )
    lens = torch.full((bs,), seq_len, dtype=torch.int64, device=device)
    offsets = torch.zeros(bs + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lens, dim=0)

    req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int64, device=device)
    pool = torch.zeros((max_reqs, max_context_len), dtype=torch.int32, device=device)
    return dict(
        flat_in=flat,
        offsets=offsets,
        req_pool_indices=req_pool_indices,
        pool_out=pool,
    )


@marker.parametrize("bs,seq_len", _BS_SEQ_FULL, _BS_SEQ_CI)
@marker.benchmark("provider", ["triton"])
def benchmark(bs: int, seq_len: int, provider: str):
    inputs = _build_inputs(bs=bs, seq_len=seq_len, device=torch.device(DEFAULT_DEVICE))
    return marker.do_bench(
        lambda: launch_scatter_req_token_ids_kernel(**inputs),
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
